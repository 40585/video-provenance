// Register eq(x,r) polynomial
// Register v(x) polynomial (mask).
// Register m(x) polynomial (message).
// Register r(x) polynomial (revealed).
// Zero-check m(x)*v(x) - r(x) over boolean hypercube
// Equivalent to (m(x)*v(x) - r(x))*eq(r,x) over boolean hypercube. Use one variant of the build_eq_x_r function

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead};
use std::marker::PhantomData;
use std::path::Path;

use ark_std::log2;
use either::Either;
use itertools::{izip, Itertools};
use multilinear_extensions::{
    mle::{ArcMultilinearExtension, Point},
    monomial::Term,
    virtual_poly::{build_eq_x_r, VPAuxInfo, VirtualPolynomial},
    virtual_polys::VirtualPolynomials,
};
use p3_matrix_git::dense::RowMajorMatrix as P3RowMajorMatrix;
// use rand::distributions::{Distribution, Standard};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sumcheck::structs::{IOPProverState, IOPVerifierState};
use witness::{InstancePaddingStrategy, RowMajorMatrix};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use std::time::Duration;
use std::time::Instant;

use ff_ext::GoldilocksExt2;
use mpcs::{
    test_util::{
        commit_polys_individually, get_point_from_challenge, get_points_from_challenge, setup_pcs,
    },
    Basefold, BasefoldRSParams, Evaluation, PolynomialCommitmentScheme, SecurityLevel,
};
use p3_field_git::{extension::BinomialExtensionField, PrimeCharacteristicRing};
use p3_goldilocks_git::Goldilocks as CenoGoldilocks;
use transcript::{BasicTranscript, Transcript};
use video_trimming::{compute_video_mle_evaluations_vec, to_binary_vec};

type PcsGoldilocksRSCode = Basefold<GoldilocksExt2, BasefoldRSParams>;
// type PcsGoldilocksBasecode = Basefold<GoldilocksExt2, BasefoldBasecodeParams>;
type T = BasicTranscript<GoldilocksExt2>;
type E = GoldilocksExt2;
type Pcs = PcsGoldilocksRSCode;

fn parse_filename(file_name: &str) -> Option<(u32, char)> {
    let parts: Vec<&str> = file_name.trim_end_matches(".txt").split('_').collect();
    if parts.len() == 3 {
        if let Ok(frame_number) = parts[1].parse::<u32>() {
            if let Some(channel) = parts[2].chars().next() {
                return Some((frame_number, channel));
            }
        }
    }
    None
}

fn read_file_as_vec<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        values.extend(
            line.split_whitespace()
                .filter_map(|num| num.parse::<u8>().ok()),
        );
    }

    Ok(values)
}

fn seeded_rng() -> impl Rng {
    ChaCha20Rng::seed_from_u64(18)
}

fn main() -> io::Result<()> {
    let file_io_start = Instant::now();
    // Define image vector/properties.
    let frame_size: usize = (240 * 320);
    let num_frames = 47;
    let video_size = num_frames * frame_size / 2;
    let num_vars = log2(video_size) as usize; // goldilocks can store 2 pixels per felt!
    let poly_size = 1 << num_vars;

    // Python style indexing where end isn't inclusive. Also FFMPEG decomposes frames in a
    // 1-indexed way.
    let segment_start: usize = 10 - 1;
    let segment_end: usize = 41 - 1;

    // Read full video for the signature part.
    let dir_path = "decomposed_frames";
    let mut signer_frames: BTreeMap<u32, Vec<Vec<u8>>> = BTreeMap::new();

    let entries = fs::read_dir(dir_path)?;

    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name().into_string().unwrap();

        if let Some((frame_number, channel)) = parse_filename(&file_name) {
            let file_path = entry.path();
            let content = read_file_as_vec(&file_path)?;

            let frame = signer_frames
                .entry(frame_number)
                .or_insert_with(|| vec![vec![], vec![], vec![]]);
            match channel {
                'B' => frame[0] = content,
                'G' => frame[1] = content,
                'R' => frame[2] = content,
                _ => (),
            }
        }
    }

    // Note: BTreeMap is already sorted by frame count.
    let sorted_frames = signer_frames
        .into_values()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();
    let mut video_evaluations = sorted_frames
        .chunks_exact(6)
        .map(|chunk| {
            (chunk[0] as u64)
                | ((chunk[1] as u64) << 8)
                | ((chunk[2] as u64) << 16)
                | ((chunk[3] as u64) << 24)
                | ((chunk[4] as u64) << 32)
                | ((chunk[5] as u64) << 40)
        })
        .map(|x| CenoGoldilocks::from_u64(x))
        .collect::<Vec<CenoGoldilocks>>();
    let mut pixel_rmm_inner = P3RowMajorMatrix::new_col(video_evaluations.clone());
    let mut pixel_rmm =
        RowMajorMatrix::new_by_inner_matrix(pixel_rmm_inner, InstancePaddingStrategy::Default);
    let rmms = BTreeMap::from([(0, pixel_rmm.clone())]);

    let file_io_duration = file_io_start.elapsed();
    println!(
        "Signer reading and setting up video took: {:?}",
        file_io_duration
    );

    let (pp, vp) = {
        let param = Pcs::setup(poly_size, SecurityLevel::default()).unwrap();
        Pcs::trim(param, poly_size).unwrap()
    };

    let commit_start = Instant::now();
    let poly: ArcMultilinearExtension<_> = pixel_rmm.to_mles::<E>().remove(0).into();
    let mut pcs_transcript = T::new(b"BaseFold");

    let comm = Pcs::batch_commit_and_write(&pp, rmms, &mut pcs_transcript).unwrap();
    let commit_duration = commit_start.elapsed();
    println!("Gen commitment took: {:?}", commit_duration);

    let opening_proof_start = Instant::now();

    let mut point = get_point_from_challenge(num_vars, &mut pcs_transcript);
    let eq_x_r = build_eq_x_r(&point);

    let start_index = segment_start * (frame_size / 2);
    let end_index = segment_end * (frame_size / 2);
    let mask_poly_vec = (0..video_size)
        .map(|i| {
            if i >= start_index && i < end_index {
                CenoGoldilocks::ONE
            } else {
                CenoGoldilocks::ZERO
            }
        })
        .collect::<Vec<_>>();
    let mut mask_rmm_inner = P3RowMajorMatrix::new_col(mask_poly_vec);
    let mut mask_rmm =
        RowMajorMatrix::new_by_inner_matrix(mask_rmm_inner, InstancePaddingStrategy::Default);
    let mask_poly: ArcMultilinearExtension<_> = mask_rmm.to_mles::<E>().remove(0).into();

    // We are sumchecking (m(x)*v(x) - r(x))*eq(r,x) = 0.
    // What they call virtual polys is really just a sum of different products of multilinear polynomials.
    let revealed_poly_vec = (0..video_size)
        .map(|i| {
            if i >= start_index && i < end_index {
                CenoGoldilocks::from(video_evaluations[i])
            } else {
                CenoGoldilocks::ZERO
            }
        })
        .collect::<Vec<_>>();
    let mut revealed_rmm_inner = P3RowMajorMatrix::new_col(revealed_poly_vec);
    let mut revealed_rmm =
        RowMajorMatrix::new_by_inner_matrix(revealed_rmm_inner, InstancePaddingStrategy::Default);
    let revealed_poly: ArcMultilinearExtension<_> = revealed_rmm.to_mles::<E>().remove(0).into();
    // let sumcheck_vp_2 =
    //     VirtualPolynomial::<E>::new_from_product(vec![revealed_poly, eq_x_r], -E::ONE);
    let sumcheck_poly = VirtualPolynomials::<E>::new_from_monimials(
        8,
        num_vars,
        vec![
            Term {
                scalar: either::Right(E::ONE),
                product: vec![
                    either::Left(&poly),
                    either::Left(&mask_poly),
                    either::Left(&eq_x_r),
                ],
            },
            Term {
                scalar: either::Right(-E::ONE),
                product: vec![either::Left(&revealed_poly), either::Left(&eq_x_r)],
            },
        ],
    );

    let num_instances = vec![(0, 1 << num_vars)];
    let circuit_num_polys = vec![(1, 0)]; // batch size of 1 at the moment
    let mut sumcheck_transcript = T::new(b"BaseFold");
    let (proof, _) = IOPProverState::<E>::prove(sumcheck_poly.as_view(), &mut sumcheck_transcript);
    let eval = poly.evaluate(&proof.point);

    let pcs_proof = Pcs::batch_open(
        &pp,
        &num_instances,
        None,
        &comm,
        &[proof.point.clone()], // as vec
        &[vec![eval.clone()]],  // as vec
        &circuit_num_polys,
        &mut pcs_transcript,
    )
    .unwrap();

    let opening_proof_duration = opening_proof_start.elapsed();
    println!("Gen opening proof took: {:?}", opening_proof_duration);

    // Verify
    let verify_start = Instant::now();
    let mut pcs_transcript = T::new(b"BaseFold");
    let comm = Pcs::get_pure_commitment(&comm);
    Pcs::write_commitment(&comm, &mut pcs_transcript).unwrap();
    let mut point = get_point_from_challenge(num_vars, &mut pcs_transcript);

    let mut sumcheck_transcript = T::new(b"BaseFold");
    let subclaim = IOPVerifierState::<E>::verify(
        E::ZERO,
        &proof,
        &VPAuxInfo {
            max_degree: 3,
            max_num_variables: num_vars,
            ..Default::default()
        },
        &mut sumcheck_transcript,
    );
    let sumcheck_r: Point<E> = subclaim
        .point
        .iter()
        .map(|c| c.elements)
        .collect::<Vec<_>>();

    Pcs::batch_verify(
        &vp,
        &num_instances,
        &[sumcheck_r.clone()],
        None,
        &comm,
        &[vec![eval]],
        &pcs_proof,
        &circuit_num_polys,
        &mut pcs_transcript,
    );

    let eq_x_r = build_eq_x_r(&point);
    let eq_x_r_eval = eq_x_r.evaluate(&sumcheck_r);

    let start_index = segment_start * (frame_size / 2);
    let end_index = segment_end * (frame_size / 2);
    let mask_poly_vec = (0..video_size)
        .map(|i| {
            if i >= start_index && i < end_index {
                CenoGoldilocks::ONE
            } else {
                CenoGoldilocks::ZERO
            }
        })
        .collect::<Vec<_>>();
    let mut mask_rmm_inner = P3RowMajorMatrix::new_col(mask_poly_vec);
    let mut mask_rmm =
        RowMajorMatrix::new_by_inner_matrix(mask_rmm_inner, InstancePaddingStrategy::Default);
    let mask_poly: ArcMultilinearExtension<_> = mask_rmm.to_mles::<E>().remove(0).into();
    let mask_poly_eval = mask_poly.evaluate(&sumcheck_r);

    let revealed_poly_vec = (0..video_size)
        .map(|i| {
            if i >= start_index && i < end_index {
                CenoGoldilocks::from(video_evaluations[i])
            } else {
                CenoGoldilocks::ZERO
            }
        })
        .collect::<Vec<_>>();
    let mut revealed_rmm_inner = P3RowMajorMatrix::new_col(revealed_poly_vec);
    let mut revealed_rmm =
        RowMajorMatrix::new_by_inner_matrix(revealed_rmm_inner, InstancePaddingStrategy::Default);
    let revealed_poly: ArcMultilinearExtension<_> = revealed_rmm.to_mles::<E>().remove(0).into();
    let revealed_poly_eval = revealed_poly.evaluate(&sumcheck_r);

    println!(
        "Inferred eval {:?}",
        eval * mask_poly_eval * eq_x_r_eval - revealed_poly_eval * eq_x_r_eval
    );

    let verify_duration = verify_start.elapsed();
    println!("Verification of opening took: {:?}", verify_duration);
    Ok(())
}
