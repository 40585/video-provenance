use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead};
use std::marker::PhantomData;
use std::path::Path;

use ark_std::log2;
use itertools::{izip, Itertools};
use multilinear_extensions::mle::ArcMultilinearExtension;
use p3_matrix_git::dense::RowMajorMatrix as P3RowMajorMatrix;
// use rand::distributions::{Distribution, Standard};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
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
    let m = log2(frame_size / 2) as usize; // goldilocks can store 2 pixels per felt!
    let num_frames = 47;
    let n = log2(num_frames) as usize;
    let num_vars = m + n;
    let poly_size = 1 << num_vars;
    let video_size = num_frames * frame_size;

    // List of "compact ranges" (as in Meiklejohn paper) to redact.
    // Tuples of form block_size, block_num (block_num is at that resolution, so there would be 50 blocks of size 1, 25 of size 2, etc. when there are 50 frames)
    let ranges_to_redact = vec![
        (1, 9),  // Frame 9
        (2, 5),  // Frames 10-11
        (4, 3),  // Frames 12-15
        (16, 1), // Frames 16-31
        (8, 4),  // Frames 32-39
        (1, 40), // Frame 40
    ]; // Set to (1,9) for a single frame
    let num_ranges = ranges_to_redact.len();

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
    let mut signer_u64s: Vec<u64> = sorted_frames
        .chunks_exact(6)
        .map(|chunk| {
            (chunk[0] as u64)
                | ((chunk[1] as u64) << 8)
                | ((chunk[2] as u64) << 16)
                | ((chunk[3] as u64) << 24)
                | ((chunk[4] as u64) << 32)
                | ((chunk[5] as u64) << 40)
        })
        .collect();
    // Factor in the /2 in the pixel-felt frame packing as part of frame_size/2
    let mut evaluations =
        compute_video_mle_evaluations_vec(&signer_u64s, frame_size / 2, num_frames);
    let mut pixel_rmm_inner = P3RowMajorMatrix::new_col(evaluations);
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
    let poly: ArcMultilinearExtension<_> = pixel_rmm.to_mles().remove(0).into();
    let mut transcript = T::new(b"BaseFold");

    let comm = Pcs::batch_commit_and_write(&pp, rmms, &mut transcript).unwrap();
    let commit_duration = commit_start.elapsed();
    println!("Gen commitment took: {:?}", commit_duration);

    let opening_proof_start = Instant::now();

    let mut evals = Vec::new();
    let mut points = Vec::new();
    let mut proofs = Vec::new();

    let num_instances = vec![(0, 1 << (m + n))];
    let circuit_num_polys = vec![(1, 0)]; // batch size of 1 at the moment

    for (block_size, block_num) in ranges_to_redact.clone() {
        let log_block_size = log2(block_size) as usize;

        let mut point = get_point_from_challenge(num_vars, &mut transcript);
        let index_vars_length = n - log_block_size;
        let block_num_binary = to_binary_vec(block_num, n - log_block_size);
        for i in 0..index_vars_length {
            point[num_vars - 1 - i] = E::from_u64(block_num_binary[i]);
        }
        let eval = vec![poly.evaluate(point.as_slice())];
        evals.push(eval.clone());
        transcript.append_field_element_ext(&eval[0]);
        points.push(point.clone());

        let proof = Pcs::batch_open(
            &pp,
            &num_instances,
            None,
            &comm,
            &[point],        // as vec
            &[eval.clone()], // as vec
            &circuit_num_polys,
            &mut transcript,
        )
        .unwrap();
        proofs.push(proof);
    }
    let opening_proof_duration = opening_proof_start.elapsed();
    println!("Gen opening proof took: {:?}", opening_proof_duration);

    // Verify
    let verify_start = Instant::now();
    let mut transcript = T::new(b"BaseFold");
    let comm = Pcs::get_pure_commitment(&comm);
    Pcs::write_commitment(&comm, &mut transcript).unwrap();

    let mut idx = 0;
    for (block_size, block_num) in ranges_to_redact {
        let log_block_size = log2(block_size) as usize;
        let block_size_felts = (frame_size / 2) * block_size;
        let mut unpadded_block_contents = Vec::with_capacity(block_size_felts.next_power_of_two());
        for i in 0..block_size_felts {
            unpadded_block_contents
                .push(signer_u64s[block_num * block_size * (frame_size / 2) + i]);
        }
        let mut block_contents = compute_video_mle_evaluations_vec(
            &unpadded_block_contents,
            (frame_size / 2),
            block_size,
        );

        let mut point = get_point_from_challenge(num_vars, &mut transcript);
        let index_vars_length = n - log_block_size;
        let block_num_binary = to_binary_vec(block_num, n - log_block_size); // number of bits that are the index of the block vs number of bits that are random.
        for i in 0..index_vars_length {
            point[num_vars - 1 - i] = E::from_u64(block_num_binary[i]);
        }
        transcript.append_field_element_ext(&evals[idx][0]);
        Pcs::batch_verify(
            &vp,
            &num_instances,
            &[point.clone()],
            None,
            &comm,
            &[evals[idx].clone()],
            &proofs[idx],
            &circuit_num_polys,
            &mut transcript,
        )
        .unwrap();

        let mut frame_rmm_inner = P3RowMajorMatrix::new_col(block_contents);
        let mut frame_rmm =
            RowMajorMatrix::new_by_inner_matrix(frame_rmm_inner, InstancePaddingStrategy::Default);
        let frame_poly: ArcMultilinearExtension<_> =
            frame_rmm.to_mles::<GoldilocksExt2>().remove(0).into();
        let eval2 = frame_poly.evaluate(&point.as_slice()[0..(m + log_block_size)]);
        println!("Eval of polynomial the regular way is: {:?}", evals[idx]);
        println!("Eval of polynomial via interpolation is: {:?}", eval2);
        idx += 1;
    }

    let verify_duration = verify_start.elapsed();
    println!("Verification of opening took: {:?}", verify_duration);
    Ok(())
}
