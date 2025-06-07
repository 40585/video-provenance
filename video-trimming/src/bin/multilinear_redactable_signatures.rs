use ark_std::log2;
use ff_ext::GoldilocksExt2;
use mpcs::{
    test_util::{
        commit_polys_individually, get_point_from_challenge, get_points_from_challenge, setup_pcs,
    },
    Basefold, BasefoldRSParams, Evaluation, PolynomialCommitmentScheme, SecurityLevel,
};
use multilinear_extensions::{
    mle::ArcMultilinearExtension,
    // mle::{DenseMultilinearExtension, MultilinearExtension},
};
use p3_field_git::{extension::BinomialExtensionField, PrimeCharacteristicRing};
use p3_goldilocks_git::Goldilocks as CenoGoldilocks;
use p3_matrix_git::dense::RowMajorMatrix as P3RowMajorMatrix;
use rand::rngs::OsRng;
use rand::Rng;
use rand::RngCore;
use std::collections::BTreeMap;
use std::time::Duration;
use std::time::Instant;
use transcript::{BasicTranscript, Transcript};
use video_trimming::{compute_video_mle_evaluations_vec, to_binary_vec};
use witness::{InstancePaddingStrategy, RowMajorMatrix};

type PcsGoldilocksRSCode = Basefold<GoldilocksExt2, BasefoldRSParams>;
type T = BasicTranscript<GoldilocksExt2>;
type E = GoldilocksExt2;
type Pcs = PcsGoldilocksRSCode;

fn main() {
    let frame_size: usize = 240 * 360;
    let m = log2(frame_size) as usize;
    let num_frames: usize = 50;
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
    ];

    let (pp, vp) = {
        let param = Pcs::setup(poly_size, SecurityLevel::default()).unwrap();
        Pcs::trim(param, poly_size).unwrap()
    };

    let mut rng = OsRng;
    // let pixels: Vec<u64> = (0..video_size)
    //     .map(|_| rng.try_next_u64().unwrap())
    //     .collect();
    // let mut rng = rand::rng();
    let pixels: Vec<u64> = (0..video_size).map(|_| rng.next_u64()).collect();

    // Generate video polynomial:
    let mut evaluations = compute_video_mle_evaluations_vec(&pixels, frame_size, num_frames);
    let mut pixel_rmm_inner = P3RowMajorMatrix::new_col(evaluations);
    let mut pixel_rmm =
        RowMajorMatrix::new_by_inner_matrix(pixel_rmm_inner, InstancePaddingStrategy::Default);
    let rmms = BTreeMap::from([(0, pixel_rmm.clone())]);

    let commit_start = Instant::now();
    let poly: ArcMultilinearExtension<_> = pixel_rmm.to_mles().remove(0).into();
    let mut transcript = T::new(b"BaseFold");

    let comm = Pcs::batch_commit_and_write(&pp, rmms, &mut transcript).unwrap();
    let commit_duration = commit_start.elapsed();
    println!("Gen commitment took: {:?}", commit_duration);
    let mut evals = Vec::new();
    let mut points = Vec::new();
    let mut proofs = Vec::new();

    let num_instances = vec![(0, 1 << (m + n))];
    let circuit_num_polys = vec![(1, 0)]; // batch size of 1 at the moment

    let opening_proof_start = Instant::now();
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
        let block_size_felts = frame_size * block_size;
        let mut unpadded_block_contents = Vec::with_capacity(block_size_felts.next_power_of_two());
        for i in 0..block_size_felts {
            unpadded_block_contents.push(pixels[block_num * block_size * frame_size + i]);
        }
        let mut block_contents =
            compute_video_mle_evaluations_vec(&unpadded_block_contents, frame_size, block_size);

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
}
