// This file benchmarks the work the a signer would need to do to validate an outsourced signature.
// In our setup, these are the different options for steps a camera would be doing.
// We don't include the time to sign a value, we just include the time to hash/or validate that a
// PCS is correctly produced.
// We benchmark the following approaches:
// - Barycentric Evaluation of the video at a random point (Clever way of evaluating video's polynomial at a given point)
// - Multilinear extension of the video (default way of doing this for a multilinear PCS, not super memory friendly)
// - "Horners rule" evaluation (our more optimized univariate implementation)
// - Plonky3 Poseidon-based Merkle Tree
// - IFFT approach (dumb nlogn/linear memory cost way of doing univariate interpolation)

use ark_std::log2;
use blstrs::Scalar as Fr;
use ff_ext::GoldilocksExt2;
use multilinear_extensions::mle::ArcMultilinearExtension;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_matrix_git::dense::RowMajorMatrix as P3RowMajorMatrix;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, TruncatedPermutation};
use witness::{InstancePaddingStrategy, RowMajorMatrix};

use std::time::Instant;
use video_trimming::*;

// use rand::{Rng, RngCore, SeedableRng};
use ark_std::rand::{Rng, RngCore, SeedableRng};
// use rand_chacha::ChaCha20Rng;
use ark_std::rand::rngs::StdRng; // or keep ChaCha20Rng if preferred
use serde::Serialize;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Val = BabyBear;

fn seeded_rng() -> impl Rng {
    StdRng::seed_from_u64(18)
}

#[derive(Serialize)]
struct ExperimentResult {
    log_pixels: usize,
    // All times are in seconds.
    naive_merkle_time: f64,
    merkle_time: f64,
    poseidon_time: f64,
    barycentric_time: f64,
    mle_time: f64,
    horners_time: f64,
}

fn main() {
    let mut rng = seeded_rng();

    // Total sizes to benchmark: [18, 21, 24, 27, 30]
    let LOG_FRAME_SIZES_FOR_BENCHMARK = vec![15, 18, 21, 21, 21];
    let LOG_FRAME_COUNTS_FOR_BENCHMARK = vec![3, 3, 3, 6, 9];

    for i in (0..LOG_FRAME_COUNTS_FOR_BENCHMARK.len()) {
        let log_frame_size = LOG_FRAME_SIZES_FOR_BENCHMARK[i];
        let frame_size = 1 << log_frame_size;
        let log_frame_count = LOG_FRAME_COUNTS_FOR_BENCHMARK[i];
        let frame_count = 1 << log_frame_count;

        let pixels = frame_size * frame_count;

        let image: Vec<u8> = (0..pixels).map(|_| rng.next_u32() as u8).collect();

        let eval_point = random_felt(rng.next_u64(), rng.next_u64());

        // Merkle Tree approach (dumb)
        let image_clone = image.clone();
        let start = Instant::now();
        let result1 = image_merkle_tree(image_clone, 32);
        let duration = start.elapsed();
        let naive_merkle_time = duration.as_secs_f64();
        // println!("Result from Merklization: {:?}", result1.root());

        // Merkle Tree approach (smart)
        let image_clone = image.clone();
        let start = Instant::now();
        let result1 = image_merkle_tree(image_clone, frame_size);
        let duration = start.elapsed();
        let merkle_time = duration.as_secs_f64();
        // println!("Result from Merklization: {:?}", result1.root());

        // Poseidon Approach (basically same perf as poseidon merkle tree)
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear::default(),
            &mut rng,
        );
        let hash = MyHash::new(perm.clone());
        let mut data_as_felts: Vec<BabyBear> = image
            .clone()
            .into_iter()
            .map(|chunk| BabyBear::new((chunk as u32)))
            .collect();
        let start = Instant::now();
        hash.hash_iter(data_as_felts);
        let duration = start.elapsed();
        let poseidon_time = duration.as_secs_f64();
        // println!("Result from Poseidon: {:?}", result);

        // Barycentric Approach (with batching)
        let image_clone = image.clone();
        let start = Instant::now();
        let result = barycentric_evaluation(image_clone, eval_point);
        let duration = start.elapsed();
        let barycentric_time = duration.as_secs_f64();
        // println!("Result from Barycentric: {:?}", result);

        // Multilinear extension approach
        let image_u64_copy = image.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let start = Instant::now();
        let evaluations_vec =
            compute_video_mle_evaluations_vec(&image_u64_copy, frame_size, frame_count);
        let num_vars = log2(evaluations_vec.len());
        let mle_point = (0..num_vars)
            .map(|_| random_felt_ceno(rng.next_u64()))
            .collect::<Vec<_>>();
        let mut rmm_inner = P3RowMajorMatrix::new_col(evaluations_vec);
        let mut rmm =
            RowMajorMatrix::new_by_inner_matrix(rmm_inner, InstancePaddingStrategy::Default);
        let ml_poly: ArcMultilinearExtension<_> = rmm.to_mles::<GoldilocksExt2>().remove(0).into();
        let eval = ml_poly.evaluate(&mle_point);

        let duration = start.elapsed();
        let mle_time = duration.as_secs_f64();
        // println!("Result from MLE Eval: {:?}", eval);

        // Horner's rule evaluation approach.
        let start = Instant::now();
        let result = horners_evaluation(image, eval_point);
        let duration = start.elapsed();
        let horners_time = duration.as_secs_f64();
        // println!("Result from Horners: {:?}", result);

        let result = ExperimentResult {
            log_pixels: log_frame_count + log_frame_size,
            naive_merkle_time,
            merkle_time,
            poseidon_time,
            barycentric_time,
            mle_time,
            horners_time,
        };
        let json = serde_json::to_string_pretty(&result).unwrap();
        // Print to command line
        println!("{}", json);
    }
}
