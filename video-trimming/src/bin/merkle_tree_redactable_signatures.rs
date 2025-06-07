// This is a "baseline" for SNARK-friendly redactable signatures.
// We test to see what the

use itertools::Itertools;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_commit::Mmcs;
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{
    CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
};
use rand::thread_rng;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Duration;
use std::time::Instant;

use p3_merkle_tree::FieldMerkleTreeMmcs;

type F = BabyBear;

type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyMmcs =
    FieldMerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;

fn seeded_rng() -> impl Rng {
    ChaCha20Rng::seed_from_u64(18)
}

fn main() {
    // Setup the PCS with basic parameters
    let mut rng = seeded_rng();
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut rng,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(hash, compress);

    static FRAME_SIZE: usize = 360 * 240;
    static FRAME_COUNT: usize = 50;
    static PIXELS: usize = FRAME_SIZE * FRAME_COUNT;
    let start_frame = 10;
    let end_frame = 40;

    let video: Vec<u8> = (0..PIXELS).map(|_| rng.next_u32() as u8).collect();
    // let image_copy = image.clone();
    let mut video_as_felts: Vec<BabyBear> = video
        .into_iter()
        .map(|chunk| BabyBear::new((chunk as u32)))
        .collect();

    let video_matrix = RowMajorMatrix::new(video_as_felts, FRAME_SIZE);

    let commit_start = Instant::now();
    let (commit, prover_data) = mmcs.commit(vec![video_matrix]);
    let commit_duration = commit_start.elapsed();
    println!("Merkle tree construction took: {:?}", commit_duration);

    let proof_generation_start = Instant::now();
    let mut value_proof_pairs = Vec::new();
    for frame_num in start_frame..=end_frame {
        value_proof_pairs.push(mmcs.open_batch(frame_num, &prover_data));
    }
    let proof_duration = proof_generation_start.elapsed();
    println!("Proof generation took: {:?}", proof_duration);

    let video_matrix_dims = vec![Dimensions {
        height: FRAME_COUNT,
        width: FRAME_SIZE,
    }];

    let proof_verification_start = Instant::now();
    for frame_num in start_frame..=end_frame {
        let (opened_values, proof) = &value_proof_pairs[frame_num - start_frame];
        mmcs.verify_batch(
            &commit,
            &video_matrix_dims,
            frame_num,
            &opened_values,
            &proof,
        )
        .expect("expected verification to succeed");
    }
    let verification_duration = proof_verification_start.elapsed();
    println!("Proof verification took: {:?}", verification_duration);
}
