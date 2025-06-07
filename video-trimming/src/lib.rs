use ff_ext::GoldilocksExt2;
use p3_field_git::PrimeCharacteristicRing;
use p3_goldilocks_git::Goldilocks as CenoGoldilocks;

use ark_std::log2;
use p3_field::{extension::BinomialExtensionField, AbstractField, Field, TwoAdicField};
use p3_goldilocks::Goldilocks;
use rayon::prelude::*;
use rs_merkle::{algorithms::Sha256, Hasher, MerkleProof, MerkleTree};

type F = Goldilocks;
type E = GoldilocksExt2;
type EF = BinomialExtensionField<F, 2>;

pub fn barycentric_evaluation(mut image: Vec<u8>, eval_point: EF) -> EF {
    // Using formula at the bottom of this page: https://hackmd.io/@vbuterin/barycentric_evaluation
    // Degree is N in that formula
    let degree = image.len().next_power_of_two();
    // while image.len() < degree {
    //     image.push(0_u8);
    // }
    let degree_bits = log2(degree);
    let omega = EF::two_adic_generator(degree_bits as usize);
    // Will contain powers of omega needed for sum
    let mut omega_acc = EF::one();
    // Running sum:
    let mut sum = EF::zero();
    static MONTGOMERY_STEP_SIZE: usize = 256;
    let mut numerators = vec![EF::zero(); MONTGOMERY_STEP_SIZE];
    let mut denominators = vec![EF::zero(); MONTGOMERY_STEP_SIZE];
    let mut montgomery_partial_prods = vec![EF::zero(); MONTGOMERY_STEP_SIZE];
    for i in (0..image.len()).step_by(MONTGOMERY_STEP_SIZE) {
        for j in 0..MONTGOMERY_STEP_SIZE {
            numerators[j] = EF::from_canonical_u8(image[i + j]) * omega_acc;
            denominators[j] = eval_point - omega_acc;
            montgomery_partial_prods[j] = if j == 0 {
                denominators[j]
            } else {
                denominators[j] * montgomery_partial_prods[j - 1]
            };
            omega_acc *= omega;
        }

        let mut last_inverse = montgomery_partial_prods[MONTGOMERY_STEP_SIZE - 1].inverse();
        for j in (1..MONTGOMERY_STEP_SIZE).rev() {
            let inverted_denominator = montgomery_partial_prods[j - 1] * last_inverse;
            sum += inverted_denominator * numerators[j];
            last_inverse *= denominators[j];
        }
        sum += last_inverse * numerators[0];
    }
    let front_quantity =
        (eval_point.exp_u64(degree as u64) - EF::one()) / EF::from_canonical_u64(degree as u64);

    front_quantity * sum
}

pub fn horners_evaluation(mut image: Vec<u8>, eval_point: EF) -> EF {
    let mut sum = EF::zero();
    for i in (0..image.len()) {
        sum *= eval_point;
        sum += EF::from_canonical_u8(image[i]);
    }

    sum
}

pub fn random_felt(x1: u64, x2: u64) -> EF {
    EF::new(
        Goldilocks::from_canonical_u64(x1),
        Goldilocks::from_canonical_u64(x2),
    )
}

pub fn random_felt_ceno(x1: u64) -> E {
    E::from_u64(x1)
}

pub fn image_merkle_tree(image: Vec<u8>, chunk_size: usize) -> MerkleTree<Sha256> {
    let mut leaves = Vec::new();

    // Process image in 32-byte chunks
    for chunk in image.chunks(chunk_size) {
        leaves.push(Sha256::hash(chunk));
    }

    // Construct and return the Merkle tree
    MerkleTree::<Sha256>::from_leaves(&leaves)
}

pub fn compute_video_mle_evaluations_vec(
    pixels: &[u64],
    frame_size: usize,
    num_frames: usize,
) -> Vec<CenoGoldilocks> {
    assert!(
        frame_size * num_frames == pixels.len(),
        "Pixels is of an unexpected size!"
    );
    let frame_size_padded = frame_size.next_power_of_two();
    let num_frames_padded = num_frames.next_power_of_two();
    let video_size = num_frames * frame_size;
    let poly_size = num_frames_padded * frame_size_padded;

    let mut evaluations = Vec::with_capacity(poly_size);
    for i in 0..num_frames.next_power_of_two() {
        for j in 0..frame_size.next_power_of_two() {
            if i < num_frames && j < frame_size {
                evaluations.push(CenoGoldilocks::from_u64(pixels[i * frame_size + j]));
            } else {
                evaluations.push(CenoGoldilocks::from_u64(0));
            }
        }
    }
    evaluations
}

pub fn to_binary_vec(frame_num: usize, n: usize) -> Vec<u64> {
    (0..n)
        .rev()
        .map(|i| (frame_num >> i) & 1)
        .map(|bit| bit as u64)
        .collect()
}
