use std::marker::PhantomData;

use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_keccak::Keccak256Hash;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};

use ark_std::log2;
use itertools::{izip, Itertools};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, PolynomialSpace};
use p3_dft::Radix2DitParallel;
use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, ExtensionField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, TruncatedPermutation};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

use std::time::Duration;
use std::time::Instant;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

type Dft = Radix2DitParallel;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn seeded_rng() -> impl Rng {
    ChaCha20Rng::seed_from_u64(18)
}

fn main() {
    let setup_and_commit_start = Instant::now();
    // Setup the PCS with basic parameters
    let mut rng = seeded_rng();
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut rng,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());

    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // Use log blowup 1 for now.
    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = MyPcs::new(Dft {}, val_mmcs, fri_config);
    let challenger = Challenger::new(perm.clone());

    let mut p_challenger = challenger.clone();

    // Define image vector/properties.
    let frame_size: usize = 240 * 360;
    let num_frames: usize = 50;
    let video_size = num_frames * frame_size;
    let frame_index = 24; // 0-indexed
    let log_n = log2(video_size);

    let mut pixels: Vec<u32> = (0..video_size).map(|_| rng.next_u64() as u32).collect();
    while pixels.len() < (1 << log_n) {
        pixels.push(0);
    }
    let pixel_felts = pixels.iter().map(|x| BabyBear::new(*x)).collect::<Vec<_>>();
    let pixel_felts_clone = pixel_felts.clone();

    // Setup for commitment to polynomial of image
    let d = <MyPcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(
        &pcs,
        1 << log_n,
    );
    // Coefficients are just the pixels since we are using horner's rule.
    let coeffs = RowMajorMatrix::new(pixel_felts, 1);
    let dft = Dft::default();

    let evals = dft.dft_batch(coeffs).to_row_major_matrix();

    let (comm, data) =
        <MyPcs as p3_commit::Pcs<Challenge, Challenger>>::commit(&pcs, vec![(d, evals)]);
    let commit_duration = setup_and_commit_start.elapsed();
    println!("Setup/Gen commitment took: {:?}", commit_duration);

    let opening_proof_start = Instant::now();
    let zeta: Challenge = rng.gen();

    let f_size = frame_size;
    let mut frame_felts = Vec::new();
    for e in &pixel_felts_clone[frame_size * frame_index..frame_size * (frame_index + 1)] {
        frame_felts.push(*e);
    }

    let r_size = frame_size * frame_index;
    let r_degree = r_size.next_power_of_two();
    let mut r_coeffs_vec = Vec::new();
    for i in 0..(r_degree - r_size) {
        r_coeffs_vec.push(BabyBear::new(0));
    }
    for e in &pixel_felts_clone[0..frame_size * frame_index] {
        r_coeffs_vec.push(*e);
    }
    let r_coeffs = RowMajorMatrix::new(r_coeffs_vec, 1);
    let r_evals = dft.dft_batch(r_coeffs).to_row_major_matrix();
    let r_domain =
        <MyPcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, r_degree);

    let q_size = pixel_felts_clone.len() - frame_size * (frame_index + 1);
    let q_degree = q_size.next_power_of_two();
    let mut q_coeffs_vec = Vec::new();
    for i in 0..(q_degree - q_size) {
        q_coeffs_vec.push(BabyBear::new(0));
    }
    for e in &pixel_felts_clone[frame_size * (frame_index + 1)..pixel_felts_clone.len()] {
        q_coeffs_vec.push(*e);
    }
    let q_coeffs = RowMajorMatrix::new(q_coeffs_vec, 1);
    let q_evals = dft.dft_batch(q_coeffs).to_row_major_matrix();
    let q_domain =
        <MyPcs as p3_commit::Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, q_degree);

    let (trim_comm, trim_data) = <MyPcs as p3_commit::Pcs<Challenge, Challenger>>::commit(
        &pcs,
        vec![(r_domain, r_evals), (q_domain, q_evals)],
    );

    let (values, proof) = pcs.open(
        vec![
            (&data, vec![vec![zeta]]),
            (&trim_data, vec![vec![zeta], vec![zeta]]),
        ],
        &mut p_challenger,
    );
    let opening_proof_duration = opening_proof_start.elapsed();
    println!(
        "Generating trimming proof took: {:?}",
        opening_proof_duration
    );

    let verification_start = Instant::now();
    let mut v_challenger = challenger.clone();
    let result = pcs
        .verify(
            vec![
                (comm, vec![(d, vec![(zeta, values[0][0][0].clone())])]),
                (
                    trim_comm,
                    vec![
                        (r_domain, vec![(zeta, values[1][0][0].clone())]),
                        (q_domain, vec![(zeta, values[1][1][0].clone())]),
                    ],
                ),
            ],
            &proof,
            &mut v_challenger,
        )
        .unwrap();
    println!("verification result was: {:?}", result);
    println!("Values from commitment are: {:?}", values);

    let frame_eval_start = Instant::now();
    // Correct approach
    let mut frame_eval = Challenge::zero();
    for i in frame_felts.iter().rev() {
        frame_eval *= zeta;
        frame_eval += *i;
    }
    let frame_eval_duration = frame_eval_start.elapsed();
    println!("frame eval took: {:?}", frame_eval_duration);
    println!("Frame eval is: {:?}", frame_eval);
    // To pad q,r to polynomials of degree 2^n, we multiply by a correct power of x.
    // This allows us to convert the range check for degrees of powers of 2 in FRI to one for arbitrary degrees.
    let r_padding_degree = r_degree - r_size;
    let r_extra_pow = zeta.exp_u64(r_padding_degree as u64);
    let r_proof_value = values[1][0][0][0].clone();
    let r_actual = r_proof_value / r_extra_pow;

    let q_padding_degree = q_degree - q_size;
    let q_extra_pow = zeta.exp_u64(q_padding_degree as u64);
    let q_proof_value = values[1][1][0][0].clone();
    let q_actual = q_proof_value / q_extra_pow;

    let q_shift_factor = zeta.exp_u64((frame_size * (frame_index + 1)) as u64);
    let f_shift_factor = zeta.exp_u64((frame_size * frame_index) as u64);

    let overall_eval_value = values[0][0][0][0];

    let inferred_eval_value = q_actual * q_shift_factor + frame_eval * f_shift_factor + r_actual;
    println!("Overall eval value is: {:?}", overall_eval_value);
    println!("Inferred eval value is: {:?}", inferred_eval_value);

    let verification_duration = verification_start.elapsed();
    println!("Verification took: {:?}", verification_duration);
}
