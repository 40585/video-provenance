use ark_bls12_377::Fr;
use ark_bls12_381::Fr as Fr381;
use ark_crypto_primitives::{
    crh::{sha256::Sha256, CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
    sponge::{
        poseidon::{PoseidonConfig, PoseidonSponge},
        CryptographicSponge,
    },
};
use ark_ff::{Field, PrimeField};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
use ark_poly_commit::{
    linear_codes::{LigeroPCParams, LinearCodePCS, UnivariateLigero},
    LabeledPolynomial, PolynomialCommitment,
};
use ark_std::{test_rng, UniformRand};
use blake2::Blake2s256;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use std::time::Duration;
use std::time::Instant;

use ark_pcs_bench_templates::{FieldToBytesColHasher, LeafIdentityHasher};

type LeafH = LeafIdentityHasher;
type CompressH = Sha256;
type ColHasher<F, D> = FieldToBytesColHasher<F, D>;

struct MerkleTreeParams;

impl Config for MerkleTreeParams {
    type Leaf = Vec<u8>;

    type LeafDigest = <LeafH as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

    type LeafHash = LeafH;
    type TwoToOneHash = CompressH;
}

type MTConfig = MerkleTreeParams;
type LigeroPCS = LinearCodePCS<
    UnivariateLigero<Fr, MTConfig, DensePolynomial<Fr>, ColHasher<Fr, Blake2s256>>,
    Fr,
    DensePolynomial<Fr>,
    MTConfig,
    ColHasher<Fr, Blake2s256>,
>;

type LigeroPcsF<F> = LinearCodePCS<
    UnivariateLigero<F, MTConfig, DensePolynomial<F>, ColHasher<F, Blake2s256>>,
    F,
    DensePolynomial<F>,
    MTConfig,
    ColHasher<F, Blake2s256>,
>;

fn test_sponge<F: PrimeField>() -> PoseidonSponge<F> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut v = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        v.push(res);
    }
    let config = PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, v, 2, 1);
    PoseidonSponge::new(&config)
}

fn rand_poly<Fr: PrimeField>(
    degree: usize,
    _: Option<usize>,
    rng: &mut ChaCha20Rng,
) -> DensePolynomial<Fr> {
    DensePolynomial::rand(degree, rng)
}

// This works as follows:
// P(x) is a polynomial where the coefficients are the pixels of the video ordered by frame/position in the video.
// P_{fi} is the polynomial whose coefficients are just the pixels of frame i. It should be a subinterval of the coefficients of P(x)
// q(x), r(x) must be subject to certain degree bounds (deg(q(x)) < nm - im + 1) and (deg(r(x)) < im)
//  P(x) = q(x)* x^{(i+1)m} + P_{fi}(x)*x^{im} + r(x)
// Ligero/Fri Commitments let us check that a given polynomial is of degree < 2^n for some n.
// To check degree bound for not power of 2, we require the prover to commit to P_{fi}(x) * x^P, where P = i.next_power_of_2() - i, and then divide the commitment value by x^P in evaluation.
// TODO: is this technique cryptographically secure? I think it works but I am bad at rational functions.
fn main() {
    // Video size constants
    let frame_size: usize = 240 * 360;
    let num_frames: usize = 50;
    let video_size = frame_size * num_frames;
    let degree = video_size.next_power_of_two();
    let frame_index = 24; // 0-indexed

    let setup_and_commit_start = Instant::now();
    let mut rng = &mut test_rng();
    // just to make sure we have the right degree given the FFT domain for our field
    let leaf_hash_param = <LeafH as CRHScheme>::setup(&mut rng).unwrap();
    let two_to_one_hash_param = <CompressH as TwoToOneCRHScheme>::setup(&mut rng)
        .unwrap()
        .clone();
    let col_hash_params = <ColHasher<Fr, Blake2s256> as CRHScheme>::setup(&mut rng).unwrap();
    let check_well_formedness = true;

    let pp: LigeroPCParams<Fr, MTConfig, ColHasher<_, _>> = LigeroPCParams::new(
        128,
        4,
        check_well_formedness,
        leaf_hash_param,
        two_to_one_hash_param,
        col_hash_params,
    );

    let (ck, vk) = LigeroPCS::trim(&pp, 0, 0, None).unwrap();

    let rand_chacha = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let video_polynomial = rand_poly(degree, None, rand_chacha);
    let video_polynomial_copy = video_polynomial.clone();
    let labeled_poly = LabeledPolynomial::new("test".to_string(), video_polynomial, None, None);

    let mut test_sponge = test_sponge::<Fr>();
    let (c, rands) = LigeroPCS::commit(&ck, &[labeled_poly.clone()], None).unwrap();
    let commit_duration = setup_and_commit_start.elapsed();
    println!("Setup/Gen commitment took: {:?}", commit_duration);

    let opening_proof_start = Instant::now();
    let point = Fr::rand(rand_chacha);
    let value = labeled_poly.evaluate(&point);

    // Overall opening proof
    let proof = LigeroPCS::open(
        &ck,
        &[labeled_poly],
        &c,
        &point,
        &mut (test_sponge.clone()),
        &rands,
        None,
    )
    .unwrap();

    // Generate q,r,f and openings
    let f_coeffs = &video_polynomial_copy[frame_size * frame_index..frame_size * (frame_index + 1)];
    let r_coeffs = &video_polynomial_copy[0..frame_size * frame_index];
    let q_coeffs =
        &video_polynomial_copy[frame_size * (frame_index + 1)..video_polynomial_copy.len()];

    let f_degree = f_coeffs.len().next_power_of_two();
    let r_degree = r_coeffs.len().next_power_of_two();
    let q_degree = q_coeffs.len().next_power_of_two();

    let f_polynomial = DensePolynomial::from_coefficients_slice(f_coeffs);
    let r_polynomial = DensePolynomial::from_coefficients_slice(r_coeffs);
    let q_polynomial = DensePolynomial::from_coefficients_slice(q_coeffs);

    let labeled_f = LabeledPolynomial::new("test".to_string(), f_polynomial, None, None);
    let labeled_r = LabeledPolynomial::new("test".to_string(), r_polynomial, None, None);
    let labeled_q = LabeledPolynomial::new("test".to_string(), q_polynomial, None, None);

    let (c_f, rands_f) = LigeroPCS::commit(&ck, &[labeled_f.clone()], None).unwrap();
    let (c_r, rands_r) = LigeroPCS::commit(&ck, &[labeled_r.clone()], None).unwrap();
    let (c_q, rands_q) = LigeroPCS::commit(&ck, &[labeled_q.clone()], None).unwrap();

    let f_value = labeled_f.evaluate(&point);
    let f_proof = LigeroPCS::open(
        &ck,
        &[labeled_f],
        &c_f,
        &point,
        &mut (test_sponge.clone()),
        &rands_f,
        None,
    )
    .unwrap();

    let r_value = labeled_r.evaluate(&point);
    let r_proof = LigeroPCS::open(
        &ck,
        &[labeled_r],
        &c_r,
        &point,
        &mut (test_sponge.clone()),
        &rands_r,
        None,
    )
    .unwrap();

    let q_value = labeled_q.evaluate(&point);
    let q_proof = LigeroPCS::open(
        &ck,
        &[labeled_q],
        &c_q,
        &point,
        &mut (test_sponge.clone()),
        &rands_q,
        None,
    )
    .unwrap();

    let opening_proof_duration = opening_proof_start.elapsed();
    println!(
        "Generating trimming proof took: {:?}",
        opening_proof_duration
    );

    let verification_start = Instant::now();
    assert!(LigeroPCS::check(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut (test_sponge.clone()),
        None
    )
    .unwrap());
    assert!(LigeroPCS::check(
        &vk,
        &c_f,
        &point,
        [f_value],
        &f_proof,
        &mut (test_sponge.clone()),
        None
    )
    .unwrap());
    assert!(LigeroPCS::check(
        &vk,
        &c_r,
        &point,
        [r_value],
        &r_proof,
        &mut (test_sponge.clone()),
        None
    )
    .unwrap());
    assert!(LigeroPCS::check(
        &vk,
        &c_q,
        &point,
        [q_value],
        &q_proof,
        &mut (test_sponge.clone()),
        None
    )
    .unwrap());
    let verification_duration = verification_start.elapsed();
    println!("Verification took: {:?}", verification_duration);
    println!("Regular eval: {}", value);
    println!(
        "Eval the other way: {}",
        r_value
            + f_value * (point.pow([(frame_size * frame_index) as u64]))
            + q_value * (point.pow([(frame_size * (frame_index + 1)) as u64]))
    );
}
