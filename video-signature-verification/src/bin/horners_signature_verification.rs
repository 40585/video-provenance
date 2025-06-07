// Code largely adapted from: https://github.com/Pratyush/hekaton-system/blob/main/cp-groth16/benches/bench.rs
// The code in that file did like 90% of what I needed already.

use ark_bls12_381::{Bls12_381 as E, Fr as F};
use ark_cp_groth16::{
    committer::CommitmentBuilder,
    generator::generate_parameters,
    verifier::{prepare_verifying_key, verify_proof},
    MultiStageConstraintSynthesizer, MultiStageConstraintSystem,
};
use ark_ff::{Field, One, PrimeField, UniformRand};
use ark_groth16::r1cs_to_qap::LibsnarkReduction as QAP;
use ark_r1cs_std::{
    eq::EqGadget,
    fields::fp::FpVar,
    prelude::{AllocVar, FieldVar},
};
use ark_relations::{
    ns,
    r1cs::{ConstraintSystemRef, SynthesisError},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use sha2::{Digest, Sha256};
use std::any::type_name;

const MESSAGE_LENGTH: usize = 1 << 16;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// A multistage circuit
/// Stage 1. Witness the coefficients of the polynomial.
/// Stage 2. Prove that the polynomial evaluates to a given value at a random point derived in an F-S way from the coefficients.
#[derive(Clone)]
struct PolyEvalCircuit {
    // A polynomial that is committed in stage 0. Constrained to be monic.
    pub polynomial: Vec<F>,

    // The variable corresponding to `polynomial` that is generated after stage 0.
    pub polynomial_var: Option<Vec<FpVar<F>>>,

    // The evaluation point for the polynomial. Derived via Fiat-Shamir + Commit and Prove.
    pub point: Option<F>,

    // The evaluation of `self.polynomial` at `self.root`.
    pub evaluation: Option<F>,
}

impl PolyEvalCircuit {
    fn new(polynomial: Vec<F>) -> Self {
        Self {
            polynomial,
            polynomial_var: None,
            point: None,
            evaluation: None,
        }
    }

    fn rand(mut rng: impl Rng) -> Self {
        // Sample a random monic polynomial of degree MESSAGE_LENGTH - 1
        let degree = MESSAGE_LENGTH - 1;
        let mut polynomial = (0..degree).map(|_| F::rand(&mut rng)).collect::<Vec<_>>();
        polynomial.push(F::one());
        Self::new(polynomial)
    }

    fn add_point(&mut self, point: F) {
        use ark_std::Zero;
        self.point = Some(point);
        self.evaluation = Some(
            self.polynomial
                .iter()
                .enumerate()
                .fold(F::zero(), |acc, (i, c)| acc + c * &point.pow(&[i as u64])),
        );
    }

    fn stage_0(&mut self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let polynomial_var = self
            .polynomial
            .iter()
            .map(|c| FpVar::new_witness(ns!(cs, "coeff"), || Ok(c)))
            .collect::<Result<Vec<_>, _>>()?;
        polynomial_var
            .last()
            .unwrap()
            .enforce_equal(&FpVar::one())?;
        self.polynomial_var = Some(polynomial_var);

        Ok(())
    }

    fn stage_1(&mut self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let point = FpVar::new_input(ns!(cs, "point"), || Ok(self.point.unwrap()))?;
        let evaluation = FpVar::new_input(ns!(cs, "point"), || Ok(self.evaluation.unwrap()))?;

        // This way of implementing evaluation takes half the coefficients as what was in Hekaton!
        let claimed_eval = self
            .polynomial_var
            .as_ref()
            .unwrap()
            .iter()
            .rev()
            .fold(FpVar::zero(), |acc, x| (acc * &point) + x);

        // Assert that it's a root
        claimed_eval.enforce_equal(&evaluation)?;
        Ok(())
    }
}

impl MultiStageConstraintSynthesizer<F> for PolyEvalCircuit {
    fn total_num_stages(&self) -> usize {
        2
    }

    fn generate_constraints(
        &mut self,
        stage: usize,
        cs: &mut MultiStageConstraintSystem<F>,
    ) -> Result<(), SynthesisError> {
        let out = match stage {
            0 => cs.synthesize_with(|c| self.stage_0(c)),
            1 => cs.synthesize_with(|c| self.stage_1(c)),
            _ => panic!("unexpected stage stage {}", stage),
        };

        out
    }
}

fn main() {
    let mut rng = ark_std::test_rng();
    let circuit = PolyEvalCircuit::rand(&mut rng);

    // Run the circuit and make sure it succeeds
    {
        let mut circuit = circuit.clone();
        let mut cs = MultiStageConstraintSystem::default();
        circuit.generate_constraints(0, &mut cs).unwrap();
        let point = F::rand(&mut rng);
        circuit.add_point(point);
        circuit.generate_constraints(1, &mut cs).unwrap();
        // assert!(cs.is_satisfied().unwrap());
        println!("Num constraints: {:?}", cs.num_constraints());
    }

    // Generate the proving key
    let start = ark_std::time::Instant::now();
    let pk = generate_parameters::<_, E, QAP>(circuit.clone(), &mut rng).unwrap();
    println!(
        "setup time for BLS12-381: {} s",
        start.elapsed().as_secs_f64()
    );

    // Generate a commitment to coefficients
    let mut rng = ark_std::test_rng();
    let mut cb = CommitmentBuilder::<_, E, QAP>::new(circuit, &pk);
    let start = ark_std::time::Instant::now();
    let (comm, rand) = cb.commit(&mut rng).unwrap();
    println!(
        "commitment time for BLS12-381: {} s",
        start.elapsed().as_secs_f64()
    );
    println!("Comm is: {:?}", comm);
    // println!("Comm bytes are: {:?}", comm.into_bytes());

    // Generate point from the commitment
    let start = ark_std::time::Instant::now();
    let mut compressed_comm = Vec::new();
    comm.serialize_compressed(&mut compressed_comm).unwrap();

    let mut hasher = Sha256::new();
    hasher.update(&compressed_comm);
    let result = hasher.finalize();

    let point = F::from_be_bytes_mod_order(&result);
    cb.circuit.add_point(point); // This takes surprisingly long when there are a lot of coefficients
    println!(
        "generating point from commit and prove data: {} s",
        start.elapsed().as_secs_f64()
    );

    // Actual proof.
    let start = ark_std::time::Instant::now();
    let inputs = [point, cb.circuit.evaluation.unwrap()];
    let proof = cb.prove(&[comm], &[rand], &mut rng).unwrap();
    println!(
        "proving time for BLS12-381: {} s",
        start.elapsed().as_secs_f64()
    );

    // Verify
    let start = ark_std::time::Instant::now();
    let pvk = prepare_verifying_key(&pk.vk());
    assert!(verify_proof(&pvk, &proof, &inputs).unwrap());

    // Verify that eval point was derived correctly.
    let mut compressed_comm = Vec::new();
    proof.ds[0]
        .serialize_compressed(&mut compressed_comm)
        .unwrap();

    let mut hasher = Sha256::new();
    hasher.update(&compressed_comm);
    let result = hasher.finalize();

    let claimed_point = F::from_be_bytes_mod_order(&result);
    assert!(claimed_point == inputs[0]);

    println!(
        "verification time for BLS12-381: {} s",
        start.elapsed().as_secs_f64()
    );
}
