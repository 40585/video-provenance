use sp1_sdk::{
    HashableKey, ProverClient, SP1Proof, SP1ProofWithPublicValues, SP1Stdin, SP1VerifyingKey,
};

use lib::{load_image_from_file, Context};

/// ELF that aggregates the proofs
const AGGREGATION_ELF: &[u8] = include_bytes!("../../../program/elf/riscv32im-succinct-zkvm-elf");

/// ELF that resizs images
const RESIZING_ELF: &[u8] =
    include_bytes!("../../../../video-resizing/elf/riscv32im-succinct-zkvm-elf");

const INPUT_WIDTH: i32 = 192;
const INPUT_HEIGHT: i32 = 108;
const OUTPUT_WIDTH: i32 = 48;
const OUTPUT_HEIGHT: i32 = 27;

/// An input to the aggregation program.
///
/// Consists of a proof and a verification key.
struct AggregationInput {
    pub proof: SP1ProofWithPublicValues,
    pub vk: SP1VerifyingKey,
}

fn main() {
    // Setup the logger.
    sp1_sdk::utils::setup_logger();

    // Initialize the proving client.
    let client = ProverClient::new();

    // Setup the proving and verifying keys.
    let (aggregation_pk, aggregation_vk) = client.setup(AGGREGATION_ELF);
    let (resizing_pk, resizing_vk) = client.setup(RESIZING_ELF);

    // Compute the proof of the vidoe
    let mut proofs: Vec<SP1ProofWithPublicValues> = vec![];    
    // iter over the frames 
    for i in 1..=3 {
        // iter over three channels
        for channel in ["R", "G", "B"].iter() {
            let proof_of_single_channel = tracing::info_span!("generate proof for the {} (iteration {})", channel, i).in_scope(|| {
                let input_file = format!("../../resources/ffmpeg_original_frames_192_108/output_{:03}_{}.txt", i, channel);
                let target_file = format!("../../resources/ffmpeg_resized_frames_48_27/output_{:03}_{}.txt", i, channel);
                let context = Context::new(INPUT_WIDTH, INPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_HEIGHT).unwrap();
                let image: Vec<u8> = load_image_from_file(&input_file);
                let target_image: Vec<u8> = load_image_from_file(&target_file);
                let mut stdin = SP1Stdin::new();
                stdin.write(&context);
                stdin.write(&image);
                stdin.write(&target_image);
                client
                    .prove(&resizing_pk, stdin)
                    .compressed()
                    .run()
                    .expect("proving failed")
            });
            println!("proof_{:?} generated for {} frames", channel, i);
            proofs.push(proof_of_single_channel);
        }
    
    }
    let mut inputs: Vec<AggregationInput> = vec![];
    for proof in proofs.iter() {
        inputs.push(AggregationInput{
            proof: proof.clone(),
            vk: resizing_vk.clone(),
        });
    }

    // Aggregate the proofs.
    print!("proof aggregation started");
    tracing::info_span!("aggregate the proofs").in_scope(|| {
        let mut stdin = SP1Stdin::new();

        // Write the verification keys.
        let vkeys = inputs
            .iter()
            .map(|input| input.vk.hash_u32())
            .collect::<Vec<_>>();
        stdin.write::<Vec<[u32; 8]>>(&vkeys);

        // Write the public values.
        let public_values = inputs
            .iter()
            .map(|input| input.proof.public_values.to_vec())
            .collect::<Vec<_>>();
        stdin.write::<Vec<Vec<u8>>>(&public_values);

        // Write the proofs.
        //
        // Note: this data will not actually be read by the aggregation program, instead it will be
        // witnessed by the prover during the recursive aggregation process inside SP1 itself.
        for input in inputs {
            let SP1Proof::Compressed(proof) = input.proof.proof else {
                panic!()
            };
            stdin.write_proof(proof, input.vk.vk);
        }

        // todo verify the proof to make this program end to end
        
        // Generate the plonk bn254 proof.
        // client
        //     .prove(&aggregation_pk, stdin)
        //     .plonk()
        //     .run()
        //     .expect("proving failed");
    });
    print!("proof aggregation finished");
}
