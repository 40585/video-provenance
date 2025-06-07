use sp1_sdk::{utils, ProverClient, SP1ProofWithPublicValues, SP1Stdin, SP1VerifyingKey};
use std::io::Write;
use sp1_sdk::include_elf;
use clap::Parser;

/// The ELF we want to execute inside the zkVM.
const ELF: &[u8] = include_elf!("fibonacci-program");


fn main() {
    // Setup logging.
    utils::setup_logger();
    let client = ProverClient::from_env();
    
    let config_file = std::fs::File::open("config.json").expect("Failed to open config file");
    let config: serde_json::Value = serde_json::from_reader(config_file).expect("Failed to parse JSON");
    let target_vk_file = config["target_vk_file"].as_str().expect("Missing target_vk_file");
    let target_pk_file = config["target_pk_file"].as_str().expect("Missing target_prove_file");

    let client = ProverClient::from_env();

    let (pk, vk) = client.setup(ELF);
    
    let serialized_vk = serde_cbor::to_vec(&vk).expect("serializing vk failed");
    std::fs::write(target_vk_file, serialized_vk).expect("saving serialized vk failed");

    let serialized_pk = serde_cbor::to_vec(&pk).expect("serializing pk failed");
    std::fs::write(target_pk_file, serialized_pk).expect("saving serialized pk failed");

    println!("Successfully Save Pk and Vk");
}