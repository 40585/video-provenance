use rand::Rng;
use sp1_sdk::{include_elf, utils, ProverClient, SP1ProofWithPublicValues, SP1Stdin};

const ELF: &[u8] = include_elf!("inner-product-proof");

fn main() {
    utils::setup_logger();

    let p: u32 = 0xFFF5001; // babybear
    let vec_len = 4096;

    let mut rng = rand::rng();
    let mut a: Vec<u32> = vec![vec_len];
    a.extend((0..vec_len).map(|_| rng.gen_range(1..=255)));
    let mut b: Vec<u32> = vec![vec_len];
    b.extend((0..vec_len).map(|_| rng.gen_range(1..=255)));

    let mut stdin = SP1Stdin::new();
    stdin.write(&a);
    stdin.write(&b);
    stdin.write(&p);

    // Create a `ProverClient` method.
    let client = ProverClient::from_env();
    // Execute the program using the `ProverClient.execute` method, without generating a proof.
    let start = std::time::Instant::now();
    let (_, report) = client.execute(ELF, &stdin).run().unwrap();
    let duration = start.elapsed();
    println!("Execution time: {} milliseconds", duration.as_micros());
    println!(
        "inner-product-proof executed program with {} cycles",
        report.total_instruction_count()
    );

    
    // Generate the proof for the given program and input 
    let (pk, vk) = client.setup(ELF);
    let mut proof = client.prove(&pk, &stdin).compressed().run().unwrap();

    println!("generated proof");

    let inner_product = proof.public_values.read::<u32>();

    println!{"inner product: {}", inner_product};

    client.verify(&proof, &vk).expect("verification failed");

}
