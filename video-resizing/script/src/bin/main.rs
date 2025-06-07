use lib::{load_image_from_file, BlurContext, ResizeContext};
use sp1_sdk::{utils, ProverClient, SP1ProofWithPublicValues, SP1Stdin};

/// The ELF we want to execute inside the zkVM.
const ELF: &[u8] = include_bytes!("../../../elf/riscv32im-succinct-zkvm-elf");

const _FRAME_NUM: usize = 10;
const INPUT_WIDTH: i32 = 320;
const INPUT_HEIGHT: i32 = 240;
const OUTPUT_WIDTH: i32 = 160;
const OUTPUT_HEIGHT: i32 = 120;

fn main() {
    // Setup logging.
    utils::setup_logger();

    //fake example
    let input_file = "../resources/decomposed_frames_demo.txt"; // "../../resources/ffmpeg_original_frames_192_108/output_001_R.txt";
    let target_file = "../resources/decomposed_resized_frams_demo.txt"; //"../../resources/ffmpeg_resized_frames_48_27/output_001_R.txt";
                                                                        //let output_file = "image_output.txt";

    let context =
        ResizeContext::new(INPUT_WIDTH, INPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_HEIGHT).unwrap();
    // let context =
    //     BlurContext::new(10.0, 10.0, 1, INPUT_WIDTH as usize, INPUT_HEIGHT as usize).unwrap();
    // Get the Image
    let image: Vec<u8> = load_image_from_file(input_file);
    let target_image: Vec<u8> = load_image_from_file(target_file);
    println!("image_len(): {:?}", image.len());
    println!("target_image.len(): {:?}", target_image.len());

    // The input stream that the program will read from using `sp1_zkvm::io::read`.
    // Note that the types of the elements in the input stream must match the types being
    // read in the program.
    let mut stdin = SP1Stdin::new();
    stdin.write::<ResizeContext>(&context);
    stdin.write_vec(image);
    stdin.write_vec(target_image);

    // Create a `ProverClient` method.
    let client = ProverClient::from_env();

    // Execute the program using the `ProverClient.execute` method, without generating a proof.
    // let (_, report) = client.execute(ELF, stdin.clone()).run().unwrap();
    // println!(
    //     "executed program with {} cycles",
    //     report.total_instruction_count()
    // );

    // Generate the proof for the given program and input.
    let (pk, vk) = client.setup(ELF);
    let mut proof = client.prove(&pk, &stdin).run().unwrap();

    //client
    // .prove(&resizing_pk, stdin)
    // .compressed()
    // .run()
    // .expect("proving failed")
    println!("generated proof");

    // Read and verify the output.
    //
    // Note that this output is read from values commited to in the program using
    // `sp1_zkvm::io::commit`.
    //let _ = proof.public_values.read::<u32>();

    // Calculate the L1 norm here
    // Todo: change to range check & speed up

    // Note: if commit a Vec<> the proof time will be very long
    // for quick verification, use a usize to sum it up
    // TODO: investigate why commit to Vec<> will hurt the performance

    // ======================================================================
    //let difference: Vec<usize> = proof.public_values.read::<Vec<usize>>();
    //println!("difference[0]: {}", difference[0]);
    let within_limit: bool = proof.public_values.read::<bool>();
    assert!(within_limit, "within_limit = {}", within_limit);

    // Verify proof and public values
    client.verify(&proof, &vk).expect("verification failed");

    // Test a round trip of proof serialization and deserialization.
    proof
        .save("proof-with-pis.bin")
        .expect("saving proof failed");
    let deserialized_proof =
        SP1ProofWithPublicValues::load("proof-with-pis.bin").expect("loading proof failed");

    // Verify the deserialized proof.
    client
        .verify(&deserialized_proof, &vk)
        .expect("verification failed");

    println!("successfully generated and verified proof for the program!")
}
