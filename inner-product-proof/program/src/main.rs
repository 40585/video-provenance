#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let a: Vec<u32> = sp1_zkvm::io::read::<Vec<u32>>();
    let b: Vec<u32> = sp1_zkvm::io::read::<Vec<u32>>();
    let p: u32 = sp1_zkvm::io::read::<u32>();

    let mut inner_product: u32 = 0;

    println!("cycle-tracker-start: compute");
    let mut len_a = a[0];
    let len_a_usize = len_a as usize;
    for i in 1..len_a_usize {
        inner_product = (inner_product + a[i] * b[i]) % p
    }
    println!("cycle-tracker-end: compute");

    sp1_zkvm::io::commit(&inner_product);
}
