#![no_main]
sp1_zkvm::entrypoint!(main);

use sp1_zkvm::syscalls::syscall_inner_product;

pub fn main() {
    let a: Vec<u32> = sp1_zkvm::io::read::<Vec<u32>>();
    let b: Vec<u32> = sp1_zkvm::io::read::<Vec<u32>>();
    let p: u32 = sp1_zkvm::io::read::<u32>();

    // Create vectors with length prefix as expected by syscall
    let mut a_with_len = vec![a.len() as u32];
    a_with_len.extend_from_slice(&a);

    let mut b_with_len = vec![b.len() as u32];
    b_with_len.extend_from_slice(&b);

    // Call syscall - result will be written to a_with_len[0]
    println!("cycle-tracker-start: compute");
    unsafe {
        syscall_inner_product(a_with_len.as_mut_ptr(), b_with_len.as_mut_ptr());
    }
    println!("cycle-tracker-end: compute");

    // Get result from a_with_len[0] and apply modulo p
    let inner_product = a_with_len[0] % p;

    sp1_zkvm::io::commit(&inner_product);
}
