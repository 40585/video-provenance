#![no_main]
sp1_zkvm::entrypoint!(main);
use blake3::hash;
use sp1_zkvm::syscalls::syscall_inner_product;
#[sp1_derive::cycle_tracker]

pub fn main() {
    println!("cycle-tracker-start: setup");
    let original_image: Vec<u8> = sp1_zkvm::io::read_vec(); // 320 x 240
    let target_image: Vec<u8> = sp1_zkvm::io::read_vec(); // 160 x 120

    let middle_target_image: Vec<u32> = sp1_zkvm::io::read::<Vec<u32>>();
    let r = sp1_zkvm::io::read::<Vec<u32>>(); // 240
    let s = sp1_zkvm::io::read::<Vec<u32>>(); // 320

    let freivalds_r = sp1_zkvm::io::read::<Vec<u32>>(); // 120
    let freivalds_s = sp1_zkvm::io::read::<Vec<u32>>(); // 160

    // let mut tmp = vec![0u8; c.width as usize * c.height as usize];

    /*
    Prove 1: prove r @ original_image @ s = freivalds_r @ middle_target_image @ freivalds_s
    r = freivalds_r @ H
    s = V @ freivalds_s
    */

    // Calculate dst = r @ original_image @ s using matrix multiplication
    // original_image is a flattened matrix of size r_size * s_size
    let r_size = r.len();
    let s_size = s.len();

    // First multiply original_image (r_size x s_size) with s (s_size x 1)
    //let mut temp = vec![0u32; r_size];
    // for i in 0..r_size {
    //     for j in 0..s_size {
    //         temp[i] += (original_image[i * s_size + j] as u32) * s[j];
    //     }
    // }
    // First multiply original_image (r_size x s_size) with s (s_size x 1)
    let mut temp = vec![0u32; r_size];
    for i in 0..r_size {
        // Create vectors with length prefix as expected by syscall
        let mut row_with_len = vec![s_size as u32];
        row_with_len.extend_from_slice(
            &original_image[i * s_size..(i + 1) * s_size]
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<u32>>(),
        );

        let mut s_with_len = vec![s_size as u32];
        s_with_len.extend_from_slice(&s);

        // Call syscall - result will be written to row_with_len[0]
        unsafe {
            syscall_inner_product(row_with_len.as_mut_ptr(), s_with_len.as_mut_ptr());
        }
        temp[i] = row_with_len[0];
    }

    let mut sum = 0u32;
    for i in 0..r_size {
        sum += r[i] * temp[i];
    }
    // Then multiply r (1 x r_size) with the result (r_size x 1)
    // padding zero to make the length of r_with_len and temp_with_len 320 for syscall
    // let mut r_with_len = vec![320];
    // r_with_len.extend_from_slice(&r);
    // r_with_len.extend(vec![0u32; 320 - r.len()]);

    // let mut temp_with_len = vec![320];
    // temp_with_len.extend_from_slice(&temp);
    // temp_with_len.extend(vec![0u32; 320 - temp.len()]);

    // // Call syscall - result will be written to r_with_len[0]
    // unsafe {
    //     syscall_inner_product(r_with_len.as_mut_ptr(), temp_with_len.as_mut_ptr());
    // }
    // let sum = r_with_len[0];

    let mut tmp_freivalds = vec![0u32; r_size];
    let freivalds_s_size = freivalds_s.len();
    let freivalds_r_size = freivalds_r.len();

    for i in 0..freivalds_r_size {
        for j in 0..freivalds_s_size {
            tmp_freivalds[i] +=
                (middle_target_image[i * freivalds_s_size + j] as u32) * freivalds_s[j];
        }
    }
    // let mut tmp_freivalds = vec![0u32; r_size];
    // let freivalds_s_size = freivalds_s.len();
    // let freivalds_r_size = freivalds_r.len();

    // for i in 0..freivalds_r_size {
    //     // Create vectors with length prefix as expected by syscall
    //     let mut row_with_len = vec![320];
    //     row_with_len.extend_from_slice(
    //         &middle_target_image[i * freivalds_s_size..(i + 1) * freivalds_s_size]
    //             .iter()
    //             .map(|&x| x as u32)
    //             .collect::<Vec<u32>>(),
    //     );
    //     row_with_len.extend(vec![0u32; 320 - freivalds_s_size]);

    //     let mut s_with_len = vec![320];
    //     s_with_len.extend_from_slice(&freivalds_s);
    //     s_with_len.extend(vec![0u32; 320 - freivalds_s.len()]);

    //     // Call syscall - result will be written to row_with_len[0]
    //     unsafe {
    //         syscall_inner_product(row_with_len.as_mut_ptr(), s_with_len.as_mut_ptr());
    //     }
    //     tmp_freivalds[i] = row_with_len[0];
    // }

    let mut sum_freivalds = 0u32;
    for i in 0..freivalds_r_size {
        sum_freivalds += freivalds_r[i] * tmp_freivalds[i];
    }
    // Create vectors with length prefix for final inner product
    // let mut freivalds_r_with_len = vec![320];
    // freivalds_r_with_len.extend_from_slice(&freivalds_r);
    // freivalds_r_with_len.extend(vec![0u32; 320 - freivalds_r.len()]);

    // let mut tmp_freivalds_with_len = vec![320];
    // tmp_freivalds_with_len.extend_from_slice(&tmp_freivalds);
    // tmp_freivalds_with_len.extend(vec![0u32; 320 - tmp_freivalds.len()]);

    // // Call syscall - result will be written to freivalds_r_with_len[0]
    // unsafe {
    //     syscall_inner_product(
    //         freivalds_r_with_len.as_mut_ptr(),
    //         tmp_freivalds_with_len.as_mut_ptr(),
    //     );
    // }
    // let sum_freivalds = freivalds_r_with_len[0];

    if sum != sum_freivalds {
        sp1_zkvm::io::commit(&false);
        return;
    } else {
        sp1_zkvm::io::commit(&true);
    }

    /*
    Prove 2: prove |target_image - middle_target_image| <= 20
    */
    let limit1 = 20;
    let limit2 = 50;
    let mut cnt1 = 0;
    let mut cnt2 = 0;
    for i in 0..middle_target_image.len() {
        let middle_val = middle_target_image[i] / (1 << 22) as u32;
        let target_val = target_image[i] as u32;
        let difference = if middle_val > target_val {
            middle_val - target_val
        } else {
            target_val - middle_val
        };
        if difference > limit1 {
            cnt1 += 1;
        }
        if difference > limit2 {
            cnt2 += 1;
        }
    }
    sp1_zkvm::io::commit(&cnt1);
    sp1_zkvm::io::commit(&cnt2);
    let hash = blake3::hash(&target_image);
    sp1_zkvm::io::commit(&hash);
    //sp1_zkvm::io::commit(&hash(&target_image));
    //print!("difference: {:?}", &difference);
    println!("cycle-tracker-start: commit");
    println!("cycle-tracker-end: commit");
}
