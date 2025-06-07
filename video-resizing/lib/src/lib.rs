pub mod resize;
pub use resize::*;

pub mod gblur;
pub use gblur::*;

pub mod avgblur;
pub use avgblur::*;

pub mod freivalds_gblur;
pub use freivalds_gblur::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// demo image processing
// pub fn demo_processing(image: &[u8]) -> Vec<u8> {
//     image.iter().map(|&x| x.wrapping_add(1)).collect()
// }

// pub fn print_data_sample(data: &[u8], sample_size: usize) {
//     println!("Data sample (first {} values):", sample_size);
//     for (i, &value) in data.iter().take(sample_size).enumerate() {
//         print!("{:3} ", value);
//         if (i + 1) % 16 == 0 {
//             println!();
//         }
//     }
//     println!();
// }

// pub fn print_image_summary(width: usize, height: usize, data: &[u8]) {
//     let mut sum: u64 = 0;
//     let mut min_val: u8 = 255;
//     let mut max_val: u8 = 0;

//     for &pixel in data.iter() {
//         sum += pixel as u64;
//         min_val = min(min_val, pixel);
//         max_val = max(max_val, pixel);
//     }

//     let avg = sum as f64 / (width * height) as f64;

//     println!("Image Summary ({}x{}):", width, height);
//     println!("Min value: {}", min_val);
//     println!("Max value: {}", max_val);
//     println!("Average value: {:.2}", avg);
// }

pub fn load_image_from_file(input_file: &str) -> Vec<u8> {
    let mut image: Vec<u8> = Vec::new();

    let input_path = Path::new(input_file);
    let file = File::open(&input_path).expect("Failed to open input file");
    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        match line {
            Ok(line_data) => {
                // Split the line into separate numbers
                for value_str in line_data.split_whitespace() {
                    match value_str.parse::<u8>() {
                        Ok(value) => {
                            image.push(value);
                        }
                        Err(_) => {
                            println!("Error parsing value on line {}: '{}'", i + 1, value_str);
                        }
                    }
                }
            }
            Err(e) => {
                println!("Error reading line {}: {:?}", i + 1, e);
            }
        }
    }

    image
}
