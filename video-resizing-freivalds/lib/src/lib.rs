pub mod  freivalds_resize;
pub use  freivalds_resize::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;


pub fn load_image_from_file(input_file: &str) -> Vec<u8> {
    let mut image: Vec<u8> = Vec::new();

    let input_path = Path::new(input_file);
    println!("input_path: {:?}", input_path);
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
