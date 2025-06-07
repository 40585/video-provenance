use fixed::{traits::Fixed, types::extra::U32, FixedU64};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use sprs::{linalg::ordering::start, CsMat, CsVec, TriMat};

fn gaussian_kernel1d(sigma: i32, radius: usize) -> Vec<f64> {
    // Compute sigma squared as f64 for precision
    let sigma2 = (sigma * sigma) as f64;

    // Create a vector to hold the Gaussian kernel
    let mut kernel = Vec::new();

    // Generate values from -radius to +radius
    for x in -(radius as i32)..=(radius as i32) {
        let value = (-0.5 / sigma2 * (x as f64).powi(2)).exp();
        kernel.push(value);
    }

    // Normalize the kernel so that it sums to 1
    let sum: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|x| *x /= sum);

    kernel
}

fn gaussian_kernel1d_fixed_point(sigma: i32, radius: usize) -> Vec<u64> {
    let floating_point_kernel = gaussian_kernel1d(sigma, radius);
    floating_point_kernel
        .iter()
        .map(|x| ((x * 2f64.powf(32.0)) as u64))
        .collect::<Vec<u64>>()
}

fn freivalds_matrix(kernel: Vec<u64>, height: usize, width: usize) -> CsMat<u64> {
    let numpixels = height * width;
    let mut matrix = TriMat::new((numpixels, numpixels));
    let radius = (kernel.len() - 1) / 2;
    let one_side_sum = kernel[0..radius].iter().sum();

    for pixel in 0..numpixels {
        let (row, col) = (pixel / width, pixel % width);
        let mut start_idx = pixel - radius;
        let row_start = row * width;
        if row_start > start_idx {
            start_idx = row_start;
        }
        let mut end_idx = pixel + radius;
        let row_end = (row + 1) * width - 1;
        if row_end < end_idx {
            end_idx = row_end;
        }

        matrix.add_triplet(pixel, pixel, kernel[radius]);

        let mut remaining_left_mass = one_side_sum;
        if start_idx == pixel {
            let triplet_index = matrix.find_locations(pixel, pixel);
            matrix.set_triplet(
                triplet_index[0],
                pixel,
                pixel,
                kernel[radius] + remaining_left_mass,
            );
        } else {
            let mut col_index = pixel - 1;
            let mut filter_index = radius - 1;
            while col_index > start_idx {
                matrix.add_triplet(pixel, col_index, kernel[filter_index]);
                remaining_left_mass -= kernel[filter_index];
                col_index -= 1;
                filter_index -= 1;
            }
            matrix.add_triplet(pixel, col_index, remaining_left_mass);
        }

        let mut remaining_right_mass = one_side_sum;
        if end_idx == pixel {
            let triplet_index = matrix.find_locations(pixel, pixel);
            matrix.set_triplet(
                triplet_index[0],
                pixel,
                pixel,
                kernel[radius] + remaining_right_mass,
            );
        } else {
            let mut col_index = pixel + 1;
            let mut filter_index = radius + 1;
            while col_index < end_idx {
                matrix.add_triplet(pixel, col_index, kernel[filter_index]);
                remaining_right_mass -= kernel[filter_index];
                col_index += 1;
                filter_index += 1;
            }

            matrix.add_triplet(pixel, col_index, remaining_right_mass);
        }
    }

    matrix.to_csr()
}

fn freivalds_prover(
    sigma: i32,
    radius: usize,
    image_width: usize,
    image_height: usize,
    src: &[u8],
) -> (Vec<u64>, Vec<u64>, Vec<u8>) {
    let gaussian_kernel = gaussian_kernel1d_fixed_point(sigma, radius);
    let freivalds_matrix = freivalds_matrix(gaussian_kernel, image_height, image_width);

    let src_vector = CsVec::new(
        src.len(),
        (0..src.len()).collect(),
        src.into_iter().map(|x| *x as u64).collect(),
    );

    let channel_blurred_sparse = &freivalds_matrix * &src_vector;
    let channel_blurred = channel_blurred_sparse
        .to_dense()
        .to_vec()
        .into_iter()
        .map(|x| (x >> 32) as u8)
        .collect();
    let mut rng = thread_rng();
    let freivalds_randomness: Vec<u64> = (0..src.len()).map(|_| rng.gen()).collect();
    let random_vector = CsVec::new(
        freivalds_randomness.len(),
        (0..freivalds_randomness.len()).collect(),
        freivalds_randomness.clone(),
    );

    let channel_t_c = &freivalds_matrix.transpose_into() * &random_vector;
    (
        freivalds_randomness,
        channel_t_c.to_dense().to_vec(),
        channel_blurred,
    )
}

fn freivalds_verifier(
    freivalds_randomness: Vec<u64>,
    channel_t_c: Vec<u64>,
    src: &[u8],
    channel_blurred: &[u8],
) {
    if freivalds_randomness.len() != channel_blurred.len() {
        panic!("Vector lengths are wrong");
    }
    if channel_t_c.len() != src.len() {
        panic!("Vector lengths are wrong");
    }
    let dot_product_1: u64 = freivalds_randomness
        .iter()
        .zip(channel_blurred.iter())
        .map(|(x, y)| ((*x) * (*y as u64)))
        .sum();
    let dot_product_2: u64 = channel_t_c
        .iter()
        .zip(src.iter())
        .map(|(x, y)| ((*x) * (*y as u64)))
        .sum();

    if (dot_product_1 != dot_product_2) {
        panic!("Verification failed!");
    }
}
