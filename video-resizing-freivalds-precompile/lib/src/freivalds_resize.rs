const FILTER_BITS: i32 = 11;
const FILTER_SCALE: i32 = 1 << FILTER_BITS;

pub fn generate_horizontal_filter(src_w: i32, dst_w: i32) -> (Vec<i32>, Vec<i32>) {
    let filter_size = 4;
    let mut filter_pos = Vec::with_capacity(dst_w as usize);
    let mut filter_coeffs = Vec::with_capacity((dst_w * filter_size) as usize);
    
    // x_inc is scaling factor in 16.16 fixed point
    let x_inc = ((src_w << 16) / dst_w + 1) >> 1;
    
    for i in 0..dst_w {
        // Get source position in 16.16 fixed point
        let src_pos = (i * x_inc) >> 15;
        
        // Get fractional part normalized to FILTER_BITS
        let xx_inc = x_inc & 0xffff;
        let xx = (xx_inc * (1 << FILTER_BITS)) / x_inc;
        
        filter_pos.push(src_pos);
        
        // Calculate filter weights
        for j in 0..filter_size {
            let coeff = if j == 0 {
                (1 << FILTER_BITS) - xx
            } else {
                xx
            };
            filter_coeffs.push(coeff);
        }
    }

    (filter_pos, filter_coeffs)
}

pub fn generate_vertical_filter(src_h: i32, dst_h: i32, filter_size: i32) -> (Vec<i32>, Vec<i32>) {
    let mut filter_pos = Vec::with_capacity(dst_h as usize);
    let mut filter_coeffs = Vec::with_capacity((dst_h * filter_size) as usize);
    let scale = src_h as f64 / dst_h as f64;
    
    for i in 0..dst_h {
        let center = (i as f64 + 0.5) * scale - 0.5;
        let top = (center - filter_size as f64 / 2.0).ceil() as i32;
        filter_pos.push(top);
        
        let mut weights = Vec::with_capacity(filter_size as usize);
        for j in 0..filter_size {
            let weight = if filter_size > 1 {
                1.0 - (((j as f64) - (center - top as f64)).abs() / (filter_size as f64 / 2.0))
            } else {
                1.0
            };
            weights.push((weight * FILTER_SCALE as f64) as i32);
        }
        
        // Normalize weights
        let total: i32 = weights.iter().sum();
        let normalized_weights: Vec<i32> = weights
            .iter()
            .map(|&w| w * FILTER_SCALE / total)
            .collect();
        filter_coeffs.extend(normalized_weights);
    }
    
    (filter_pos, filter_coeffs)
}

pub fn build_horizontal_matrix(src_w: i32, dst_w: i32) -> Vec<Vec<i32>> {
    let (h_pos, h_coeffs) = generate_horizontal_filter(src_w, dst_w);
    let filter_size = 4;
    
    // Create matrix for horizontal scaling
    let mut h = vec![vec![0; dst_w as usize]; src_w as usize];
    
    // Fill in the matrix based on filter positions and coefficients
    for x in 0..dst_w {
        let src_pos = h_pos[x as usize];
        for z in 0..filter_size {
            if src_pos + z < src_w {
                h[(src_pos + z) as usize][x as usize] = h_coeffs[(x * filter_size + z) as usize];
            }
        }
    }
    h
}

pub fn build_vertical_matrix(src_h: i32, dst_h: i32) -> Vec<Vec<i32>> {
    let (v_pos, v_coeffs) = generate_vertical_filter(src_h, dst_h, 4);
    let filter_size = 4;
    
    // Create matrix for vertical scaling
    let mut v = vec![vec![0; src_h as usize]; dst_h as usize];
    // Fill in the matrix based on filter positions and coefficients
    for y in 0..dst_h {
        let src_pos = v_pos[y as usize];
        for z in 0..filter_size {
            if src_pos + z < src_h && src_pos + z >= 0{
                v[y as usize][(src_pos + z) as usize] = v_coeffs[(y * filter_size + z) as usize];
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_generation() {
        let src_w = 270;
        let src_h = 480;
        let dst_w = 135;
        let dst_h = 240;

        let (h_pos, h_coeffs) = generate_horizontal_filter(src_w, dst_h);
        let (v_pos, v_coeffs) = generate_vertical_filter(src_h, dst_h, 4);

        assert!(!h_pos.is_empty());
        assert!(!h_coeffs.is_empty());
        assert!(!v_pos.is_empty());
        assert!(!v_coeffs.is_empty());
    }
}
