import numpy as np

FILTER_BITS = 11
FILTER_SCALE = 1 << FILTER_BITS

def generate_horizontal_filter(src_w: int, dst_w: int) -> tuple[list[int], list[int]]:
    """
    Generate horizontal filter positions and coefficients
    Returns (filter_pos, filter) tuple
    """
    filter_size = 4
    filter_pos = []
    filter_coeffs = []
    
    # x_inc is scaling factor in 16.16 fixed point
    x_inc = ((src_w << 16) // dst_w + 1) >> 1
    
    for i in range(dst_w):
        # Get source position in 16.16 fixed point
        src_pos = (i * x_inc) >> 15
        
        # Get fractional part normalized to FILTER_BITS
        xx_inc = x_inc & 0xffff 
        xx = (xx_inc * (1 << FILTER_BITS)) // x_inc
        
        filter_pos.append(int(src_pos))
        
        # Calculate filter weights
        for j in range(filter_size):
            if j == 0:
                coeff = (1 << FILTER_BITS) - xx
            else:
                coeff = xx
            filter_coeffs.append(int(coeff))

    return filter_pos, filter_coeffs

def generate_vertical_filter(src_h: int, dst_h: int, filter_size: int = 4) -> tuple[list[int], list[int]]:
    """
    Generate vertical filter positions and coefficients
    Returns (filter_pos, filter) tuple
    """
    filter_pos = []
    filter_coeffs = []
    scale = src_h / dst_h
    
    for i in range(dst_h):
        center = (i + 0.5) * scale - 0.5
        top = int(np.ceil(center - filter_size / 2))
        filter_pos.append(top)
        
        weights = []
        for j in range(filter_size):
            if filter_size > 1:
                weight = 1.0 - (abs(j - (center - top)) / (filter_size / 2))
            else:
                weight = 1.0
            weights.append(int(weight * FILTER_SCALE))
            
        # Normalize weights
        total = sum(weights)
        normalized_weights = [w * FILTER_SCALE // total for w in weights]
        filter_coeffs.extend(normalized_weights)
    
    return filter_pos, filter_coeffs


def main():
    # Example dimensions (can be modified as needed)
    src_w, src_h = 270 , 480
    dst_w, dst_h = 135, 240
    print(generate_vertical_filter(270,135))
    
    h_pos, h_coeffs = generate_horizontal_filter(src_w, dst_w)
    
    v_pos, v_coeffs = generate_vertical_filter(src_h, dst_h)
    print(h_pos, h_coeffs, v_pos, v_coeffs)
    

if __name__ == "__main__":
    main()
