import numpy as np
import cv2
import argparse

INPUT_PATH="./../resources/random_19201080_images/image1.jpg"
OUTPUT_PATH="./../resources/random_19201080_images/image1_simple_resized.jpg"

def process_image_channel(channel, src_w, src_h, dst_w, dst_h):
    """
    Apply bilinear interpolation using FFmpeg's coordinate calculation method.
    """
    # FFmpeg style coordinate mapping
    x_ratio = src_w / dst_w
    y_ratio = src_h / dst_h
    
    # Calculate x and y coordinates for each pixel in the output
    x = np.arange(dst_w)
    y = np.arange(dst_h)
    X, Y = np.meshgrid(x, y)
    
    # FFmpeg maps coordinates to the center of pixels
    src_x = (X + 0.5) * x_ratio - 0.5
    src_y = (Y + 0.5) * y_ratio - 0.5
    
    # Clamp coordinates to valid range
    src_x = np.clip(src_x, 0, src_w - 1)
    src_y = np.clip(src_y, 0, src_h - 1)
    
    # Get the four surrounding pixels for each destination pixel
    x1 = np.floor(src_x).astype(int)
    x2 = np.minimum(x1 + 1, src_w - 1)
    y1 = np.floor(src_y).astype(int)
    y2 = np.minimum(y1 + 1, src_h - 1)
    
    # Calculate interpolation weights
    wx2 = src_x - x1
    wx1 = 1 - wx2
    wy2 = src_y - y1
    wy1 = 1 - wy2
    
    # Get values of surrounding pixels
    Q11 = channel[y1, x1]
    Q12 = channel[y1, x2]
    Q21 = channel[y2, x1]
    Q22 = channel[y2, x2]
    
    # Calculate weighted sum for bilinear interpolation
    result = (Q11 * wx1 * wy1 +
              Q12 * wx2 * wy1 +
              Q21 * wx1 * wy2 +
              Q22 * wx2 * wy2)
    
    # Compare with OpenCV resize for quality assessment
    opencv_result = cv2.resize(channel, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    difference = np.sum(np.abs(result - opencv_result))
    
    return result, difference

def main():
    parser = argparse.ArgumentParser(description='Apply bilinear filtering to an image')
    parser.add_argument('--width', type=int, default=960, help='Output width')
    parser.add_argument('--height', type=int, default=540, help='Output height')
    args = parser.parse_args()

    # Read input image
    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise ValueError(f"Could not read image from {INPUT_PATH}")
    
    src_h, src_w = img.shape[:2]
    dst_w, dst_h = args.width, args.height
    
    # Process each channel separately
    processed_channels = []
    total_difference = 0
    
    for channel_idx in range(3):  # BGR channels
        channel = img[:, :, channel_idx]
        processed_channel, diff = process_image_channel(
            channel, src_w, src_h, dst_w, dst_h
        )
        processed_channels.append(processed_channel.reshape(dst_h, dst_w))
        total_difference += diff
    
    # Combine channels and save result
    result_img = np.stack(processed_channels, axis=-1)
    cv2.imwrite(OUTPUT_PATH, result_img)
    
    print(f"Total difference from resize: {total_difference}")
    print(f"Average difference per pixel: {total_difference / (dst_w * dst_h * 3):.2f}")

if __name__ == "__main__":
    main()