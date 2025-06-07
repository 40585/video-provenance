import numpy as np
from PIL import Image

def decompose_image(file_path, save_path):
    # Open the image file
    img = Image.open(file_path)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Check if the image is RGB
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        raise ValueError("The image must be in RGB format")
    
    # Decompose into R, G, B channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]
    
    for channel, name in zip([r_channel, g_channel, b_channel], ['R', 'G', 'B']):
        output_file = f'{save_path}_{name}.txt'
        np.savetxt(output_file, channel, fmt='%d', delimiter=' ')
        print(f"Saved {name} channel to {output_file}")
    
    return r_channel, g_channel, b_channel

# Usage
if __name__ == '__main__':
    file_path = './../resources/ffmpeg_resized_frames_48_27/output_003.png'
    r, g, b = decompose_image(file_path, "decomposed_file")

    print("Red channel shape:", r.shape)
    print("Green channel shape:", g.shape)
    print("Blue channel shape:", b.shape)