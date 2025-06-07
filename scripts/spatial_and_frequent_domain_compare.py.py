import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dct
import seaborn as sns


IMAGE1_PATH="./../resources/random_19201080_images/image1_simple_resized.jpg"
IMAGE2_PATH="./../resources/random_19201080_images/image1_attack_additional_component2_resize.jpg"

# def bgr_to_yuv(bgr_img):
#     """Convert BGR image to YUV"""
#     # OpenCV's BGR to YUV conversion matrix
#     transform = np.array([
#         [ 0.114,  0.587,  0.299],    # Y
#         [-0.081, -0.419,  0.500],    # U (Cb)
#         [ 0.500, -0.331, -0.169]     # V (Cr)
#     ])
    
#     # Reshape image to 2D array of pixels
#     pixels = bgr_img.reshape(-1, 3).astype(np.float32)
    
#     # Apply transformation
#     yuv = np.dot(pixels, transform.T)
    
#     # Reshape back to image dimensions
#     return yuv.reshape(bgr_img.shape)

def load_and_preprocess_images(path1, path2, size=(960, 540)):
    """Load and preprocess two images to ensure they're comparable."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")
    
    # Convert BGR to YUV
    # yuv1 = bgr_to_yuv(img1)
    # yuv2 = bgr_to_yuv(img2)
    
    # return yuv1, yuv2
    return img1, img2

def create_difference_heatmap(img1, img2):
    """Create heatmaps showing pixel-by-pixel differences for each channel."""
    # Calculate absolute difference for each channel
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    
    # Create figure for the three channel differences
    plt.figure(figsize=(15, 5))
    channel_names = ['Blue Channel', 'Green Channel', 'Red Channel']
    
    # Plot difference heatmap for each channel
    for i in range(3):
        plt.subplot(1, 3, i+1)
        sns.heatmap(diff[:,:,i], 
                   cmap='hot',
                   cbar_kws={'label': 'Difference'},
                   vmin=0,
                   vmax=max(np.max(diff[:,:,i]), 1))  # Ensure non-zero scale
        plt.title(f'{channel_names[i]} Difference\nMax diff: {np.max(diff[:,:,i]):.2f}')
    
    plt.tight_layout()
    
    # Return the channel differences separately
    return {
        'blue_diff': diff[:,:,0],
        'green_diff': diff[:,:,1],
        'red_diff': diff[:,:,2],
        'channel_maxes': [np.max(diff[:,:,i]) for i in range(3)],
        'channel_means': [np.mean(diff[:,:,i]) for i in range(3)]
    }

def perform_channel_dct(img):
    """Perform DCT on each channel separately."""
    dct_channels = []
    for i in range(3):  # BGR channels
        channel_dct = dct(dct(img[:,:,i].T, norm='ortho').T, norm='ortho')
        dct_channels.append(channel_dct)
    return np.stack(dct_channels, axis=2)

def visualize_channel_dct(dct_result, title):
    """Visualize DCT coefficients for each channel."""
    channel_names = ['Blue', 'Green', 'Red']
    plt.figure(figsize=(15, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(np.log(np.abs(dct_result[:,:,i]) + 1), cmap='viridis')
        plt.title(f'{title} - {channel_names[i]} Channel')
        plt.colorbar()
    
    plt.tight_layout()

def perform_dct_analysis(img1, img2):
    """Perform DCT analysis on both images and their difference for each channel."""
    # Compute DCT for both images
    dct1 = perform_channel_dct(img1)
    dct2 = perform_channel_dct(img2)
    
    # Compute difference of DCT coefficients
    dct_diff = np.abs(dct1 - dct2)
    
    # Visualize DCT coefficients for each image
    visualize_channel_dct(dct1, 'DCT Image 1')
    visualize_channel_dct(dct2, 'DCT Image 2')
    
    # Visualize DCT differences
    visualize_channel_dct(dct_diff, 'DCT Difference')
    
    return dct1, dct2, dct_diff

def analyze_frequency_distribution(dct1, dct2):
    """Analyze and plot the frequency distribution of DCT coefficients for each channel."""
    channel_names = ['Blue', 'Green', 'Red']
    plt.figure(figsize=(15, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        
        dct1_flat = np.abs(dct1[:,:,i]).flatten()
        dct2_flat = np.abs(dct2[:,:,i]).flatten()
        
        plt.hist(np.log(dct1_flat + 1), bins=50, alpha=0.5, label='Image 1', density=True)
        plt.hist(np.log(dct2_flat + 1), bins=50, alpha=0.5, label='Image 2', density=True)
        
        plt.title(f'{channel_names[i]} Channel DCT Distribution')
        plt.xlabel('Log(Coefficient + 1)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return dct1, dct2

def main(image1_path, image2_path):
    """Main function to run the analysis."""
    # Load and preprocess images
    img1, img2 = load_and_preprocess_images(image1_path, image2_path)
    
    # Create difference heatmap
    diff_mean = create_difference_heatmap(img1, img2)
    
    # Perform DCT analysis
    dct1, dct2, dct_diff = perform_dct_analysis(img1, img2)
    
    # Analyze frequency distribution
    # analyze_frequency_distribution(dct1, dct2)
    
    # Show all plots
    plt.show()
    
    # Calculate channel-wise statistics
    channel_names = ['Blue', 'Green', 'Red']
    results = {
        'pixel_diff_mean': np.mean(diff_mean)
    }
    
    for i, channel in enumerate(channel_names):
        results.update({
            f'{channel}_dct_diff_mean': np.mean(np.abs(dct1[:,:,i] - dct2[:,:,i])),
            f'{channel}_dct1_energy': np.sum(np.abs(dct1[:,:,i])),
            f'{channel}_dct2_energy': np.sum(np.abs(dct2[:,:,i]))
        })
    
    return results

if __name__ == "__main__":
    results = main(IMAGE1_PATH, IMAGE2_PATH)
    
    print("\nAnalysis Results:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")