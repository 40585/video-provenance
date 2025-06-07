import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from scipy.signal import resample

def zero_pad_dct(image, target_size):
    """Pad the DCT coefficients with zeros to achieve larger DCT size"""
    original_size = image.shape[0]
    padded = np.zeros((target_size, target_size))
    dct_output = dct(dct(image.T, norm='ortho').T, norm='ortho')
    padded[:original_size, :original_size] = dct_output
    return padded

def reconstruct_from_padded(padded_dct, original_size):
    """Reconstruct image from padded DCT coefficients"""
    # Take only the coefficients corresponding to original size
    dct_output = padded_dct[:original_size, :original_size]
    return idct(idct(dct_output.T, norm='ortho').T, norm='ortho')

# Create a test image with high frequency components
size = 256
x = np.linspace(0, 1, size)
y = np.linspace(0, 1, size)
X, Y = np.meshgrid(x, y)
image = np.sin(2*np.pi*30*X) * np.sin(2*np.pi*30*Y)  # High frequency pattern

# Try different DCT sizes
dct_sizes = [256, 512, 1024]
reconstructions = []
energy_preservations = []

# Calculate original image energy
original_energy = np.sum(image**2)

# Process with different DCT sizes
for dct_size in dct_sizes:
    # Pad DCT
    padded_dct = zero_pad_dct(image, dct_size)
    
    # Reconstruct
    reconstructed = reconstruct_from_padded(padded_dct, size)
    reconstructions.append(reconstructed)
    
    # Calculate energy preservation
    reconstructed_energy = np.sum(reconstructed**2)
    energy_preservation = (reconstructed_energy / original_energy) * 100
    energy_preservations.append(energy_preservation)

# Now let's try to find minimum DCT size for target energy preservation
def find_min_dct_size(image, target_energy_preservation, max_size=2048):
    original_energy = np.sum(image**2)
    current_size = image.shape[0]
    
    while current_size <= max_size:
        padded_dct = zero_pad_dct(image, current_size)
        reconstructed = reconstruct_from_padded(padded_dct, image.shape[0])
        reconstructed_energy = np.sum(reconstructed**2)
        energy_preservation = (reconstructed_energy / original_energy) * 100
        
        if energy_preservation >= target_energy_preservation:
            return current_size, energy_preservation
        
        current_size *= 2
    
    return None, None

# Find minimum size for 99.9% energy preservation
target_preservation = 99.9
min_size, achieved_preservation = find_min_dct_size(image, target_preservation)

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plt.suptitle(f'DCT Energy Preservation Analysis\nOriginal Image Size: {size}x{size}', fontsize=16)

# Original image
axes[0,0].imshow(image, cmap='viridis')
axes[0,0].set_title('Original Image')

# Plot reconstructions for different DCT sizes
for idx, (dct_size, reconstructed, energy) in enumerate(zip(dct_sizes, reconstructions, energy_preservations)):
    axes[0 if idx < 2 else 1, (idx+1)%3].imshow(reconstructed, cmap='viridis')
    axes[0 if idx < 2 else 1, (idx+1)%3].set_title(f'DCT Size: {dct_size}x{dct_size}\nEnergy Preserved: {energy:.3f}%')

# Print analysis
print("\nEnergy Preservation Analysis:")
print(f"Original image size: {size}x{size}")
for dct_size, energy in zip(dct_sizes, energy_preservations):
    print(f"\nDCT Size: {dct_size}x{dct_size}")
    print(f"Energy preserved: {energy:.3f}%")
    print(f"Energy lost: {100-energy:.3f}%")

if min_size:
    print(f"\nMinimum DCT size for {target_preservation}% energy preservation:")
    print(f"Size: {min_size}x{min_size}")
    print(f"Achieved preservation: {achieved_preservation:.3f}%")
else:
    print(f"\nCould not achieve {target_preservation}% energy preservation within max size limit")

plt.tight_layout()
plt.show()