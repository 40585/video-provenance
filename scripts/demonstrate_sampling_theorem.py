import numpy as np
import matplotlib.pyplot as plt

# Setup parameters
N = 64  # samples
sampling_rate = N  # Hz
nyquist_freq = sampling_rate / 2  # Hz

# Create time vectors
t_sparse = np.linspace(0, 1, N)  # Sampling points
t_dense = np.linspace(0, 1, 1000)  # Dense points for "true" signal

# Original high frequencies
f1_high, f2_high = 40, 80  # Hz

# Calculate aliased frequencies
def get_aliased_freq(f, fs):
    """Calculate the aliased frequency"""
    return abs(fs * round(f/fs) - f)

f1_alias = get_aliased_freq(f1_high, sampling_rate)
f2_alias = get_aliased_freq(f2_high, sampling_rate)

# Create true high-frequency signal and get its samples
true_signal = 2*np.sin(2*np.pi*f1_high*t_dense) + 0.5*np.sin(2*np.pi*f2_high*t_dense)
sampled_points = 2*np.sin(2*np.pi*f1_high*t_sparse) + 0.5*np.sin(2*np.pi*f2_high*t_sparse)

# Create the aliased signal using the same samples
aliased_signal = 2*np.sin(2*np.pi*f1_alias*t_dense) + 0.5*np.sin(2*np.pi*f2_alias*t_dense)

# Verify alignment by interpolating through samples
from scipy.interpolate import interp1d
interpolator = interp1d(t_sparse, sampled_points, kind='cubic')
reconstructed_signal = interpolator(t_dense)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
plt.suptitle(f'Aliasing Demonstration\nSampling Rate: {sampling_rate} Hz, Nyquist Frequency: {nyquist_freq} Hz', 
             fontsize=16)

# Plot original high-frequency signal
ax1.plot(t_dense, true_signal, 'b-', label=f'True Signal (f1={f1_high}Hz, f2={f2_high}Hz)', alpha=0.7)
ax1.plot(t_sparse, sampled_points, 'r.', label='Sampled Points', markersize=10)
ax1.set_title('Original High-Frequency Signal')
ax1.grid(True)
ax1.legend()
ax1.set_ylim(-3, 3)

# Plot reconstructed signal from samples
ax2.plot(t_dense, reconstructed_signal, 'g-', 
         label=f'Reconstructed Signal (appears as f1={f1_alias:.1f}Hz, f2={f2_alias:.1f}Hz)', alpha=0.7)
ax2.plot(t_sparse, sampled_points, 'r.', label='Sampled Points', markersize=10)
ax2.set_title('What We Actually See (Signal Reconstructed from Samples)')
ax2.grid(True)
ax2.legend()
ax2.set_ylim(-3, 3)

# Plot both together
ax3.plot(t_dense, true_signal, 'b-', label='True High-Frequency Signal', alpha=0.4)
ax3.plot(t_dense, reconstructed_signal, 'g-', label='Reconstructed Signal', alpha=0.4)
ax3.plot(t_sparse, sampled_points, 'r.', label='Sampled Points', markersize=10)
ax3.set_title('Both Signals Together - Same Samples!')
ax3.grid(True)
ax3.legend()
ax3.set_ylim(-3, 3)

plt.tight_layout()
plt.show()

# Print analysis
print("\nAliasing Analysis:")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Nyquist frequency: {nyquist_freq} Hz")
print("\nOriginal frequencies:")
print(f"f1 = {f1_high} Hz")
print(f"f2 = {f2_high} Hz")
print("\nThese appear as:")
print(f"f1 appears as {f1_alias:.1f} Hz")
print(f"f2 appears as {f2_alias:.1f} Hz")