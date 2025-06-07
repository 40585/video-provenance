import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.fft import dct, idct

class PerceptualQualityValidator:
    def __init__(self):
        # CSF parameters (Contrast Sensitivity Function)
        self.csf_params = {
            'peak_sensitivity': 250.0,
            'peak_frequency': 3.0,
            'min_sensitivity': 0.1,
            'bandwidth': 1.5
        }
        
    def csf_weight(self, frequencies):
        """Calculate CSF weights based on spatial frequencies."""
        # Simplified CSF model based on Mannos & Sakrison
        freq_magnitude = np.sqrt(frequencies[0]**2 + frequencies[1]**2)
        csf = self.csf_params['peak_sensitivity'] * np.exp(
            -(freq_magnitude - self.csf_params['peak_frequency'])**2 / 
            (2 * self.csf_params['bandwidth']**2)
        )
        return np.maximum(csf, self.csf_params['min_sensitivity'])

    def compute_perceptual_features(self, img):
        """Compute comprehensive perceptual features."""
        # Convert to YUV
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:,:,0].astype(np.float32) / 255.0

        features = {}
        
        # 1. Multi-scale contrast sensitivity
        features['contrast'] = self._compute_contrast_sensitivity(y_channel)
        
        # 2. Local structure analysis
        features['structure'] = self._analyze_local_structure(y_channel)
        
        # 3. Edge coherence
        features['edges'] = self._analyze_edge_coherence(y_channel)
        
        # 4. Texture analysis
        features['texture'] = self._analyze_texture(y_channel)
        
        # 5. Frequency domain analysis
        features['frequency'] = self._analyze_frequency_domain(y_channel)
        
        return features

    def _compute_contrast_sensitivity(self, y_channel):
        """Compute contrast sensitivity at multiple scales."""
        scales = [1, 2, 4, 8]
        contrasts = []
        
        for scale in scales:
            # Gaussian blur for each scale
            blurred = gaussian_filter(y_channel, sigma=scale)
            
            # Local contrast computation
            window_size = 2 * scale + 1
            pad_size = window_size // 2
            padded = np.pad(blurred, pad_size, mode='reflect')
            
            local_std = np.zeros_like(y_channel)
            local_mean = np.zeros_like(y_channel)
            
            for i in range(y_channel.shape[0]):
                for j in range(y_channel.shape[1]):
                    window = padded[i:i+window_size, j:j+window_size]
                    local_std[i,j] = np.std(window)
                    local_mean[i,j] = np.mean(window)
            
            # RMS contrast
            rms_contrast = local_std / (local_mean + 1e-6)
            contrasts.append(rms_contrast)
        
        return {
            'multi_scale_contrast': contrasts,
            'contrast_energy': [np.mean(c**2) for c in contrasts]
        }

    def _analyze_local_structure(self, y_channel):
        """Analyze local structure using SSIM-like measurements."""
        window_size = 8
        pad_size = window_size // 2
        padded = np.pad(y_channel, pad_size, mode='reflect')
        
        # Structure tensors
        grad_x = cv2.Sobel(y_channel, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(y_channel, cv2.CV_32F, 0, 1)
        
        # Compute structure tensor components
        Jxx = gaussian_filter(grad_x * grad_x, 1.5)
        Jxy = gaussian_filter(grad_x * grad_y, 1.5)
        Jyy = gaussian_filter(grad_y * grad_y, 1.5)
        
        # Compute eigenvalues
        lambda1 = 0.5 * ((Jxx + Jyy) + np.sqrt((Jxx - Jyy)**2 + 4*Jxy**2))
        lambda2 = 0.5 * ((Jxx + Jyy) - np.sqrt((Jxx - Jyy)**2 + 4*Jxy**2))
        
        return {
            'coherence': (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6),
            'strength': lambda1 + lambda2,
            'orientation': 0.5 * np.arctan2(2*Jxy, Jxx - Jyy)
        }

    def _analyze_edge_coherence(self, y_channel):
        """Analyze edge coherence using phase congruency."""
        # Compute gradients at multiple scales
        scales = [1, 2, 4]
        orientations = [0, 45, 90, 135]
        
        edge_maps = []
        for scale in scales:
            scaled_edges = []
            for theta in orientations:
                # Create oriented filter
                kernel_size = 2 * scale + 1
                kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size), 
                    sigma=scale, 
                    theta=theta * np.pi/180,
                    lambd=kernel_size,
                    gamma=0.5,
                    psi=0
                )
                filtered = cv2.filter2D(y_channel, cv2.CV_32F, kernel)
                scaled_edges.append(filtered)
            
            edge_maps.append(scaled_edges)
        
        return {
            'edge_maps': edge_maps,
            'edge_strength': [np.max(np.abs(em), axis=0) for em in edge_maps],
            'edge_coherence': [np.std([em for em in scale_edges], axis=0) 
                             for scale_edges in edge_maps]
        }

    def _analyze_texture(self, y_channel):
        """Advanced texture analysis using multiple descriptors."""
        # LBP-like texture descriptor
        def compute_lbp(img, points=8, radius=1):
            lbp = np.zeros_like(img)
            for i in range(points):
                theta = 2 * np.pi * i / points
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                
                # Bilinear interpolation
                x1 = int(np.floor(x))
                x2 = x1 + 1
                y1 = int(np.floor(y))
                y2 = y1 + 1
                
                # Get values using shift and compare
                shifted = np.roll(np.roll(img, y1, axis=0), x1, axis=1)
                lbp += (img > shifted) * (1 << i)
            
            return lbp
        
        # Compute textures at multiple scales
        texture_scales = [1, 2, 4]
        texture_features = []
        
        for scale in texture_scales:
            # Smooth image for each scale
            smoothed = gaussian_filter(y_channel, scale)
            
            # Compute LBP
            lbp = compute_lbp(smoothed)
            
            # Compute GLCM features
            glcm = self._compute_glcm(smoothed)
            
            texture_features.append({
                'lbp_histogram': np.histogram(lbp, bins=256)[0],
                'glcm_contrast': np.sum(glcm * np.square(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])[:, None])),
                'glcm_homogeneity': np.sum(glcm / (1 + np.square(np.arange(glcm.shape[0]) - np.arange(glcm.shape[1])[:, None])))
            })
        
        return {
            'multi_scale_texture': texture_features
        }

    def _compute_glcm(self, img, levels=8):
        """Compute Gray Level Co-occurrence Matrix."""
        # Quantize image to fewer levels
        quantized = np.floor(img * levels).astype(np.int32)
        glcm = np.zeros((levels, levels))
        
        # Compute GLCM for horizontal direction
        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1]-1):
                glcm[quantized[i,j], quantized[i,j+1]] += 1
        
        # Normalize
        return glcm / np.sum(glcm)

    def _analyze_frequency_domain(self, y_channel):
        """Analyze frequency domain characteristics."""
        # Compute 2D DCT
        dct_coeffs = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')
        
        # Create frequency grid
        freq_y, freq_x = np.meshgrid(
            np.fft.fftfreq(dct_coeffs.shape[0]),
            np.fft.fftfreq(dct_coeffs.shape[1])
        )
        
        # Apply CSF weights
        csf_weights = self.csf_weight((freq_x, freq_y))
        weighted_coeffs = dct_coeffs * csf_weights
        
        # Analyze frequency bands
        bands = {
            'low': np.sum(np.abs(weighted_coeffs[:8, :8])),
            'mid': np.sum(np.abs(weighted_coeffs[8:32, 8:32])),
            'high': np.sum(np.abs(weighted_coeffs[32:, 32:]))
        }
        
        return {
            'frequency_bands': bands,
            'csf_weighted_energy': np.sum(np.abs(weighted_coeffs))
        }

    def compare_images(self, img1, img2):
        """Compare two images using all perceptual features."""
        features1 = self.compute_perceptual_features(img1)
        features2 = self.compute_perceptual_features(img2)
        
        comparison = {}
        
        # Compare contrast sensitivity
        comparison['contrast'] = {
            'energy_difference': [
                np.abs(e1 - e2) for e1, e2 in 
                zip(features1['contrast']['contrast_energy'],
                    features2['contrast']['contrast_energy'])
            ]
        }
        
        # Compare structure
        comparison['structure'] = {
            'coherence_difference': np.mean(np.abs(
                features1['structure']['coherence'] - 
                features2['structure']['coherence']
            )),
            'orientation_similarity': np.mean(np.cos(
                features1['structure']['orientation'] - 
                features2['structure']['orientation']
            ))
        }
        
        # Compare frequency characteristics
        comparison['frequency'] = {
            'band_differences': {
                band: abs(features1['frequency']['frequency_bands'][band] - 
                         features2['frequency']['frequency_bands'][band])
                for band in ['low', 'mid', 'high']
            }
        }
        
        return comparison

def validate_perceptual_quality(img1, img2):
    """Main function to validate perceptual quality."""
    validator = PerceptualQualityValidator()
    comparison = validator.compare_images(img1, img2)
    
    # Define thresholds for each metric
    thresholds = {
        'contrast_energy_diff': 0.1,
        'structure_coherence_diff': 0.2,
        'orientation_similarity': 0.8,
        'frequency_band_diff': {
            'low': 0.1,
            'mid': 0.2,
            'high': 0.3
        }
    }
    
    # Validate against thresholds
    validation = {
        'passed': all([
            all(diff < thresholds['contrast_energy_diff'] 
                for diff in comparison['contrast']['energy_difference']),
            comparison['structure']['coherence_difference'] < thresholds['structure_coherence_diff'],
            comparison['structure']['orientation_similarity'] > thresholds['orientation_similarity'],
            all(comparison['frequency']['band_differences'][band] < thresholds['frequency_band_diff'][band]
                for band in ['low', 'mid', 'high'])
        ]),
        'metrics': comparison
    }
    
    return validation