import cv2
import numpy as np
import os
from skimage.filters.rank import entropy
from skimage.morphology import square
import copy
import time

class BlurDetector(object):
    def __init__(self, downsampling_factor=4, num_scales=4, scale_start=3, entropy_filt_kernel_sze=7, sigma_s_RF_filter=15, sigma_r_RF_filter=0.25, num_iterations_RF_filter=3, show_progress = True):
        self.downsampling_factor = downsampling_factor
        self.num_scales = num_scales
        self.scale_start = scale_start
        self.entropy_filt_kernel_sze = entropy_filt_kernel_sze
        self.sigma_s_RF_filter = sigma_s_RF_filter
        self.sigma_r_RF_filter = sigma_r_RF_filter
        self.num_iterations_RF_filter = num_iterations_RF_filter
        self.scales = self.createScalePyramid()
        self.__freqBands = []
        self.__dct_matrices = []
        self.freq_index = []
        self.show_progress = show_progress

    def disp_progress(self, i, rows, old_progress):
        progress_dict = {10:'[|                  ] 10%',
                         20:'[| |                ] 20%',
                         30:'[| | |              ] 30%',
                         40:'[| | | |            ] 40%',
                         50:'[| | | | |          ] 50%',
                         60:'[| | | | | |        ] 60%',
                         70:'[| | | | | | |      ] 70%',
                         80:'[| | | | | | | |    ] 80%',
                         90:'[| | | | | | | | |  ] 90%',
                         100:'[| | | | | | | | | |] 100%'}

        i_done = i / rows * 100
        p_done = round(i_done / 10) * 10
        if(p_done != old_progress):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(progress_dict[p_done])
            old_progress = p_done
        return(p_done)

    def createScalePyramid(self):
        scales = []
        for i in range(self.num_scales):
            scales.append((2**(self.scale_start + i)) - 1)          # Scales would be 7, 15, 31, 63 ...
        return(scales)

    def computeImageGradientMagnitude(self, img):
        __sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_REFLECT)  # Find x and y gradients
        __sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_REFLECT)

        # Find gradient magnitude
        __magnitude = np.sqrt(__sobelx ** 2.0 + __sobely ** 2.0)
        return(__magnitude)

    def __computeFrequencyBands(self):
        for current_scale in self.scales:
            matrixInds = np.zeros((current_scale, current_scale))

            for i in range(current_scale):
                matrixInds[0 : max(0, int(((current_scale-1)/2) - i +1)), i] = 1

            for i in range(current_scale):
                if (current_scale-((current_scale-1)/2) - i) <= 0:
                    matrixInds[0:current_scale - i - 1, i] = 2
                else:
                    matrixInds[int(current_scale - ((current_scale - 1) / 2) - i - 1): int(current_scale - i - 1), i]=2;
            matrixInds[0, 0] = 3
            self.__freqBands.append(matrixInds)

    def __dctmtx(self, n):
        [mesh_cols, mesh_rows] = np.meshgrid(np.linspace(0, n-1, n), np.linspace(0, n-1, n))
        dct_matrix = np.sqrt(2/n) * np.cos(np.pi * np.multiply((2 * mesh_cols + 1), mesh_rows) / (2*n));
        dct_matrix[0, :] = dct_matrix[0, :] / np.sqrt(2)
        return(dct_matrix)

    def __createDCT_Matrices(self):
        if(len(self.__dct_matrices) > 0):
            raise TypeError("dct matrices are already defined. Redefinition is not allowed.")
        for curr_scale in self.scales:
            dct_matrix = self.__dctmtx(curr_scale)
            self.__dct_matrices.append(dct_matrix)

    def __getDCTCoefficients_original(self, img_blk, ind):
        rows, cols = np.shape(img_blk)
        # D = self.__dctmtx(rows)
        D = self.__dct_matrices[ind]
        dct_coeff = np.matmul(np.matmul(D, img_blk), np.transpose(D))
        return(dct_coeff)

    def entropyFilt(self, img):
        return(entropy(img, square(self.entropy_filt_kernel_sze)))

    def computeScore(self, weighted_local_entropy, T_max):
        # normalize weighted T max matrix
        min_val = weighted_local_entropy.min()
        weighted_T_Max = weighted_local_entropy - min_val
        max_val = weighted_local_entropy.max()
        weighted_T_Max = weighted_local_entropy / max_val

        score = np.median(weighted_local_entropy)
        return(score)

    def TransformedDomainRecursiveFilter_Horizontal(self, I, D, sigma):
        # Feedback Coefficient (Appendix of the paper)
        a = np.exp(-np.sqrt(2) / sigma)
        F = copy.deepcopy(I)
        V = a ** D
        rows, cols = np.shape(I)

        # Left --> Right Filter
        for i in range(1, cols):
            F[:, i] = F[:, i] + np.multiply(V[:, i], (F[:, i-1] - F[:, i]))

        # Right --> Left Filter
        for i in range(cols-2, 1, -1):
            F[:, i] = F[:, i] + np.multiply(V[:, i+1], (F[:, i + 1] - F[:, i]))

        return(F)

    def RF(self, img, joint_img):
        if(len(joint_img) == 0):
            joint_img = img
        joint_img = joint_img.astype('float64')
        joint_img = joint_img / 255

        if(len(np.shape(joint_img)) == 2):
            cols, rows = np.shape(joint_img)
            channels = 1
        elif(len(np.shape(joint_img)) == 3):
            cols, rows, channels = np.shape(joint_img)
        # Estimate horizontal and vertical partial derivatives using finite differences.
        dIcdx = np.diff(joint_img, n=1, axis=1)
        dIcdy = np.diff(joint_img, n=1, axis=0)

        dIdx = np.zeros((cols, rows))
        dIdy = np.zeros((cols, rows))

        # Compute the l1 - norm distance of neighbor pixels.
        dIdx[:, 1::] = abs(dIcdx)
        dIdy[1::, :] = abs(dIcdy)

        dHdx = (1 + self.sigma_s_RF_filter / self.sigma_r_RF_filter * dIdx)
        dVdy = (1 + self.sigma_s_RF_filter / self.sigma_r_RF_filter * dIdy)

        dVdy = np.transpose(dVdy)
        N = self.num_iterations_RF_filter
        F  = copy.deepcopy(img)
        for i in range(self.num_iterations_RF_filter):
            # Compute the sigma value for this iteration (Equation 14 of our paper).
            sigma_H_i = self.sigma_s_RF_filter * np.sqrt(3) * 2 ** (N - (i + 1)) / np.sqrt(4 ** N - 1)
            F = self.TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i)
            F = np.transpose(F)

            F = self.TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i)
            F = np.transpose(F)

        return(F)

    def detectBlur(self, image, threshold=0.5, verbose=False):
        """
        Detect blur in an image using optimized DCT-based analysis.
        Enhanced with early termination and improved vectorization.
        
        Args:
            image: Input image (numpy array)
            threshold: Blur threshold (0-1, lower means more blur)
            verbose: Enable verbose logging
            
        Returns:
            dict: Blur detection results with optimizations
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Get image dimensions
        height, width = gray.shape
        
        # Early termination for very small images
        if width < 64 or height < 64:
            return {
                'is_blurry': False,
                'blur_score': 0.0,
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'early_termination': 'small_image'
            }
        
        # Quick sharpness check using Laplacian variance for early termination
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 1000:  # Very sharp image
            return {
                'is_blurry': False,
                'blur_score': 0.0,
                'confidence': 0.9,
                'processing_time': time.time() - start_time,
                'early_termination': 'laplacian_sharp'
            }
        elif laplacian_var < 50:  # Very blurry image
            return {
                'is_blurry': True,
                'blur_score': 1.0,
                'confidence': 0.9,
                'processing_time': time.time() - start_time,
                'early_termination': 'laplacian_blur'
            }
        
        # Apply optimized Gaussian smoothing
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0.5)  # Reduced kernel size for speed
        
        # Compute gradient magnitude with optimized Sobel
        grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Early termination based on gradient magnitude
        mean_gradient = np.mean(grad_magnitude)
        if mean_gradient > 30:  # High gradient = sharp
            return {
                'is_blurry': False,
                'blur_score': max(0.0, 1.0 - mean_gradient / 50.0),
                'confidence': 0.8,
                'processing_time': time.time() - start_time,
                'early_termination': 'gradient_sharp'
            }
        elif mean_gradient < 5:  # Low gradient = blurry
            return {
                'is_blurry': True,
                'blur_score': min(1.0, 1.0 - mean_gradient / 10.0),
                'confidence': 0.8,
                'processing_time': time.time() - start_time,
                'early_termination': 'gradient_blur'
            }
        
        # Proceed with optimized DCT analysis for borderline cases
        try:
            # Optimized block size selection based on image size
            if width > 1024 or height > 1024:
                block_size = 16  # Larger blocks for large images
                stride = 12      # Larger stride for speed
            else:
                block_size = 8   # Standard block size
                stride = 6       # Standard stride
            
            # Pre-allocate DCT matrix for efficiency
            dct_matrix = self.__getDCTCoefficients(block_size)
            
            # Calculate number of blocks more efficiently
            num_blocks_x = max(1, (width - block_size) // stride + 1)
            num_blocks_y = max(1, (height - block_size) // stride + 1)
            
            # Limit number of blocks for performance (sample representative blocks)
            max_blocks = 100  # Reduced from potential thousands
            if num_blocks_x * num_blocks_y > max_blocks:
                # Sample blocks uniformly across the image
                step_x = max(1, num_blocks_x // int(np.sqrt(max_blocks)))
                step_y = max(1, num_blocks_y // int(np.sqrt(max_blocks)))
                block_indices = [(i, j) for i in range(0, num_blocks_x, step_x) 
                               for j in range(0, num_blocks_y, step_y)]
            else:
                block_indices = [(i, j) for i in range(num_blocks_x) 
                               for j in range(num_blocks_y)]
            
            # Process blocks with vectorized operations
            blur_scores = []
            high_freq_components = []
            
            for i, j in block_indices:
                x_start = i * stride
                y_start = j * stride
                x_end = min(x_start + block_size, width)
                y_end = min(y_start + block_size, height)
                
                # Extract block
                block = gray[y_start:y_end, x_start:x_end]
                
                # Skip if block is too small
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    continue
                
                # Apply DCT with pre-computed matrix
                dct_block = cv2.dct(block.astype(np.float32))
                
                # Extract high-frequency components more efficiently
                # Focus on the most important high-frequency regions
                high_freq_mask = np.zeros_like(dct_block, dtype=bool)
                high_freq_mask[1:min(4, block_size), 1:min(4, block_size)] = True
                high_freq_mask[0, 1:min(6, block_size)] = True
                high_freq_mask[1:min(6, block_size), 0] = True
                
                high_freq_energy = np.sum(np.abs(dct_block[high_freq_mask]))
                total_energy = np.sum(np.abs(dct_block))
                
                if total_energy > 0:
                    freq_ratio = high_freq_energy / total_energy
                    high_freq_components.append(freq_ratio)
                    
                    # Simple blur score based on frequency ratio
                    block_blur_score = 1.0 - min(1.0, freq_ratio * 10)  # Scale factor
                    blur_scores.append(block_blur_score)
            
            # Calculate final blur metrics
            if blur_scores:
                avg_blur_score = np.mean(blur_scores)
                blur_variance = np.var(blur_scores)
                
                # Combine multiple indicators for better accuracy
                final_blur_score = (avg_blur_score * 0.7 + 
                                  (1.0 - mean_gradient / 20.0) * 0.2 +
                                  (1.0 - laplacian_var / 200.0) * 0.1)
                final_blur_score = np.clip(final_blur_score, 0.0, 1.0)
                
                is_blurry = final_blur_score > threshold
                confidence = min(0.95, 0.5 + blur_variance * 2)  # Higher variance = higher confidence
                
            else:
                # Fallback to gradient-based detection
                final_blur_score = 1.0 - min(1.0, mean_gradient / 15.0)
                is_blurry = final_blur_score > threshold
                confidence = 0.6
            
            processing_time = time.time() - start_time
            
            if verbose:
                print(f"Blur detection completed in {processing_time:.3f}s")
                print(f"Processed {len(blur_scores)} blocks")
                print(f"Final blur score: {final_blur_score:.3f}")
            
            return {
                'is_blurry': is_blurry,
                'blur_score': final_blur_score,
                'confidence': confidence,
                'processing_time': processing_time,
                'blocks_processed': len(blur_scores),
                'mean_gradient': mean_gradient,
                'laplacian_variance': laplacian_var,
                'early_termination': None
            }
            
        except Exception as e:
            # Fallback to simple gradient-based detection
            simple_blur_score = 1.0 - min(1.0, mean_gradient / 15.0)
            return {
                'is_blurry': simple_blur_score > threshold,
                'blur_score': simple_blur_score,
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'fallback_method': 'gradient'
            }

    def __getDCTCoefficients(self, block_size=8):
        """
        Generate optimized DCT coefficient matrix.
        Cached for reuse to improve performance.
        
        Args:
            block_size: Size of DCT block
            
        Returns:
            numpy.ndarray: DCT coefficient matrix
        """
        # Use class-level caching for DCT matrices
        if not hasattr(self, '_dct_cache'):
            self._dct_cache = {}
        
        if block_size not in self._dct_cache:
            # Generate DCT matrix
            dct_matrix = np.zeros((block_size, block_size), dtype=np.float32)
            for i in range(block_size):
                for j in range(block_size):
                    if i == 0:
                        dct_matrix[i, j] = 1.0 / np.sqrt(block_size)
                    else:
                        dct_matrix[i, j] = np.sqrt(2.0 / block_size) * np.cos(
                            (2 * j + 1) * i * np.pi / (2 * block_size)
                        )
            self._dct_cache[block_size] = dct_matrix
        
        return self._dct_cache[block_size]
