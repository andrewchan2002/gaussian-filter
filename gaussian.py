import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

image_path = "dog.jpg"
original_image = cv2.imread(image_path)
# Function to apply Gaussian filter and compute PSNR
def apply_gaussian_filter_and_psnr(image, filter_size):
    # Create Gaussian filter
    gaussian_filter = cv2.getGaussianKernel(filter_size, 0)
    gaussian_filter = gaussian_filter * gaussian_filter.T

    # Apply filter to each channel
    filtered_image = np.zeros_like(image)
    for i in range(3):
        padded_channel = np.pad(image[:, :, i], ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2)), mode='constant')
        # Apply convolution
        filtered_channel = convolution(padded_channel, gaussian_filter)
        # Trim the result to the original image size
        filtered_image[:, :, i] = filtered_channel[:image.shape[0], :image.shape[1]]

    # Compute PSNR
    psnr = peak_signal_noise_ratio(image, filtered_image)
    return filtered_image, psnr
def convolution(image_channel, kernel):
    m, n = kernel.shape  # Get the dimensions of the kernel
    y, x = image_channel.shape   # Get the dimensions of the input image channel
    # Calculate the output dimensions after convolution
    y = y - m + 1  
    x = x - n + 1 
    # Initialize an array to store the result of the convolution
    result = np.zeros((y, x))
     # Iterate over the input image channel
    for i in range(y):
        for j in range(x):
            result[i][j] = np.sum(image_channel[i:i+m, j:j+n] * kernel[:image_channel[i:i+m, j:j+n].shape[0], :image_channel[i:i+m, j:j+n].shape[1]])

    return result
# Sizes of Gaussian filters
filter_sizes = [3, 7, 11]
# Apply filters and compute PSNR
for size in filter_sizes:
    filtered_image, psnr = apply_gaussian_filter_and_psnr(original_image, size)
    # Save the filtered image
    cv2.imwrite(f"filtered_image_{size}x{size}.jpg", filtered_image)
    # Print PSNR
    print(f"PSNR for {size}x{size} filter: {psnr:.4f}dB")

#part b
def gaussian_filter(size, sigma):
    #Generates a Gaussian filter with the given size and sigma.
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, kernel)
def image_convolution(image, sigma):
    
    #Applies image convolution using a Gaussian kernel with the given sigma.

    filter_size = 3  # 3x3 Gaussian filter by default, you can adjust this
    kernel = gaussian_filter(filter_size, sigma)
    height, width, channels = image.shape
    k_size = len(kernel)
    k_radius = k_size // 2
    # Create an empty result image
    result = np.zeros_like(image, dtype=np.float32)
    # Perform convolution for each channel
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                for m in range(k_size):
                    for n in range(k_size):
                        # Handle boundary conditions with zero-padding
                        ii = i - k_radius + m
                        jj = j - k_radius + n
                        if ii >= 0 and ii < height and jj >= 0 and jj < width:
                            result[i, j, c] += image[ii, jj, c] * kernel[m, n]
    
    return result.astype(np.uint8)

# Load the image
image = cv2.imread(image_path)
# Generate Gaussian filters with sigma = 1, 10, and 30
sigma_values = [1, 10, 30]
filter_size = 3  # 3x3 Gaussian filter

for sigma in sigma_values:
    # Apply convolution
    smoothed_image = image_convolution(image, sigma)
    # Compute PSNR
    psnr_value = peak_signal_noise_ratio(image, smoothed_image)
    
    # Display or save results
    cv2.imwrite(f'Smoothed_Image_Sigma_{sigma}.jpg', smoothed_image)
    print(f'PSNR (Sigma={sigma}): {psnr_value:.4f} dB')
    
#part d

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Function for manual convolution with padding
def convolution_d(image_channel, kernel):
    m, n = kernel.shape
    y, x = image_channel.shape[:2]
    
    # Add padding to the image
    padded_image = np.pad(image_channel, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    result = np.zeros((y, x), dtype=np.float64)

    for i in range(y):
        for j in range(x):
            result[i][j] = np.sum(padded_image[i:i+m, j:j+n] * kernel)

    return result

# Apply convolution to each color channel
unsharp_mask_channels = [convolution_d(image[:, :, i], kernel) for i in range(3)]
# Stack the channels to form the output image
unsharp_mask = np.stack(unsharp_mask_channels, axis=-1)
# Save Unsharp Mask result
cv2.imwrite("unsharp_mask_result.jpg", unsharp_mask)
# Apply Edge Detection to each color channel
_, edges = cv2.threshold(unsharp_mask, 30, 255, cv2.THRESH_BINARY)
# Save Edge Detection result
cv2.imwrite("edge_detection_result.jpg", edges)
