# Gaussian Filter and Image Processing with PSNR: README

This Python script processes an image using various image filtering techniques, evaluates the output using Peak Signal-to-Noise Ratio (PSNR), and applies both unsharp masking and edge detection to enhance image details and edges. The script is based on the image **dog.jpg** and explores how different filter sizes and sigma values affect image quality.

## Code Functionality:
### (a) **apply_gaussian_filter_and_psnr(image, filter_size)**
- **Purpose**: This function applies a Gaussian filter to the input image with the specified filter size (e.g., 3x3, 7x7, 11x11) and calculates the PSNR between the original and the filtered image.
- **Steps**:
  - Creates a Gaussian filter of a given size using `cv2.getGaussianKernel`.
  - Pads the image to ensure proper convolution without losing edge information.
  - Performs convolution (filtering) channel by channel on the image (i.e., for RGB channels).
  - Computes the PSNR value to assess the quality of the filtered image.
- **Returns**: The filtered image and its PSNR value.
  
### (b) **convolution(image_channel, kernel)**
- **Purpose**: This function performs convolution manually by applying the Gaussian filter (or kernel) to a padded image channel. It slides the filter over the image and multiplies the corresponding pixel values to calculate the output for each pixel.
- **Steps**:
  - The image channel is padded to handle boundary pixels.
  - For each pixel in the image, the function applies the filter to calculate the new pixel value by multiplying the filter's kernel with the corresponding pixel region.
- **Returns**: The result of the convolution for the specific image channel.

### (c) **gaussian_filter(size, sigma)**
- **Purpose**: Generates a Gaussian filter with a specified size and standard deviation (sigma). This function computes a 2D Gaussian kernel by multiplying two 1D Gaussian distributions.
- **Returns**: A Gaussian filter to be used for image convolution.

### (d) **image_convolution(image, sigma)**
- **Purpose**: This function applies a Gaussian filter to the entire image using a specific sigma value. It convolves the filter with the image to smooth it.
- **Steps**:
  - Uses the `gaussian_filter()` function to create a Gaussian filter based on sigma.
  - Applies the filter to each channel of the image using convolution.
- **Returns**: The smoothed image.

### (e) **convolution_d(image_channel, kernel)**
- **Purpose**: This function applies convolution manually using a specific kernel (e.g., unsharp mask or edge detection) to enhance edges or details in the image. It works similarly to the `convolution()` function but focuses on edge detection and sharpening.
- **Returns**: The result of the convolution (filtered) image.

## Core Functionalities:

### (a) Gaussian Filter with Varying Filter Sizes
- **What it does**: 
  - Creates Gaussian filters of different sizes (3x3, 7x7, 11x11).
  - Applies these filters to smooth the image and reduce noise.
  - Computes and prints the PSNR between the original image and the filtered image to measure the difference in image quality.
- **Output**: The filtered images and corresponding PSNR values.

### (b) Gaussian Filter with Varying Sigma Values
- **What it does**:
  - Creates Gaussian filters with different sigma values (1, 10, 30) for a fixed filter size of 3x3.
  - Applies the filters to smooth the image.
  - Computes and prints the PSNR to assess the image quality.
- **Output**: Filtered images with sigma variations and their PSNR values.

### (c) Discussion on PSNR and Image Quality
This section provides an analysis of the effects of filter size and sigma on the PSNR:
- **Filter Size**: Smaller filters like 3x3 preserve more details and provide higher PSNR values, while larger filters lead to more smoothing and lower PSNR.
- **Sigma Variation**: PSNR values remain stable for different sigma values in the 3x3 filter, indicating that the image is not very sensitive to these sigma changes.
- **Trade-offs**: Balancing between noise reduction and detail preservation is crucial when selecting the right filter size and sigma.

### (d) Unsharp Mask and Edge Detection
- **Unsharp Mask**: This part applies a sharpening filter to highlight edges and details in the image, making it appear sharper.
- **Edge Detection**: Detects edges by applying a convolution with an edge detection kernel. It emphasizes the areas of strong intensity changes (edges).
- **Output**: Both the sharpened (unsharp mask) and edge-detected images are saved for comparison.

## Key Observations:
1. **Filter Size**: The 3x3 Gaussian filter provides the best balance between smoothing and retaining image details, giving the highest PSNR values.
2. **Sigma Variation**: Varying sigma for a fixed 3x3 filter does not drastically change the PSNR, indicating less sensitivity to this parameter in the current context.
3. **Trade-offs**: A smaller filter size is better for preserving details, while larger filters are more effective in reducing noise but may lead to information loss.

## Conclusion:
This code demonstrates the application of Gaussian filters with varying sizes and sigma values, unsharp masking, and edge detection to an image. The choice of filter size and sigma depends on the specific goals of the image processing task, whether it be noise reduction, edge enhancement, or detail preservation. 

PSNR values provide a quantitative measure of image quality, helping assess the trade-offs between smoothing and maintaining fine details.
