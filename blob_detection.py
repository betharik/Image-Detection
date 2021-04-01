import os

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)


def gaussian_filter(image, sigma):
    H, W = image.shape
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    
    kernel_gaussian = np.zeros((kernel_size,kernel_size))
    s = -kernel_size//2
    e = kernel_size//2+1
    for i in range(s,e):
        for j in range(s,e):
            kernel_gaussian[i][j] = (1/(2*np.pi*sigma**2))*np.exp(-(i**2+j**2)/(2*sigma**2))
    output = scipy.ndimage.convolve(image,kernel_gaussian, mode='reflect', cval=0.0)
    return output


def main():
    image = read_img('dots.png')
    
    if not os.path.exists("./polka_detections"):
        os.makedirs("./polka_detections")

    print("Detecting small dots")
    
    radius = 5
    sigma_1, sigma_2 = radius/np.sqrt(2), 1.11
    gauss_1 = gaussian_filter(image,sigma_1)  
    gauss_2 = gaussian_filter(image,sigma_2)  

    
    DoG_small = gauss_1 - gauss_2  

    # visualize maxima
    maxima = find_maxima(DoG_small, k_xy=10)
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_small.png')

    # Detect Large Circles
    print("Detecting large polka dots")
    rad = 11
    sigma_1, sigma_2 = rad/np.sqrt(2), 5
    gauss_1 = gaussian_filter(image,sigma_1)  
    gauss_2 = gaussian_filter(image,sigma_2) 

    
    DoG_large = gauss_1 - gauss_2  

    # visualize maxima
    
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          'large_dots.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     'large_dots.png')

    print("Detecting cells")

    # Detect the cells in any four (or more) images from vgg_cells
    # Create directory for cell_detections
    if not os.path.exists("./cell"):
        os.makedirs("./cell")

    for i in range(1,7):
        if i == 3: 
            continue
        image = read_img('cells/00' + str(i) + 'cell.png')
        rad = 3
        sigma_1, sigma_2 = rad/np.sqrt(2), 1.2
        gauss_1 = gaussian_filter(image,sigma_1)  # to implement
        gauss_2 = gaussian_filter(image,sigma_2) # to implement

        # calculate difference of gaussians

        DoG_f = gauss_1 - gauss_2 
        thresh = image < 13
        image[thresh] = 0
        image *= -1
        visualize_scale_space(DoG_f, sigma_1, sigma_2 / sigma_1,
                                './cell_detections/00' + str(i) + 'cell_DoG.png')
        maxima = find_maxima(DoG_f, k_xy=15, k_s=2)
        visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                                './cell_detections/00' + str(i) + 'cell.png')
        print(i, "cell maxima", len(maxima))
    # #conduct tresholding on DoG filter

if __name__ == '__main__':
    main()
