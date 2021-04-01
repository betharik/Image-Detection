import os

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    
    H, W = image.shape
    image_c = image.copy()
    image = np.roll(image, (u,v))
    squared_diff = (image_c - image)
    k = np.ones(window_size)
    output = scipy.ndimage.convolve(squared_diff,k, mode='constant', cval=0.0)
    return output


def harris_detector(image, window_size=(5, 5)):
    
    H, W = image.shape
    kernel_gaussian = np.zeros(window_size)
    s = -window_size[0]//2
    e = window_size[1]//2+1
    for i in range(s,e):
        for j in range(s,e):
            sd = 0.572 
            kernel_gaussian[i][j] = (1/(2*np.pi*sd**2))*np.exp(-(i**2+j**2)/(2*sd**2))
    k = np.ones(window_size)
    kx = np.array([[-1,0,1]])
    ky = np.array([[-1],[0],[1]])
    # compute the derivatives
    Ix = scipy.ndimage.convolve(image, kx, mode='constant', cval=0.0)
    Ix = scipy.ndimage.convolve(Ix, k, mode='constant', cval=0.0)
    Iy = scipy.ndimage.convolve(image, ky, mode='constant', cval=0.0)
    Iy = scipy.ndimage.convolve(Iy, k, mode='constant', cval=0.0)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
    # For each image location, construct the structure tensor and calculate
    # the Harris response
    M = np.zeros((H,W,3))
    for i in range(H-1):
        for j in range(W-1):
            u = i - window_size[0]//2 - 1
            v = j - window_size[1]//2 - 1
            for x in range(u,u+window_size[0]):
                for y in range(v,v+window_size[1]):
                    if not (i >= H or j >= W):
                        M[i,j,0] += Ixx[x,y]
                        M[i,j,1] += Ixy[x,y]
                        M[i,j,2] += Iyy[x,y]
    M[:,:,0] = scipy.ndimage.convolve(Ixx, k, mode='constant', cval=0.0)
    M[:,:,1] = scipy.ndimage.convolve(Ixy, k, mode='constant', cval=0.0)
    M[:,:,2] = scipy.ndimage.convolve(Iyy, k, mode='constant', cval=0.0)
    
    response = np.multiply(M[:,:,0],M[:,:,2]) - np.multiply(M[:,:,1], M[:,:,1]) - 0.05*((M[:,:,0] + M[:,:,2])**2)
    return scipy.ndimage.convolve(response, kernel_gaussian, mode='constant', cval=0.0)


def main():
    img = read_img('hopper.png')

    # Corner Score

    # Define offsets and window size and calculcate corner score
    u, v, W = 0, 5, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "corner_score.png")
    u, v = 0, -5
    score = corner_score(img, u, v, W)
    save_img(score, "corner_score1.png")
    u, v = 5, 0
    score = corner_score(img, u, v, W)
    save_img(score, "corner_score2.png")
    u, v = -5, 0
    score = corner_score(img, u, v, W)
    save_img(score, "corner_score3.png")

    # Harris Corner Detector 
    harris_corners = harris_detector(img)
    plt.imshow(harris_corners, cmap='hot')
    plt.colorbar()
    plt.show()
    save_img(harris_corners, "harris_response.png")


if __name__ == "__main__":
    main()
