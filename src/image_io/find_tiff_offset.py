

# Identify cropped region from pair of tiff stacks

import sys
import numpy as np
import numpy.linalg as la
import random
import scipy.ndimage as nd
from scipy.signal import fftconvolve, convolve
import matplotlib.pyplot as plt

from import_tiff import load_tiff

from skimage.feature import match_template


def find_offset_2d(im_orig, im_crop):

    im_orig = np.mean(im_orig, axis=0)
    im_crop = np.mean(im_crop, axis=0)

    crop_offset_i =  0 #im_crop.shape[0]/2
    crop_di = im_crop.shape[0]
    crop_offset_j =  0 #im_crop.shape[1]/2
    crop_dj = im_crop.shape[1]
    small = im_crop[crop_offset_i:crop_offset_i+crop_di,crop_offset_j:crop_offset_j+crop_dj][::-1,::-1]
    if np.std(small) == 0:
        print 'Blank crop region!'
        raise

    print np.std(small)

    u = convolve(im_orig.astype(float)-np.mean(im_orig), small.astype(float)-np.mean(small), mode='same')

    """
    plt.figure()
    plt.imshow(small)

    plt.figure()
    plt.imshow(u)
    plt.show()
    """
    

    mu = np.unravel_index(u.argmax(), u.shape)

    
    mu = np.array(mu) - np.array([crop_offset_i, crop_offset_j])-np.array([crop_di/2, crop_dj/2])

    print 'offset:', mu

    err = im_orig[mu[0]:mu[0]+im_crop.shape[0],
                  mu[1]:mu[1]+im_crop.shape[1]]

    print 'diff:', la.norm(err - im_crop), la.norm(im_crop)


    return mu

def find_offset(im_orig, im_crop):

    crop_offset_i = im_crop.shape[0]/2
    crop_di = 64
    crop_offset_j = im_crop.shape[1]/2
    crop_dj = 64
    crop_offset_k = im_crop.shape[2]/2
    crop_dk = 64
    small = im_crop[crop_offset_i:crop_offset_i+crop_di,crop_offset_j:crop_offset_j+crop_dj, crop_offset_k:crop_offset_k+crop_dk][::-1,::-1,::-1]
    if np.std(small) == 0:
        print 'Blank crop region!'
        raise

    print np.std(small)

    u = fftconvolve(im_orig, small, mode='same')

    mu = np.unravel_index(u.argmax(), u.shape)

    print 'orig_offset:', mu
    
    mu = np.array(mu) - np.array([crop_offset_i, crop_offset_j, crop_offset_k])-np.array([crop_di/2, crop_dj/2, crop_dk/2])


    print 'offset:', mu

    err = im_orig[mu[0]:mu[0]+im_crop.shape[0],
                  mu[1]:mu[1]+im_crop.shape[1],
                  mu[2]:mu[2]+im_crop.shape[2]]
                  

    print 'diff:', la.norm(err - im_crop), la.norm(im_crop)

    return mu

def find_offset_3d(im_orig, im_crop):

#    u = fftconvolve(im_orig-np.mean(im_orig), im_crop[::-1,::-1,::-1]-np.mean(im_orig), mode='full')

    print im_orig.shape, im_crop.shape

    
    u = match_template(im_orig, im_crop)

    print u.shape

    crop_offset_i = 0
    crop_di = im_crop.shape[0]
    crop_offset_j = 0
    crop_dj = im_crop.shape[1]
    crop_offset_k = 0
    crop_dk = im_crop.shape[2]

    mu = np.unravel_index(u.argmax(), u.shape)

    print 'orig mu', mu

 #   mu = np.array(mu) - np.array([crop_offset_i, crop_offset_j, crop_offset_k])-np.array([crop_di-1, crop_dj-1, crop_dk-1])


    return mu


def main():
    im_orig, _ = load_tiff(sys.argv[1])
    im_crop, _ = load_tiff(sys.argv[2])
    find_offset_2d(im_orig, im_crop)
    find_offset(im_orig, im_crop)

if __name__=="__main__":
    main()
