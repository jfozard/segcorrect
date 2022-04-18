

import sys

import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



def load_png(fn):
    i = 0
    frames = []
    im = Image.open(fn)
    im = np.asarray(im)
    if len(im.shape)==3:
        frames.append(np.max(im, axis=2))
    else:
        frames.append(im)
    
    im = np.stack(frames, axis=0)
    del frames

    spacing = np.array((1,1,1))
    
    print(im.shape, spacing)

    return im, spacing


def load_png_rgb(fn):
    i = 0
    frames = []
    im = Image.open(fn)
    im = np.asarray(im)
    spacing = np.array((1,1,1))
    
    return im[np.newaxis,:,:,:], spacing



def load_png_labels(fn, seq=0):
    i = 0
    frames = []
    im = Image.open(fn)
    im = np.asarray(im)
    if len(im.shape)==3:
        frames.append(im[:,:,2] + 256*im[:,:,1]+65536*im[:,:,0])
    else:
        frames.append(im)

    im = np.stack(frames, axis=0)
    del frames


    spacing = np.array((1.0, 1.0, 1.0))
    
    print(im.shape, spacing)

    return im, spacing


