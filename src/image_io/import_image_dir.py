

import sys

import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



def load_image_dir(fn, seq=0):
    i = 0
    frames = []
    try:
        while True:
            print('opening', fn%i)
            im = Image.open(fn%i)
            im = np.asarray(im)
            if len(im.shape)==3:
                frames.append(np.max(im, axis=2))
            else:
                frames.append(im)
            i += 1            
    except IOError:
        pass

    im = np.stack(frames, axis=0)
    del frames

    if '/' in fn:
        voxel_fn = fn.rsplit('/', 1)[0] + '/voxelspacing.txt'
    else:
        voxel_fn = 'voxelspacing.txt'

    try:
        with open(voxel_fn,'r') as f:
            sp_x = float(f.readline().split()[1])
            sp_y = float(f.readline().split()[1])
            sp_z = float(f.readline().split()[1])

        spacing = np.array((sp_x, sp_y, sp_z))
    except IOError:
        spacing = np.array((1,1,1))
    
    print(im.shape, spacing)

    return im, spacing



def load_image_dir_labels(fn, seq=0):
    i = 0
    frames = []
    try:
        while True:
            print('opening', fn%i)
            im = Image.open(fn%i)
            im = np.asarray(im)
            if len(im.shape)==3:
                frames.append(im[:,:,2] + 256*im[:,:,1]+65536*im[:,:,0])
            else:
                frames.append(im)
            i += 1            
    except IOError:
        pass

    im = np.stack(frames, axis=0)
    del frames


    spacing = np.array((1.0, 1.0, 1.0))
    
    print(im.shape, spacing)

    return im, spacing


