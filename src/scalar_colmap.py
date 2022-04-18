from __future__ import print_function

import numpy as np

from viridis import viridis as rainbow

def clip(x):
    return (x if x>0 else 0.0) if x<1 else 1.0

def gen_array_colmap(a, colmap_scale=None):
    if colmap_scale == None:
        min_a = np.percentile(a, 5)
        max_a = np.percentile(a, 95)
    else:
        min_a, max_a = colmap_scale
    range_a = max(1e-12, max_a - min_a)
        
    col_map = np.zeros((a.shape[0],3), dtype=np.float32)
    for i, v in enumerate(a):
        if i > 0:
            col_map[i,:] = rainbow(clip((v - min_a)/range_a))
    return col_map


def gen_prop_colmap(ca, nl=None, colmap_scale=None):
    if nl == None:
        nl = np.max(list(ca))+1

    a = list(ca[i] for i in ca if i>0)

#    print('areas', a)
    if colmap_scale == None:
        min_a = np.percentile(a, 5)
        max_a = np.percentile(a, 95)
    else:
        min_a, max_a = colmap_scale
    range_a = max(1e-12, max_a - min_a)
        
    col_map = np.zeros((nl,3), dtype=np.float32)
    col_map[0, :] = 0.5
    for i, v in ca.iteritems():
        if i > 0:
            col_map[i,:] = rainbow(clip((v - min_a)/range_a))

    return col_map
