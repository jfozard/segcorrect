
from log_tool import *
# import labelled stack

from labelled_stack import LabelledStack
#from projection_mesh import ProjectionMesh

#from gen_mesh import make_iso_open_z as make_iso
#from gen_mesh import make_iso_manifold_smooth
#from gen_mesh import make_iso_manifold
#from gen_mesh import make_iso_labels


import numpy as np
import scipy.ndimage as nd
import itertools

from image_io.import_tiff import *
from image_io.import_image_dir import *

from utils_new import *
from utils_itk import *

import scipy.linalg as la

import SimpleITK as sitk

from mini_controller_nogl import *

#from viewer.libintersect import stack_from_mesh


#import traceback

def downsample_mask(ma, k):
    return ma.reshape(ma.shape[0]//k, k, ma.shape[1]//k, k, ma.shape[2]//k, k).sum(axis=(1,3,5))>(k*k*k/2)


def upsample_mask(ma, k):
    return np.kron(ma, np.ones((k,k,k), dtype=ma.dtype))

def downsample_max(ma, k):
    return ma.reshape(ma.shape[0]//k, k, ma.shape[1]//k, k, ma.shape[2]//k, k).max(axis=(1,3,5))


def downsample(ma, k):
    return ma.reshape(ma.shape[0]//k, k, ma.shape[1]//k, k, ma.shape[2]//k, k).sum(axis=(1,3,5))/k/k/k

def upsample(ma, k):
    return nd.zoom(ma, k, order=1)



class Obj(object):
    pass


def apply_clip_planes(clip_planes, stack):
    x, y, z = np.ogrid[0:stack.shape[0], 0:stack.shape[1], 0:stack.shape[2]]
    transform = np.diag([stack.shape[2]-1, stack.shape[1]-1, stack.shape[0]-1, 1])
    norm_transform = (la.inv(transform)[:3,:3]).T
    for p, n in clip_planes:
            p = np.dot(transform, np.hstack((p,[1])))
            n = np.dot(norm_transform, n)
            stack[((x-p[2])*n[2]+(y-p[1])*n[1]+(z-p[0])*n[0])>0] = 0
    return stack



def make_label_obj(self, so, sso):
    o = Obj()        
    o.so = so
    o.sso = sso

    tl = np.array((so.shape[2]*so.spacing[0],
                   so.shape[1]*so.spacing[1],
                   so.shape[0]*so.spacing[2]))
        

    dx = 0.0# 0.5/so.tex_shape[2] 
    dy = 0.0# 0.5/so.tex_shape[1] 
    dz = 0.0# 0.5/so.tex_shape[0] 

    vb = [ [ 0.0, 0.0, 0.0, 0.0+dx, 0.0+dy, 0.0+dz],
           [ tl[0], 0.0, 0.0, 1.0-dx, 0.0+dy, 0.0+dz],
           [ 0.0, tl[1], 0.0, 0.0+dx, 1.0-dy, 0.0+dz],
           [ tl[0], tl[1], 0.0, 1.0-dx, 1.0-dy, 0.0+dz],
           [ 0.0, 0.0, tl[2], 0.0+dx, 0.0+dy, 1.0-dz],
           [ tl[0], 0.0, tl[2], 1.0-dx, 0.0+dy, 1.0-dz],
           [ 0.0, tl[1], tl[2], 0.0+dx, 1.0-dy, 1.0-dz],
           [ tl[0], tl[1], tl[2], 1.0-dx, 1.0-dy, 1.0-dz] ]


    vb = np.array(vb, dtype=np.float32)
    vb = vb.flatten()
        
    idx_out = np.array([[0, 2, 1], [2, 3, 1],
                        [1, 4, 0], [1, 5, 4],
                        [3, 5, 1], [3, 7, 5],
                        [2, 7, 3], [2, 6, 7],
                        [0, 6, 2], [0, 4, 6],
                        [5, 6, 4], [5, 7, 6]]
                       , dtype=np.uint32)        

    
    sc = 1.0/la.norm(tl)
    sc = 0.5*tl

    o.transform = np.array(( (sc, 0.0, 0.0, -sc*c[0]), (0.0, sc, 0.0, -sc*c[1]),  (0.0, 0.0, sc, -sc*c[2]), (0.0, 0.0, 0.0, 1.0)))

    o.tex_transform = np.array( (((1.0-2*dx)/tl[0], 0.0, 0.0, dx), 
                                 ( 0.0, (1.0-2*dy)/tl[1], 0.0, dy),
                                 ( 0.0, 0.0, (1.0-2*dz)/tl[2], dz),
                                 ( 0.0, 0.0, 0.0, 1.0) ))

    o.orig_vb = np.array(vb)
    o.orig_idx = idx_out
    
    return o

def make_stack_label_obj(stack, spacing):
    o = Obj()
    o.shape = stack.shape
    o.spacing = spacing
    return o

def make_stack_obj(stacks, spacing):
    o = Obj()
    o.shape = stacks[0].shape
    o.spacing = spacing
    return o


def clip_stack_obj(clip_planes, stack_obj):
    verts = stack_obj.orig_vb[:,:3]
    tris = stack_obj.orig_idx
    transform = stack_obj.transform
    inv_transform = la.inv(transform)
    norm_transform = transform[:3,:3].T
    for p, n in clip_planes:
        # Transform back
        p = np.dot(inv_transform, np.hstack((p,[1])))[:3]
        n = np.dot(norm_transform, n)
        verts, tris = slice_cell(p, n, verts, tris)
    tex_transform = stack_obj.tex_transform
    verts = np.array(verts)
    tex_coords = np.dot(tex_transform, np.vstack((verts.T, np.ones((1,verts.shape[0])))))[:3,:]
    
    vb = np.concatenate((verts,tex_coords.T),axis=1).astype(np.float32)
    idx_out = np.array(tris, dtype=np.uint32)


    stack_obj.vtVBO.bind()
    stack_obj.vtVBO.set_array(vb)
    stack_obj.vtVBO.copy_data()
    stack_obj.vtVBO.unbind()

    stack_obj.elVBO.set_array(idx_out)
    stack_obj.elCount = len(idx_out.flatten())


        









        



