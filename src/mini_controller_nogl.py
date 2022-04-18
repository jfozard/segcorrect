from __future__ import print_function
from log_tool import *
# import labelled stack

#from ev_split import ev_split

from labelled_stack import LabelledStack
#from projection_mesh import ProjectionMesh

#from gen_mesh import make_iso_open_z as make_iso
#from gen_mesh import make_iso_manifold_smooth
#from gen_mesh import make_iso_manifold
#from gen_mesh import make_iso_labels


import numpy as np
import scipy.ndimage as nd
import itertools

from skimage.segmentation import random_walker
from skimage.draw import line
import skimage.exposure
import skimage.measure

from image_io.import_tiff import *
from image_io.import_image_dir import *
from image_io.import_png import *

from utils_new import *
from utils_itk import *

import scipy.linalg as la

import SimpleITK as sitk

import eval_seg
#import objgraph

#from viewer.libintersect import stack_from_mesh

#from track_allocations import AllocationTracker

#import psutil
import os

import csv
import blosc
import copy


import collections

def undo_no_log(f):
    @wraps(f)
    def func_wrapper(*args, **kwargs):
        self = args[0]
        self.log.write('["{}",{},{}]'.format(f.__name__,args[1:],kwargs)+'\n')
        self.log.flush()
        v = f(*args, **kwargs)
        def u():
            f(*args, **kwargs)
        def undo():
            pass
        return v
    return func_wrapper


def undo_signal(f):
    @undo_log
    @wraps(f)
    def func_wrapper(*args, **kwargs):
        self=args[0]
        # also need to preserve cell data ....

        old_signal = self.get_signal_controller().get_stack()
        def undo():
            self.get_label_controller().set_stack(old_signal)
        v = f(*args, **kwargs)
        return undo, v

    return func_wrapper

"""
        old_selected = self.get_label_controller().get_selected()
        def undo():
            self.get_label_controller().set_selected(old_selected)
"""

def undo_selected(f):
    @undo_log
    @wraps(f)
    def func_wrapper(*args, **kwargs):
        self=args[0]

        old_selected = self.get_label_controller().get_selected()
        def undo():
            self.get_label_controller().set_selected(old_selected)

        v = f(*args, **kwargs)
        return undo, v

    return func_wrapper


def undo_label(f):
    @undo_log
    @wraps(f)
    def func_wrapper(*args, **kwargs):
        self=args[0]
        # also need to preserve cell data ....

        old_state = self.get_label_controller().get_state()
        def undo():
            self.get_label_controller().set_state(old_state)

        v = f(*args, **kwargs)
        return undo, v

    return func_wrapper



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
            print(stack.shape, x.shape, y.shape, z.shape)
            for i in range(0, stack.shape[0], 16):
                r = slice(i, min(i+16, stack.shape[0]))
                stack[r, :, :][((x[r,:,:]-p[2])*n[2]+(y-p[1])*n[1]+(z-p[0])*n[0])>0] = 0
    return stack



def make_label_obj(so, sso):
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
    c = 0.5*tl


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



class StackController(object):
    def __init__(self, stack, spacing):
        self.stack = stack
        self.spacing = spacing
        self.update_callbacks = []

        
    def apply_clip_planes(self, clip_planes):
        self.set_stack(apply_clip_planes(clip_planes, self.stack))

    def set_stack(self, s):
        self.stack = s
        
    def update(self, msg=None):
        for f in self.update_callbacks:
            f(msg)



class LabelledStackController(StackController):
    def __init__(self, stack, spacing, img_data=None, orig_shape=None, autosave_dir=None):
        self.labelled_stack = LabelledStack(stack, spacing, img_data)
        self.spacing = spacing
        self.selected = []
        self.update_callbacks = []
        self.omitted = []
        self.stack_updated = False
        if orig_shape is not None:
            self.orig_shape = orig_shape
        else:
            self.orig_shape = stack.shape
        self.autosave_dir = autosave_dir
        self.autosave_idx = 0



    def get_state(self):
        return blosc.pack_array(self.labelled_stack.labels, cname='lz4'), copy.deepcopy((self.labelled_stack.celltypes, self.labelled_stack.cell_props, self.selected, self.omitted, self.spacing, self.orig_shape))

    def set_state(self, state):
        (self.labelled_stack.labels, self.labelled_stack.celltypes, self.labelled_stack.cell_props, self.selected, self.omitted, self.spacing, self.orig_shape) = (blosc.unpack_array(state[0]),) + state[1]
        self.update_cells()
        self.update()

        
    
    def update(self, msg=None):
        StackController.update(self, msg)
        print('as', self.autosave_dir)
        if self.autosave_dir and msg!='selected':
            self.write_celltypes(self.autosave_dir + '/'+str(self.autosave_idx)+'.csv')
            self.write_tiff(self.autosave_dir + '/'+str(self.autosave_idx)+'.tif')
            self.autosave_idx = (self.autosave_idx+1)%10
            
    def update_cells(self):
        self.labelled_stack.update_cells()
        # save_cell_props

    def add_seed(self, v, r=1, use_selected=False):
        print(v)
        s = self.stack
        shape = s.shape
        if not use_selected or not self.selected:
            new_idx = np.max(self.stack)+ 1
        else:
            new_idx = self.selected[0]
        print('new_idx', new_idx)
        s[min(shape[0]-1,max(0,v[0]-r)):min(shape[0],v[0]+r+1),
          min(shape[1]-1,max(0,v[1]-r)):min(shape[1],v[1]+r+1),
          min(shape[2]-1,max(0,v[2]-r)):min(shape[2],v[2]+r+1)] = new_idx
#        print (min(shape[0],max(0,v[0]-r)), min(shape[0],v[0]+r+1))
        self.labelled_stack.celltypes[new_idx] = 0
        for p in self.labelled_stack.cell_props.values():
            p[new_idx] = 0
        self.update_cells()
        self.update()

    def classify_seg(self, other_labels):
        celltypes = self.labelled_stack.celltypes
        A = self.stack
        matching_cells, best_IoU = eval_seg.matching_IoU(A, other_labels, threshold=0.75, return_best=True)
        under_seg, acme = eval_seg.calc_acme_criterion(A, other_labels, threshold=0.5, return_criterion=True)
        under_seg = set(under_seg) - set(matching_cells)

        l = np.unique(A)
        print(best_IoU)
        for i in l:
            if i in matching_cells:
                celltypes[i] = 0
            elif i in under_seg:
                celltypes[i] = 1
            else:
                celltypes[i] = 2
        print(celltypes)
        self.update()

    def split_cc(self):
        new = skimage.measure.label(self.stack)
        self.labelled_stack.set_stack(new)
        self.labelled_stack.celltypes = {} #dict((i, 0) for i in np.unique(new))
        self.update_cells()
        self.update()
        
    def update_signal(self, stack):
        self.labelled_stack.update_img_data(stack)
        
    def write_tiff(self, fn):
        s = self.orig_shape
        write_tiff(fn, self.stack[:s[0], :s[1], :s[2]].astype(np.uint16), self.spacing)

    def set_omitted(self, omitted):
        self.omitted = omitted
        
    def update_selected(self):
        self.update('selected')

    def get_selected(self):
        return list(self.selected)

    def get_label_point(self, p):
        if self.stack.shape[0]>1:
            return self.stack[tuple(p)]
        else:
            return self.stack[(0,)+tuple(p[1:])]
    
    def gen_colmap(self, prop_name=None, celltypes=False, omitted=[], ct_weight=0.6, grey_labels=False):
        return self.labelled_stack.gen_colmap(prop_name, celltypes, self.selected, self.omitted, ct_weight, grey_labels)
        
    @property
    def stack(self):
        return self.labelled_stack.labels

    def set_stack(self, s):
        self.labelled_stack.set_stack(s)
        self.update_cells()
        self.update()

    def dilate_labels(self):
        s = self.stack
        s = nd.grey_dilation(s, size=(3,3,3))
        self.set_stack(s)
        self.update_cells()
        
        
    def get_cell_props(self):
        return self.labelled_stack.cell_props
    
    def calc_mean_signal(self, signal):
        self.labelled_stack.calc_mean_signal(signal)

    def calc_min_signal(self, signal):
        self.labelled_stack.calc_min_interior_signal(signal)

    def calc_mean_interior_signal(self, signal):
        self.labelled_stack.calc_mean_interior_signal(signal)

    def make_borders(self):
        return self.labelled_stack.make_borders()
        
    #TODO
    
    def select_by_prop(self, cond):
        props = self.get_cell_props()
        percentile_props = {}
        prop_names = list(props)
        for p in props:
            prop = props[p]
            mean_v = np.mean(props[p].values())
            percentile_props[p] = dict((i, prop[i]/ mean_v) for i in prop)

        selected = []
        for i in self.labelled_stack.cell_props[prop_names[0]]:
            v = eval(cond, {}, dict((p, percentile_props[p][i]) for p in prop_names))
            if v:
                selected.append(i)
        self._set_selected(selected)


    def write_celltypes(self, fn):
        celltypes = self.labelled_stack.celltypes
        with open(fn, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for (i,v) in celltypes.items():
                writer.writerow((i,v))

    def get_celltypes(self):
        return self.labelled_stack.celltypes
                
    def read_celltypes(self, fn):
        celltypes = self.labelled_stack.celltypes
        with open(fn, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row)>1:
                    celltypes[int(row[0])] = int(row[1])
        self.update()

                
    def set_celltype(self, ct):
        for i in self.selected:
            self.labelled_stack.celltypes[i] = ct
        self.update()

    
    def select_small(self, large):
        labels, counts = np.unique(self.stack, return_counts=True)
        mean_area = np.mean(counts)
        self._set_selected(labels[counts<large*mean_area].tolist())

    
    def select_large(self, small):
        labels, counts = np.unique(self.stack, return_counts=True)
        mean_area = np.mean(counts)
        self._set_selected(labels[counts>small*mean_area].tolist())

    
    def select_neighbours(self):
        ls = self.labelled_stack
        A = ls.get_mean_bdd_connectivity()

        selected = list(set(itertools.chain.from_iterable(A.indices[A.indptr[i]:A.indptr[i+1]] for i in self.selected)) - set(self.selected))
        if 0 in selected:
            selected.remove(0)

        self._set_selected(selected)

    
    def write_cell_graph(self, fn):
        self.labelled_stack.write_cell_graph(fn)

    
    def merge_watershed(self, level):
        return self.labelled_stack.merge_watershed(level)

    
    def expand_selection(self, threshold):
        ls = self.labelled_stack
        A = ls.get_mean_bdd_connectivity()
        threshold = float(threshold)*np.mean(A.data)

        selected = list(set(itertools.chain(itertools.chain.from_iterable(A.indices[A.indptr[i]:A.indptr[i+1]][A.data[A.indptr[i]:A.indptr[i+1]]<threshold] for i in self.selected), self.selected)))
        if 0 in selected:
            selected.remove(0)
        self._set_selected(selected)

    
    def select(self, v):
        self._set_selected([v])        
    
    def get_cell_point(self, p):
        return self.stack[p]

    
    def add_selected(self, v):
        if v not in self.selected:
            self.selected.append(v)
        else:
            self.selected.remove(v)
        self.update_selected()

    def include_selected(self, v):
        if v not in self.selected:
            self.selected.append(v)
            self.update_selected()
    
    def delete_selected(self):
        self._delete_cells(self.selected)
        self.update()
        
    
    def remove_small_large_cells(self, small_large):

        labels, counts = np.unique(self.stack, return_counts=True)
        mean_area = np.mean(counts)
        
        print('mean_area', mean_area)
        
        bad_labels = labels[np.logical_or(counts<small*mean_area, counts>large*mean_area)]

        print('bad_labels', bad_labels)
        self._delete_cells(bad_labels)
        self.update()
        
    def merge_selected(self):
        selected = sorted(self.selected)
        if not selected:
            return
        min_selected = selected[0]
        self.stack.flat[np.in1d(self.stack.flat, self.selected)] = min_selected
        
        if 'area' in self.labelled_stack.cell_props:
            new_area = sum(self.labelled_stack.cell_props['area'][i] for i in selected)
            self.labelled_stack.cell_props['area'][selected[0]] = new_area
        self._delete_cells(selected[1:])
        self.selected = [selected[0]]
        print('up')
        self.update()
        

    def _delete_cells(self, cells):
        # Fix up selection
        for i in cells:
            if i in self.labelled_stack.celltypes:
                del self.labelled_stack.celltypes[i]
            for p in self.labelled_stack.cell_props.values():
                if i in p:
                    del p[i]

        self._set_selected(list(set(self.selected) - set(cells)))

    def select_by_celltype(self, ct):
        selected = [i for i,v in self.labelled_stack.celltypes.items() if v in ct] 
        self._set_selected(selected)
    
    def set_omitted(self, ct_list):
        self.omitted = ct_list
        self.update()

    
    def set_selected(self, selected):
        self._set_selected(selected)

    def _set_selected(self, selected):
        self.selected = selected
        self.update_selected()


    
    def delete_selected(self):
        self.stack.flat[np.in1d(self.stack.flat, self.selected)] = 0
        self._delete_cells(self.selected)
        self.update()
        
class PointMarkerCollectionController(object):
    def __init__(self):
        self.points = []
        
    def add_points(self, points):
        self.points += points

    def add_point_label(self, x):
        self.points.append(x)
        

    
        
class SignalStackController(StackController):

    def write_tiff(self, fn):
        write_tiff(fn, self.stack.astype(np.float32), self.spacing)

    def blur_stack(self, radius):
        self.stack = itk_blur_stack(self.stack, self.spacing, radius)
        self.update()
        
    def apply_power(self, power):
        self.stack = np.power(self.stack, power)
        self.update()

    def flip_z(self):
        self.stack = np.ascontiguousarray(self.stack[::-1,:,:])
        self.update()

    def equalize_stack(self):
        self.stack = equalize(self.stack)
        self.update()

    def invert(self):
        self.stack = np.max(self.stack) - self.stack
        self.update()

    def grey_closing(self, radius):
        self.stack = nd.grey_closing(self.stack, radius)

    def paint(self, p):
#        self.stack[tuple(p)] = 1.0
        self.last_paint = tuple(p)[1:]
        self.update()

    def clahe(self):
        self.stack[0,:,:] = skimage.exposure.equalize_adapthist(self.stack[0,:,:], 16)

    def aniso(self):
        signal = sitk.GetImageFromArray(self.stack)
        signal = sitk.GradientAnisotropicDiffusion(signal)
        self.stack = sitk.GetArrayFromImage(signal)
        self.update()

    def subtract_bg(self):
        self.stack = np.clip(self.stack - nd.gaussian_filter(self.stack, 20), 0, 1)
        self.update()
        

    def paint_to(self, p):
        rr, cc = line(self.last_paint[0], self.last_paint[1], p[1], p[2])
        self.stack[p[0],rr,cc] = 1.0
        self.last_paint = tuple(p)[1:]
        self.update()

    
class WorldController(object):
    def __init__(self, log=None, spacing=None, autosave_dir=None):
        self.all_label_controllers = {}
        self.all_signal_controllers = {}
        self.active_signal = None
        self.active_label = None
        self.autosave_dir = autosave_dir
        self.update_callbacks = []
        self.spacing = spacing
        self.log = log
        self.undo_stack = collections.deque([], maxlen=100)
        self.log_stack = []
        self.stack_shape = None
        self.view = None


    def replay_log(self, filename):
        with open(filename, 'r') as f:
            for l in f:
                print('REPLAY COMMAND :', l)
                func_name, args, kwargs = eval(l, {'array':np.array, 'float32':np.float32})
                try:
                    y = getattr(self, func_name)
                    
                    w = self.wc
                    y = getattr(w, func_name)
                except AttributeError:
                    pass

                y(*args, **kwargs)
        self.update()

        
    def undo(self):
        self.log.write('["undo",(),{}]\n')
        print(self.undo_stack)
        if self.undo_stack and self.undo_stack[-1] != False:
            action = self.undo_stack.pop()
            action()

    def repeat(self):
        if self.log_stack:
            action = self.log_stack[-1]
            print('REPEAT', action)
            action()

        
    def set_stack_shape(self, shape):
        self.stack_shape = shape
        if self.view:
            self.view.make_stack_obj()
        
        
    def update(self, msg=None):
        for f in self.update_callbacks:
            f(msg)

    def update_signal(self, *msg):
        l = self.active_label
        s = self.active_signal
        if l is not None:
            self.get_label_controller().update_signal(self.get_signal_stack)


    def update_label(self, *msg):
        pass
            
    def get_spacing(self):
        if self.spacing is not None:
            return self.spacing
        else:
            return (1.0, 1.0, 1.0)

        
    def get_label_controller(self):
        if self.active_label is not None:
            return self.all_label_controllers[self.active_label]
        else:
            return None
        
    def get_label_stack(self):
        if self.active_label is not None:
            return self.all_label_controllers[self.active_label].stack
        else:
            return np.zeros(self.stack_shape, dtype=np.int32)
        

        
    def get_signal_controller(self):
        return self.all_signal_controllers[self.active_signal]

    def get_signal_stack(self):
        if self.active_signal is not None:
            return self.all_signal_controllers[self.active_signal].stack
        else:
            return np.zeros(self.stack_shape, dtype=np.float32)

    def get_all_signal_stacks(self):
        return [c.stack for c in self.all_signal_controllers.values()]

        

    @undo_label
    def add_seed(self, v, use_selected=False):
        if self.active_label is None:
            self.make_empty_labels()
        print(self.active_label)
        self.get_label_controller().add_seed(v, use_selected=use_selected)
            


    @undo_label
    def seed_minima(self, r=2):
        if self.active_label is None:
            self.make_empty_labels()
        signal = self.get_signal_stack()
        labels = self.get_label_stack()
        b_signal = signal #.astype(np.float32)
        b_signal = nd.gaussian_filter(b_signal, r)

        nbd = nd.generate_binary_structure(len(b_signal.shape),2)
        minima = (b_signal==nd.minimum_filter(b_signal, footprint=nbd))
        
        minima = nd.binary_dilation(minima, structure=np.ones((3,3,3)))
        m_labels, nl = nd.label(minima)
        print('labels shape', m_labels.shape, nl)
        """
        #labels[m_labels>0] = m_labels[m_labels>0]+np.max(labels)
        labels = m_labels
        minima = signal>np.mean(signal)
        """
        self.get_label_controller().set_stack(m_labels.astype(np.int32))        

        
    def watershed_from_labels(self, slice_z=None, mask_celltypes=True):
        celltypes = self.get_label_controller().labelled_stack.celltypes
        

        labels = self.get_label_stack()
        print('labels dtype', labels.dtype)
        N = np.max(labels)+1
        signal = self.get_signal_stack()
        if signal.shape[0]>labels.shape[0]:
            signal = signal[slice_z:slice_z+1,:,:]

        if mask_celltypes: 
            mask_cells = np.array([celltypes.get(i, 0)>0 for i in range(N)])
            mask = mask_cells[labels]
            old_labels = np.array(labels)
            labels[mask] = 0
            signal = np.array(signal)
            signal[mask] = 1
        signal = sitk.GetImageFromArray(signal)
        labels = sitk.GetImageFromArray(labels.astype(np.int32))
        labels = sitk.MorphologicalWatershedFromMarkers(signal, labels, markWatershedLine = False)
        labels = sitk.GetArrayFromImage(labels)
        if mask_celltypes:
            labels = np.where(mask, old_labels, labels)
        self.get_label_controller().set_stack(labels.astype(np.int32))


    @undo_label
    def resegment(self):

        lc = self.get_label_controller()
        labels = np.array(self.get_label_stack())
        selected = lc.get_selected()
        
        N = np.max(labels)+1
        signal = np.array(self.get_signal_stack())
        s_max = np.max(signal)

        mask = np.isin(labels, selected)
        mask = nd.binary_fill_holes(mask)

        obj = nd.find_objects(mask)
        if not obj:
            return
        sl = obj[0]

        labels = labels[sl]
        signal = signal[sl]
        mask = mask[sl]
        
        labels[~mask] = 0
        signal[~mask] = s_max
        for i in selected:
            labels[labels==i] = 0
        
        signal = sitk.GetImageFromArray(signal)
        labels = sitk.GetImageFromArray(labels.astype(np.int32))
        labels = sitk.MorphologicalWatershedFromMarkers(signal, labels, markWatershedLine = False)
        new_labels = sitk.GetArrayFromImage(labels)

        labels = self.get_label_stack()
        labels[sl] = np.where(mask, new_labels, labels[sl])
        self.get_label_controller().set_stack(labels.astype(np.int32))


    @undo_label
    def split_plane(self):

        lc = self.get_label_controller()
        labels = np.array(self.get_label_stack())
        selected = lc.get_selected()
        if len(selected)!=1:
            return
        
        N = np.max(labels)+1
        signal = np.array(self.get_signal_stack())
        s_max = np.max(signal)

        mask = np.isin(labels, selected)
        mask = nd.binary_fill_holes(mask)

        obj = nd.find_objects(mask)
        if not obj:
            return
        sl = obj[0]
        
        labels = labels[sl]
        mask = mask[sl]
        
        labels[~mask] = 0

        labels[labels==selected[0]] = 0

        cells = np.unique(labels)

        cm = nd.center_of_mass(np.ones_like(labels), labels, cells[1:])
        if len(cm)<3:
            return

        c_i, c_j, c_k = cm[0]

        n_i, n_j, n_k = np.cross(np.array(cm[1])-np.array(cm[0]), np.array(cm[2])-np.array(cm[0]))
        
        s = labels.shape
        i, j, k = np.ogrid[:s[0], :s[1], :s[2]]
        mask2 = ((i-c_i)*n_i + (j-c_j)*n_j + (k-c_k)*n_k >0)*mask

        new_labels = mask*selected[0]
        new_labels[mask2] = cells[1]

        
        labels = self.get_label_stack()
        labels[sl] = np.where(mask, new_labels, labels[sl])
        self.get_label_controller().set_stack(labels.astype(np.int32))


        

    @undo_label
    def rw_from_labels(self, beta=100):
        signal = self.get_signal_stack()
        labels = self.get_label_stack()
        result = random_walker(signal, labels, beta=beta, mode='cg_mg', use_gradient=False)
        self.get_label_controller().set_stack(result)

    @log
    def watershed_no_labels(self, sigma=1.0, h=1.0):
        if self.active_label is None:
            self.make_empty_labels()
        s = self.get_signal_stack()
        signal = sitk.GetImageFromArray(255*(s/float(np.max(s))))
        if sigma>0:
            signal = sitk.DiscreteGaussian(signal, sigma)
        signal = sitk.Cast(signal, sitk.sitkInt16) 
        labels = sitk.MorphologicalWatershed(signal, level=h, markWatershedLine = False, fullyConnected=False)
        labels = sitk.GetArrayFromImage(labels)
        lc = self.get_label_controller()
        lc.set_stack(labels.astype(np.int32))
        lc.update_cells()

    def get_signal_names(self):
        return self.all_signal_controllers.keys()

    def get_label_names(self):
        return self.all_label_controllers.keys()

    @undo_signal
    def apply_clip_planes_signal(self, clip_planes):
        self.get_signal_controller().apply_clip_planes(clip_planes)

    @undo_signal
    def apply_clip_planes_labels(self, clip_planes):
        self.get_label_controller().apply_clip_planes(clip_planes)


    @undo_label
    def classify_seg(self, other_name):
        self.get_label_controller().classify_seg(self.all_label_controllers[other_name].stack)
        
    @log
    def copy_signal_stack(self):
        stack_name = 'img'+str(len(self.all_signal_controllers)+1)

        u = np.array(self.get_signal_stack())
        
        self._add_signal(stack_name, u)
        self._select_active_signal(stack_name)

        self.update()


    @log
    def copy_label_stack(self):
        stack_name = 'label'+str(len(self.all_label_controllers)+1)

        u = np.array(self.get_label_stack())
        
        self._add_label(stack_name, u)
        self._select_active_label(stack_name)

        self.update()


    
    @log
    def select_active_signal(self, stack_name):
        self._select_active_signal(stack_name)
        self.update()
        
    def _select_active_signal(self, stack_name):
        self.active_signal = stack_name


    @log
    def select_active_label(self, label_name):
        self._select_active_label(label_name)
        self.update()
        
    def _select_active_label(self, label_name):
        self.active_label = label_name

        
    def _add_signal(self, new_name, new_signal):
        self.all_signal_controllers[new_name] = SignalStackController(new_signal, self.spacing)
        self.all_signal_controllers[new_name].update_callbacks.append(self.update_signal)

    def _add_label(self, new_name, new_label, orig_shape=None):
        if self.active_signal is None:
            self.all_label_controllers[new_name] = LabelledStackController(new_label, self.spacing, None, orig_shape=orig_shape, autosave_dir = self.autosave_dir)
        else:
            self.all_label_controllers[new_name] = LabelledStackController(new_label, self.spacing, self.get_signal_stack(), orig_shape=orig_shape, autosave_dir = self.autosave_dir)
        self.all_label_controllers[new_name].update_callbacks.append(self.update_label)
        self.all_label_controllers[new_name].update_callbacks.append(self.update)



    @log
    def load_signal(self, filename, img_dir=False):
        stack_name = 'img'+str(len(self.all_signal_controllers)+1)
        #
        def round_up(i, c):
            return i + (c-i%c)%c

        def process(s):
            s = np.abs(s)
            ext_shape = tuple( [ round_up(i, 16) for i in s.shape ])
            tmp = np.zeros(ext_shape, dtype = s.dtype)
            tmp[:s.shape[0], :s.shape[1], :s.shape[2]] = s
            return tmp

        if img_dir:
            u, spacing = load_image_dir(filename, 0)
            spacing = np.array(spacing)
        elif 'png' in filename:
            u, spacing = load_png(filename)
            spacing = np.array(spacing)
        else:
            u, spacing = load_tiff(filename, 0)
            spacing = np.array(spacing)

        if self.spacing is None:
            self.spacing = spacing
            
#        u = process(u).astype(np.float32)
        u = u.astype(np.float32)

        u = (u/np.max(u)).astype(np.float32)

        if self.stack_shape is None:
            self.set_stack_shape(u.shape)

        
        self._add_signal(stack_name, u)
        self._select_active_signal(stack_name)

        self.update()


    @log
    def load_signal_rgb(self, filename, img_dir=False):
        stack_name = 'img'+str(len(self.all_signal_controllers)+1)
        #
        def round_up(i, c):
            return i + (c-i%c)%c

        def process(s):
            s = np.abs(s)
            ext_shape = tuple( [ round_up(i, 16) for i in s.shape ])
            tmp = np.zeros(ext_shape, dtype = s.dtype)
            tmp[:s.shape[0], :s.shape[1], :s.shape[2]] = s
            return tmp

        if img_dir:
            u, spacing = load_image_dir(filename, 0)
            spacing = np.array(spacing)
        elif 'png' in filename:
            u, spacing = load_png_rgb(filename)
            spacing = np.array(spacing)
        else:
            u, spacing = load_tiff(filename, 0)
            spacing = np.array(spacing)

        if self.spacing is None:
            self.spacing = spacing
            

#        u = process(u).astype(np.float32)

        for i in range(u.shape[0]):
            v = u[i,:,:,:].astype(np.float32)

            v = (v/np.max(v)).astype(np.float32)

            if self.stack_shape is None:
                self.set_stack_shape(v.shape)
            self._add_signal(stack_name+'-'+['r','g','b'][i], v)
        
        self._select_active_signal(stack_name+'-r')

        self.update()
        

    @log
    def make_empty_labels(self, label_name=None):
        if label_name is None:
            label_name = 'label'+str(len(self.all_label_controllers)+1)
        #

        u = self.get_label_stack()

        if self.stack_shape is None:
            self.set_stack_shape(u.shape)

        self._add_label(label_name, u, orig_shape = u.shape)
        self._select_active_label(label_name)
        self.update()


        
    @log
    def load_label(self, filename, img_dir=False, label_name=None, remap=False):
        if label_name is None:
            label_name = 'label'+str(len(self.all_label_controllers)+1)
        #

        def round_up(i, c):
            return i + (c-i%c)%c

        def process(s):
            s = np.abs(s)
            ext_shape = tuple( [ round_up(i, 16) for i in s.shape ])
            tmp = np.zeros(ext_shape, dtype = s.dtype)
            tmp[:s.shape[0], :s.shape[1], :s.shape[2]] = s
            return tmp

        if img_dir or '%' in filename:
            u, spacing = load_image_dir_labels(filename, 0)
            remap=True
        elif 'png' in filename:
            u, spacing = load_png_labels(filename)
            remap = True
        else:
            u, spacing = load_tiff(filename, 0)



        if len(u.shape)==4:
            u = 256*256*u[:,:,:,0]+256*u[:,:,:,1] + u[:,:,:,2] # RGB labelled images

            
        orig_shape = u.shape

        if u.dtype==np.int32:
            u=u.astype(np.uint32)
        
        # REMAP

        shape = u.shape
        if remap:
            l, u = np.unique(u, return_inverse=True)
            u = u.reshape(shape)

        spacing = np.array(spacing)
        if self.spacing is None:
            self.spacing = spacing
            
#        u = process(u)
                    
        if self.stack_shape is None:
            self.set_stack_shape(u.shape)

        self._add_label(label_name, u, orig_shape = orig_shape)
        self._select_active_label(label_name)


        self.update()

        

        
    @undo_label
    def delete_selected(self):
        self.get_label_controller().delete_selected()


    @undo_no_log
    def write_label_tiff(self, fn):
        self.get_label_controller().write_tiff(fn)

    @undo_no_log
    def write_celltypes(self, fn):
        self.get_label_controller().write_celltypes(fn)


    @undo_label
    def read_celltypes(self, fn):
        self.get_label_controller().read_celltypes(fn)
        self.update()
        
    @undo_no_log
    def write_signal_tiff(self, fn):
        self.get_signal_controller().write_tiff(fn)

    @undo_no_log
    def write_mask_tiff(self, fn):
        self.get_mask_controller().write_tiff(fn)



    def get_selected(self):
        c = self.get_label_controller()
        if c:
            return c.get_selected()
        else:
            return []


    def get_celltypes(self):
        c = self.get_label_controller()
        if c:
            return c.get_celltypes()
        else:
            return {}
        
    def gen_colmap(self, prop_name=None, celltypes=False, omitted=[], ct_weight=0.6, grey_labels=False):
        lc = self.get_label_controller()
        if lc:
            return lc.gen_colmap(prop_name, celltypes, omitted, ct_weight, grey_labels)
        else:
            return np.zeros((256,3), dtype=np.float32)

        

    def gen_volume_colmap(self):
        lc = self.get_label_controller()
#        print 'gen_volume_colmap', lc
        if lc:
#            print '>', lc.gen_colmap(celltypes=True, ct_weight=0.0, grey_labels=False)
#            print '>>'
            return lc.gen_colmap(celltypes=True, ct_weight=0.0, grey_labels=False)
        else:
            return np.zeros((256,3), dtype=np.float32)

        
    def get_cell_props(self):
        return self.get_label_controller().get_cell_props()

        
    @undo_selected
    def select_by_prop(self, cond):
        self.get_label_controller().select_by_prop(cond)
        self.update()

        
    @undo_label
    def set_celltype(self, ct):
        self.get_label_controller().set_celltype(ct)
        
    @undo_selected
    def select_small(self, large):
        old_selected = self.get_label_controller().get_selected()
        def undo():
            self.get_label_controller().set_selected(old_selected)

        self.get_label_controller().select_large(small)

        
    @undo_selected
    def select_neighbours(self):
        old_selected = self.get_label_controller().get_selected()
        def undo():
            self.get_label_controller().set_selected(old_selected)

        self.get_label_controller().select_neighbours()

        
    @undo_no_log
    def write_cell_graph(self, fn):
        self.get_label_controller().write_cell_graph(fn)
        
    @log
    def merge_watershed(self, level, new_name=None):
        
        if new_name is None:
            new_name = 'merge-'+str(len(self.all_label_controllers)+1)

        new_label = self.get_label_controller().merge_watershed(level)
        self._add_label(new_name, new_label)
        self._select_active_label(new_name)

        self.update()
        
    @log
    def make_borders(self):
        new_name = 'borders'+self.active_label
        borders = self.get_label_controller().make_borders()
        self._add_signal(new_name, borders)
        self._select_active_signal(new_name)
        self.update()

    @undo_selected
    def expand_selection(self, threshold):
        self.get_label_controller().expand_selection(threshold)

        
    @undo_selected
    def select(self, v):
        self.get_label_controller().select(v)

        
    @undo_selected
    def add_selected(self, v):
        self.get_label_controller().add_selected(v)
        #self.update('selected')

    @undo_selected
    def include_selected(self, v):
        s = self.get_label_controller().include_selected(v)
        if s:
            return True
        return False
    
    @undo_label
    def split_cc(self):
        self.get_label_controller().split_cc()

    @undo_label
    def merge_selected(self):
        self.get_label_controller().merge_selected()
        
        
    @undo_label
    def delete_selected(self):
        self.get_label_controller().delete_selected()
        
    @undo_label
    def remove_small_large_cells(self, small, large):
        self.get_label_controller().remove_small_large_cells(small, large)
        
    @undo_selected
    def select_by_celltype(self, ct):
        self.get_label_controller().select_by_celltype(ct)


    @undo_label
    def split_cell(self):
        selected = self.get_label_controller().get_selected()
        self.get_label_controller().split_cell(self.get_signal_stack(), selected[0])


    @undo_label
    def dilate_labels(self):
        self.get_label_controller().dilate_labels()
        
    @undo_label
    def set_omitted(self, ct_list):
        self.get_label_controller().set_omitted(ct_list)
        
    @undo_selected
    def set_selected(self, selected):
        self.get_label_controller().set_selected(selected)
        #self.update('selected')
        
        
        
    @undo_signal
    def blur_stack(self, radius):
        self.get_signal_controller().blur_stack(radius)

    @undo_signal
    def grey_closing(self, radius):
        self.get_signal_controller().grey_closing(radius)
        

    @undo_signal
    def subtract_bg(self):
        self.get_signal_controller().subtract_bg()

    @undo_signal
    def aniso(self):
        self.get_signal_controller().aniso()

    @undo_signal
    def clahe(self):
        self.get_signal_controller().clahe()


    @undo_signal
    def invert_signal(self):
        self.get_signal_controller().invert()
        

    @undo_signal
    def paint(self, p):
        self.get_signal_controller().paint(p)

    @undo_signal
    def paint_to(self, p):
        self.get_signal_controller().paint_to(p)

    @undo_signal
    def apply_power(self, power):
        self.get_signal_controller().apply_power(power)

    @undo_log
    def flip_z(self):
        def undo():
            self.get_signal_controller().flip_z()
        
        self.get_signal_controller().flip_z()
        return undo, None
        
    @undo_signal
    def equalize_stack(self):
        self.get_signal_controller().equalize_stack()
        
    @undo_label
    def calc_mean_signal(self):
        self.get_label_controller().calc_mean_signal(self.get_signal_stack())
        
    @undo_label
    def calc_min_signal(self):
        self.get_label_controller().calc_min_interior_signal(self.get_signal_stack())
        
    @undo_label
    def calc_mean_interior_signal(self):
        self.get_label_controller().calc_mean_interior_signal(self.get_signal_stack())
        
    def get_label_point(self, p):
        return self.get_label_controller().get_label_point(p)

        









        



