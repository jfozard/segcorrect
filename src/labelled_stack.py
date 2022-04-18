

import numpy as np
import numpy.linalg as la
import sys
import math
from math import sqrt

import scipy.ndimage as nd
import numpy.random as npr

from image_io.import_tiff import *
from image_io.import_image_dir import *

from utils_new import *
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix

from scalar_colmap import gen_prop_colmap, gen_array_colmap

#from shape import sphere, cylinder

#from labelled_mesh import LabelledMesh
#from solid_mesh import SolidMesh, SolidLabelMesh

def rotation_dir(x):
    
    x = x/la.norm(x)

    i = np.argmin(np.abs(x))
    c = np.zeros((3,))
    c[i] = 1.0
    
    y = np.cross(x,c)
    y = y/la.norm(y)

    z = np.cross(x,y)
    z = z/la.norm(z)
    
    R = np.array([y,z,x]).T

    return R

def make_transform(stretch_r, x0, x1):

    d = x1 - x0
    l = la.norm(d)
    
#    d = d/l

    A = np.array(((stretch_r, 0, 0), (0, stretch_r, 0), (0, 0, l)))

    R = rotation_dir(d)

    return np.dot(R, A)

"""
class LabelledGraphView(object):
    def __init__(self, stack, selected=[], label_subset=None):

        r_cell = 0.1
        r_link = 1.0

        centroids = stack.calculate_centroids()        
        connectivity, bdd_connectivity = stack.get_bdd_connectivity()

        spheres = []

        areas = stack.cell_props['area']

        def merge_meshes(m_list):
            verts = []
            tris = []
            labels = []
            v_count = 0
            for vl, tl, cl in m_list:
                verts.extend(vl)
                tris.extend((np.array(tl) + v_count).tolist())
                v_count += len(vl)
                labels.extend(cl)
            return np.array(verts), np.array(tris), np.array(labels)

        cv, ct = cylinder(N=8)
        sv, st = sphere(N=2)

        lut = stack.gen_colmap(celltypes=True, selected=selected, ct_weight=1.0)


        if label_subset is not None:
            for i in label_subset:
                c = centroids[i]
                r = r_cell*pow(areas[i], 0.33)
                n_sv = sv*r + c[np.newaxis, :]
                n_st = st
                n_sc = np.repeat([i], (n_sv.shape[0],))
                spheres.append((n_sv, n_st, n_sc))

        else:
            for i, c in centroids.items():
                if i>0:
                    c = centroids[i]
                    r = r_cell*pow(areas[i], 0.33)
                    n_sv = sv*r + c[np.newaxis, :]
                    n_st = st
                    n_sc = np.repeat([i], (n_sv.shape[0],))
#                    n_sc = np.tile(np.array(lut[i,:])/255.0, (n_sv.shape[0], 1)) 

                    spheres.append((n_sv, n_st, n_sc))

        u = np.max(connectivity.data)

        mean_bdd = bdd_connectivity.data/connectivity.data

        edge_colmap = gen_array_colmap(mean_bdd)

# Also label edges - done

        for i in range(connectivity.shape[0]):
            for k in range(connectivity.indptr[i], connectivity.indptr[i+1]):
                j = connectivity.indices[k]
#                if 0<i<j and i in centroids and j in centroids:
                if ((label_subset is None and (0<i<j)) or 
                     (0<i<j and i in label_subset and j in label_subset)):
                    c0 = centroids[i]
                    c1 = centroids[j]

                    r = sqrt(connectivity.data[k]/u)*r_link

                    T = make_transform(r, c0, c1)
                    
                    n_sv = np.dot(T, cv.T).T + np.array(c0)[np.newaxis,:]
                    n_st = ct
                    # n_sc = np.tile(edge_colmap[k,:], (n_sv.shape[0], 1)) 
                    
                    n_sc = np.repeat([k+lut.shape[0]], (n_sv.shape[0],))

                    spheres.append((n_sv, n_st, n_sc))
                                        

        # Now add all the cylinders

        verts, tris, labels = merge_meshes(spheres)

        self.solid_mesh = SolidLabelMesh.from_data(verts, tris, labels)

        colmap = np.vstack((lut/255.0, edge_colmap))

        self.solid_mesh.col_map = colmap

    def get_solid_mesh(self):
        return self.solid_mesh

"""

class LabelledStack(object):
    def __init__(self, labels, spacing, img_data=None):
#       self.orig_labels = np.array(labels)
        self.labels = labels
        self.spacing = np.array(spacing)

        self.img_data = img_data
        ll = np.unique(self.labels.flat)
        self.base_lut = np.zeros((65536, 3), dtype=np.uint8)
        for i in range(65536):
            j = i
            r = npr.rand(3)*255*(i>0)
            self.base_lut[i,:] = 0.8*r + 0.2
        self.ct_lut = (255*npr.rand(16,3)).astype(np.uint8)

        
        self.celltypes = dict((l, 0) for l in ll)
        self.cell_props = {}

        self.connectivity = None
        self.bdd_connectivity = None
        self.mean_bdd_connectivity = None
        self.ws_graph = None
        self.calc_cell_volumes()

    def set_stack(self, s):
        self.labels = s
        new_ll = np.unique(self.labels.flat)

        self.celltypes = dict((l,self.celltypes.get(l, 0)) for l in new_ll)
        self.cell_props = {}

        self.connectivity = None
        self.bdd_connectivity = None
        self.mean_bdd_connectivity = None
        self.ws_graph = None
        self.calc_cell_volumes()


    def update_cells(self):
        print ('update cells')

        labels, areas = np.unique(self.labels.flat, return_counts=True)
        new_labels = set(labels) - set(self.celltypes)
        for i in new_labels:
            self.celltypes[i] = 0
        old_labels = set(self.celltypes) - set(labels)
        for i in old_labels:
            del self.celltypes[i]
        
        if 'area' in self.cell_props:
            areas = np.asarray(areas).astype(float)*np.product(self.spacing)
            self.cell_props['area'] = dict(zip(labels, areas))
        print ('done update cells')

            
        
    def update_img_data(self, img_data):
        self.img_data = img_data
        self.img_data_updated = True
        
    def calc_mean_signal(self, signal=None):
        if signal is None:
            signal = self.img_data
        labels = np.unique(self.labels.flat)
        mean = nd.mean(signal, self.labels, labels)
        self.cell_props['mean_signal'] = dict(zip(labels, mean))

    def calc_min_signal(self, signal=None):
        if signal is None:
            signal = self.img_data
        labels = np.unique(self.labels.flat)
        mean = nd.minimum(signal, self.labels, labels)
        self.cell_props['min_signal'] = dict(zip(labels, mean))

    def calc_orig_min_signal(self, signal=None):
        if signal is None:
            signal = self.img_data
        labels = np.unique(self.labels.flat)
        mean = nd.minimum(signal, self.labels, labels)
        return dict(zip(labels, mean))

    def calc_mean_interior_signal(self, signal=None):
        if signal is None:
            signal = self.img_data

        max_l = nd.maximum_filter(self.labels, size=3)
        min_l = nd.minimum_filter(self.labels, size=3)
        eroded_labels = np.where(max_l == min_l, self.labels, 0)
        labels = np.unique(eroded_labels.flat)
        mean = nd.mean(signal, eroded_labels, labels)
        all_labels = np.unique(self.labels.flat)
        p = dict(zip(labels, mean))
        for l in all_labels:
            if l not in p:
                p[l] = 0.0

        self.cell_props['mean_interior_signal'] = p



    def calc_cell_volumes(self):
        labels, areas = np.unique(self.labels.flat, return_counts=True)
        areas = np.asarray(areas).astype(float)*np.product(self.spacing)
        self.cell_props['area'] = dict(zip(labels, areas))

    def gen_property_colmap(self, prop_name, selected=[], omitted=[]):
        N = np.max(self.labels.flat)+1

        lut = np.zeros((N,3))
        u_lut = gen_prop_colmap(self.cell_props[prop_name], N)
        lut[:N,:] = 255*u_lut 

#        print lut.shape, self.base_lut.shape, 'lut shapes'

#        print lut, N

#        for i, v in self.celltypes.items():
#            if v in omitted:
#                lut[i,:] = np.array((0,0,0))
            #elif i>0:
            #    lut[i,:] = 0.6*self.ct_lut[v,:] + 0.4*lut[i,:]
        if selected:
            for i in selected:
                lut[i,:] = np.clip(255*np.array((0.9,0,0)) + np.array((0.1, 0.3, 0.3))*self.base_lut[i&65535,:], 0, 255)
        return lut


    def gen_colmap(self, prop_name=None, celltypes=False, selected=[], omitted=[], ct_weight=0.6, grey_labels=False):
        if prop_name is not None:
            return self.gen_property_colmap(prop_name, selected, omitted)
        N = np.max(self.labels.flat)+1
#        lut = np.array(self.base_lut[:min(N,256*8192),:])
        m = min(N,256*8192)
        lut = np.tile(self.base_lut, ((m+65535)//65536,1))
        lut = lut[:m,:] 
        if grey_labels:
            lut = np.tile(np.mean(lut, axis=1),(3,1)).T.astype(np.uint8)
            print(lut.shape)
        if celltypes:
#            N = max(self.celltypes)+1
#            lut = np.array(self.base_lut)
            for i, v in self.celltypes.items():
                if i>0 and i<lut.shape[0] and v in omitted:
                    lut[i,:] = np.array((0,0,0))
                elif i>0 and i<lut.shape[0] and 0<=v<self.ct_lut.shape[0]:
                    lut[i,:] = ct_weight*self.ct_lut[v,:] + (1-ct_weight)*self.base_lut[(i&65535),:]
        print(selected, [self.celltypes.get(i, None) for i in selected])
        if selected:
            for i in selected:
                if i < lut.shape[0]:
                    lut[i,:] = np.clip(255*np.array((0.9,0,0)) + np.array((0.1, 0.3, 0.3))*lut[i,:], 0, 255)
        return lut

    def make_connectivity(self):
        A = self.labels
        idx_1 = []
        idx_2 = []
        for i in range(3):
            B = np.rollaxis(A, i)
            d = B[:-1,:,:]!=B[1:,:,:]
            idx_1.append(B[:-1,:,:][d])
            idx_2.append(B[1:,:,:][d])
        row = np.concatenate(idx_1+idx_2)
        col = np.concatenate(idx_2+idx_1)
        s = np.ones(row.shape)
        return coo_matrix((s, (row, col))).tocsr()

    def make_borders(self):
        A = self.labels
        d = np.zeros(A.shape, dtype=np.uint8)
        if A.shape[0]>1:
            u = A[:-1,:,:]!=A[1:,:,:]
            d[:-1,:,:] += u
            d[1:,:,:] += u
        if A.shape[1]>1:
            u = A[:,:-1,:]!=A[:,1:,:]
            d[:,:-1,:] += u
            d[:,1:,:] += u
        if A.shape[2]>1:
            u = A[:,:,:-1]!=A[:,:,1:]
            d[:,:,:-1] += u
            d[:,:,1:] += u
        d[d>0] = 1
        return d

    def make_bdd_connectivity(self):
        A = self.labels
        u = self.img_data
        idx_1 = []
        idx_2 = []
        data_1 = []
        data_2 = []
        for i in range(3):
            B = np.rollaxis(A, i)
            v = np.rollaxis(u, i)
            d = B[:-1,:,:]!=B[1:,:,:]
            idx_1.append(B[:-1,:,:][d])
            idx_2.append(B[1:,:,:][d])
            data_1.append(v[:-1,:,:][d])
            data_2.append(v[1:,:,:][d])
        row = np.concatenate(idx_1+idx_2)
        col = np.concatenate(idx_2+idx_1)
        data = np.concatenate(data_1+data_2)
        s = np.ones(row.shape)
        return coo_matrix((s, (row, col))).tocsr(), coo_matrix((data, (row, col))).tocsr()

    def make_ws_graph(self):
        A = self.labels
        u = self.img_data
        idx_1 = []
        idx_2 = []
#       data_1 = []
#       data_2 = []
        data_max = []
        if A.shape[0]>1:
            for i in range(3):
                B = np.rollaxis(A, i)
                v = np.rollaxis(u, i)
                d = B[:-1,:,:]!=B[1:,:,:]
                idx_1.append(B[:-1,:,:][d])
                idx_2.append(B[1:,:,:][d])
                data_max.append(np.maximum(v[:-1,:,:][d], v[1:,:,:][d]))
        else:
            A2 = A[0,:,:]
            u2 = u[0,:,:]
            for i in range(2):
                B = np.rollaxis(A2, i)
                v = np.rollaxis(u2, i)
                d = B[:-1,:]!=B[1:,:]
                idx_1.append(B[:-1,:][d])
                idx_2.append(B[1:,:][d])
                data_max.append(np.maximum(v[:-1,:][d], v[1:,:][d]))

        idx_1 = np.concatenate(idx_1)
        idx_2 = np.concatenate(idx_2)
        data_max = np.concatenate(data_max)

        l = max(np.max(idx_1), np.max(idx_2))

        print(idx_1.shape, l)

        large = 1e20
        mtx = {}
        for i, j, v in zip(idx_1, idx_2, data_max):
            if j>i:
                mtx[(i,j)] = min(v, mtx.get((i,j), large))
            else:
                mtx[(j,i)] = min(v, mtx.get((j,i), large))

        mtx2 = dict(list(mtx.items())+ [((j,i),v) for (i,j),v in mtx.items()])

        
#        m = dok_matrix((l+1,l+1))
#        m.update(mtx2)

        return mtx2


    def get_ws_graph(self):
        if self.ws_graph is None:
            self.ws_graph = self.make_ws_graph()
        return self.ws_graph


    def make_merge_tree(self):
        ws = self.make_ws_graph()
        lm_dict = self.calc_orig_min_signal()
        h_merges = []
        merge_tree = []

        for (a, b), l in ws.items():
            h_merges.append((l - lm_dict[a], a, b))

        h_merges.sort(reverse=True)

        while h_merges:
            h, l, o = h_merges.pop()
            if True:
                # h from l must be <= h from o
                # remove (h', o, l) from h_merges
                # set (h', l, b) to be (h'', o, b) where h'' is h' + label_min[l] - labem_min[o]
                # Think that label_min[l] >= label_min[o] always

                merge_tree.append((h, l, o))

        
                d_level = lm_dict[l] - lm_dict[o]

        
                to_remove = []
                for i in range(len(h_merges)):
                    if h_merges[i][1] == o and h_merges[i][2] == l:
                        to_remove.append(i)
                    if h_merges[i][1] == l and h_merges[i][2] == o:
                        to_remove.append(i)
                    if h_merges[i][1] == l and h_merges[i][2] != o:
                        h_merges[i] = (h_merges[i][0] + d_level, o, h_merges[i][2])
                    if h_merges[i][1] != o and h_merges[i][2] == l:
                        h_merges[i] = (h_merges[i][0] + d_level, h_merges[i][1], o)

                for i in to_remove[::-1]:
                    h_merges.pop(i)

                h_merges.sort(reverse=True)

        self.merge_tree = merge_tree

    def merge_watershed(self, h_max):
        try:
            mt = self.merge_tree
        except:
            
            self.make_merge_tree()
            mt = self.merge_tree

        im = np.array(self.labels)
        for l, a, b in mt:
            if l < h_max:
                print('merge', l, a, b)
                im[im==int(a)] = int(b)
        return im

    def get_connectivity(self):
        if self.connectivity is None:
            self.connectivity = self.make_connectivity()
        return self.connectivity

    
    def get_bdd_connectivity(self):
        if self.bdd_connectivity is None or self.img_data_updated:
            self.connectivity, self.bdd_connectivity = self.make_bdd_connectivity()
            self.img_data_updated = False
        return self.connectivity, self.bdd_connectivity

    def get_mean_bdd_connectivity(self):
        c, bc = self.get_bdd_connectivity()
#        ws = self.get_ws_graph()
#        return ws
        return csr_matrix((bc.data/c.data, c.indices, c.indptr), shape=c.shape)


    def calculate_centroids(self):
        l = np.unique(self.labels)
        cc = np.array(nd.center_of_mass(np.ones_like(self.labels), self.labels, l))
        
        print('centroid shape', cc.shape, l.shape)
        cc2 = cc[:,[2,1,0]]*self.spacing[np.newaxis, :]
        self.cell_props['centroid'] = dict(zip(l, cc2))

        """
        for i, c in zip(l, cc):
            print i, c, nd.center_of_mass(self.labels==i)

        quit()
        """
        return self.cell_props['centroid']


    def make_centroid_graph_obj(self,  selected=None, label_subset=None):
        self.view = LabelledGraphView(self, selected, label_subset)
        return self.view.get_solid_mesh()


    def write_cell_graph(self, filename):
        centroids = self.calculate_centroids()
        labels = np.unique(self.labels.flat)
        with open(filename, 'w') as f:
            f.write("label, celltype\n")
            for i in labels:
                print(i)
                f.write("%d, %d\n"%(i, self.celltypes[i]))

            connectivity, bdd_connectivity = self.get_bdd_connectivity()
        
            for i in range(connectivity.indptr.shape[0]-1):
                for k in range(connectivity.indptr[i],connectivity.indptr[i+1]):
                    f.write("%d, %d, %f, %f\n"%(i, connectivity.indices[k], connectivity.data[k], bdd_connectivity.data[k]))
