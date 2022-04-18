

import numpy as np
import scipy.ndimage as nd
import scipy.linalg as la

from scipy.sparse import dok_matrix, csr_matrix
from scipy.spatial import cKDTree

from ply_parser import *

from math import sqrt, ceil, floor, log, exp
from random import randint, choice, random

from scipy.interpolate import Rbf

import numpy.random as npr

from numpy.linalg import svd, eig

from collections import defaultdict

from inferno import inferno

import heapq

from scipy.interpolate import interp1d

from OpenGL.GL.ARB.vertex_array_object import *

white_blue = interp1d([0.0,1.0],np.array([[1.0,1.0,1.0],[0.0,0.0,1.0]]).T)


def calc_split_tris(verts, tris, labels, signal):
    new_verts = list(verts)
    new_tris = []
    new_labels = list(labels)
    new_signal = list(signal)

    edge_verts = {}

    Nv = len(new_verts)
    for tri_idx in tris:
        tri_labels = [labels[i] for i in tri_idx]
        N_labels = len(set(tri_labels))
        if N_labels>1:
            shared_labels = list(set(tri_labels))
            tri_verts = [verts[i] for i in tri_idx]
            c = np.mean(tri_verts, axis=0)
            new_verts.extend([c for _ in shared_labels])
            new_labels.extend(shared_labels)
            s = np.mean([signal[i] for i in tri_idx])
            new_signal.extend([s for _ in shared_labels])
            centre_map = dict(zip(shared_labels, range(Nv, Nv+N_labels)))
            Nv += N_labels
            for i in range(3):
                j = (i+1)%3
                k = (i+2)%3
                idx_j = tri_idx[j]
                idx_k = tri_idx[k]
                label_j = tri_labels[j]
                label_k = tri_labels[k]
                if label_j==label_k:
                    # No need to split this edge
                    new_tris.append((idx_j, idx_k, centre_map[label_j]))
                else:
                    # Need to split edge
                    try:
                        lj = edge_verts[(label_j, idx_j, idx_k)]
                    except KeyError:
                        ec = 0.5*(tri_verts[j] + tri_verts[k])
                        new_verts.append(ec)
                        es = 0.5*(signal[idx_j] + signal[idx_k])
                        new_signal.append(es)
                        new_labels.append(label_j)
                        edge_verts[(label_j, idx_j, idx_k)] = Nv
                        lj = Nv
                        Nv += 1
                    try:
                        lk = edge_verts[(label_k, idx_k, idx_j)]
                    except KeyError:
                        ec = 0.5*(tri_verts[j] + tri_verts[k])
                        new_verts.append(ec)
                        es = 0.5*(signal[idx_j] + signal[idx_k])
                        new_signal.append(es)
                        new_labels.append(label_k)
                        edge_verts[(label_k, idx_k, idx_j)] = Nv
                        lk = Nv
                        Nv += 1

                    new_tris.append((idx_j, lj, centre_map[label_j]))
                    new_tris.append((idx_k, centre_map[label_k], lk))
        else:
            new_tris.append(tri_idx)
    return new_verts, new_tris, new_labels, new_signal

    
def erode_labels(mesh, labels, inplace=True):
    if not inplace:
        labels = np.array(labels)
    bg = set(i for i,v in enumerate(labels) if v==0)
    A = mesh.connectivity
    for i in range(len(labels)):
        if labels[i]>0:
            for j in A.indices[A.indptr[i]:A.indptr[i+1]]:            
                if j in bg:
                    labels[i] = 0
                    break
    return labels


def calculate_vertex_normals(verts, tris):
    v_array = np.array(verts)
    tri_array = np.array(tris, dtype=int)
    tri_pts = v_array[tri_array]
    n = np.cross( tri_pts[:,1] - tri_pts[:,0], 
                  tri_pts[:,2] - tri_pts[:,0])


    v_normals = np.zeros(v_array.shape)

    for i in range(tri_array.shape[0]):
        for j in tris[i]:
            v_normals[j,:] += n[i,:]

    nrms = np.sqrt(v_normals[:,0]**2 + v_normals[:,1]**2 + v_normals[:,2]**2)
    
    v_normals = v_normals / nrms.reshape((-1,1))

    return v_normals



def surface_watershed(mesh, signal, seeds, max_level=float('inf'), inplace=True):
    # On-mesh "watershed" to extend state labels to larger regions.
    # Not clear on exactly how to do this, so we'll pick a simple method
        
    # Priority queue for pixels (s[i], i) which neighbour
    # labelled pixels

    h = []

    labelled = 0
    Nv = len(mesh.verts)
    border = set()
    if not inplace:
        state = np.array(seeds)
    else:
        state = seeds
    A = mesh.connectivity
    # Loop over all pixels; for labelled pixels push unlabelled neighbours
    # into queue
    for i in range(Nv):
        if state[i]!=0:
            labelled +=1
            # Add nb vertices to queue
            for j in A.indices[A.indptr[i]:A.indptr[i+1]]:
                if j not in border:
                    heapq.heappush(h, (signal[j], j))
                    border.add(j)
    # Now pull pixel with smallest level, label with state of nb
    while h:
        l, i = heapq.heappop(h)
        border.remove(i)

        if l>max_level:
            break
        nbs = A.indices[A.indptr[i]:A.indptr[i+1]]
        n_states = state[nbs]
        n_signal = signal[nbs]
        state[i] = min((a,b) for a,b in zip(n_signal, n_states) if b!=0)[1]
        labelled +=1
#        print labelled, Nv, l, i
#        print state[i]
        # Push neighbours
        for j in A.indices[A.indptr[i]:A.indptr[i+1]]:
            if state[j]==0 and (j not in border):
                border.add(j)
                heapq.heappush(h, (signal[j], j))
    
    return state

"""
def triangulate_polygon(points, p, n):
    if abs(np.dot(n, [1,0,0]))>1e-3:
        s = np.array((1,0,0))
    else:
        s = np.array((0,1,0))

    x = np.cross(n, s)
    x = x/la.norm(x)
    y = np.cross(n, x)
    pp = [(np.dot(v,x), np.dot(v,y)) for v in points]
    return geom.ear_clip(pp)
"""



def local_normalize(verts, signal, r=20, sampling=0.002):
    sub_idx = [i for i in range(len(verts)) if random()<sampling]
    sub_verts = [verts[i] for i in sub_idx]
    print "samples ", len(sub_verts)
    tree = cKDTree(verts)
    sub_min = np.zeros((len(sub_verts),), dtype=signal.dtype)
    sub_med = np.zeros((len(sub_verts),), dtype=signal.dtype)
    sub_max = np.zeros((len(sub_verts),), dtype=signal.dtype)
    min_threshold = np.percentile(signal, 20)
    med_threshold = np.percentile(signal, 50)
    max_threshold = np.max(signal)
    for i in range(len(sub_verts)):
        idx = tree.query_ball_point(sub_verts[i], r)
        d = signal[idx] #[v for v in signal[idx] if v>min_threshold]
        if len(d)>0:
            sub_min[i] = np.percentile(d, 20)
            sub_med[i] = np.percentile(d, 50)#signal[idx])
            sub_max[i] = np.max(d)#signal[idx])
        else:
            sub_min[i] = min_threshold
            sub_med[i] = med_threshold
            sub_max[i] = max_threshold
        if i%100==0:
            print '*', i, sub_med[i], sub_max[i]

    sub_med = np.maximum(min_threshold, sub_med)
    sub_max = np.maximum(min_threshold, sub_max)

    min_rbf = Rbf([_[0] for _ in sub_verts], [_[1] for _ in sub_verts], [_[2] for _ in sub_verts], sub_min, smooth=10, function='linear')

 
    med_rbf = Rbf([_[0] for _ in sub_verts], [_[1] for _ in sub_verts], [_[2] for _ in sub_verts], sub_med, smooth=10, function='linear')

    #print heapy.heap()
    max_rbf = Rbf([_[0] for _ in sub_verts], [_[1] for _ in sub_verts], [_[2] for _ in sub_verts], sub_max, smooth=10, function='linear')


    r_lo = max(0.2*(med_threshold - min_threshold), 0.1)
    r_hi = max(0.2*(max_threshold - med_threshold), 0.1)

    nsignal = np.zeros(signal.shape, dtype=np.float32)
    for i in range(len(verts)):
        
        low = min_rbf(*verts[i])
        median = med_rbf(*verts[i])
        high = max_rbf(*verts[i])

        s = signal[i]
        if s>median:
            nsignal[i] = 255.0*min(0.5*(s - median)/(max(high - median, r_hi))+0.5, 1)
        else:
            nsignal[i] = 255.0*max(0.5-0.5*(median - s)/(max(median - low, r_lo)), 0.0)
        if i%1000==0:
            print i, nsignal[i], signal[i], median, high
    #nsignal = np.maximum((nsignal - median), 0)
    high = float(np.max(nsignal))
    low = float(np.min(nsignal))
    nsignal = 255*((nsignal-low)/(high-low))

    return nsignal.astype(np.uint8)
    

def triangulate_polygon(points, p, n):
    if abs(np.dot(n, [1,0,0]))>1e-3:
        s = np.array((1,0,0))
    else:
        s = np.array((0,1,0))

    x = np.cross(n, s)
    x = x/la.norm(x)
    y = np.cross(n, x)
    pp = [(np.dot(v,x), np.dot(v,y)) for v in points]
    return geom.ear_clip(pp)

def calc_aniso(pts, pt_weights):
    if len(pt_weights)==0:
        return 0.0
    mw = np.mean(pt_weights)
    if mw < 1e-12:
        return 0.0
    c = np.mean(pts*pt_weights[:,np.newaxis], axis=0)/mw
    npts = (pts - c[np.newaxis,:])*pt_weights[:,np.newaxis]
    s = eig(np.dot(npts.T, npts))
    s = s[0]
    s.sort()
    if s[1]>0:
        mu = sqrt(s[2]/s[1])-1
    else:
        mu = 0.0
    return mu



class Mesh(object):
    def __init__(self):
        self.verts = [] # array of vertex positions
        self.threshold = float("inf")

    def make_connectivity_matrix(self):
        n = len(self.verts)
        connections = {}
        for t in self.tris:
            connections[(t[0], t[1])] = 1
            connections[(t[1], t[0])] = 1
            connections[(t[1], t[2])] = 1
            connections[(t[2], t[1])] = 1
            connections[(t[2], t[0])] = 1
            connections[(t[0], t[2])] = 1
        A = dok_matrix((n,n))
        A.update(connections)
        A = A.tocsr()
        D = A.sum(axis=1)
        self.connectivity = A
        self.degree = D
        
    def smooth_signal(self, s, delta=0.1, iterations=1):
        s = s[:, np.newaxis]
        for i in range(iterations):
            s = (1-delta)*s + delta*self.connectivity.dot(s)/(self.degree+1e-12)        
        s = np.squeeze(np.asarray(s))
        return s

    def connected_components_l(self, to_visit):
        components = []
        cc = 1
        s = set()
        A = self.connectivity
        while to_visit:
            s.add(iter(to_visit).next())
            cl = []
            while s:
                i = s.pop()
                to_visit.remove(i)
                cl.append(i)
                neighbours = A.indices[A.indptr[i]:A.indptr[i+1]]
                for j in neighbours:
                    if j in to_visit:
                        s.add(j)
            #print cc, len(to_visit)
            cc +=1
            components.append(cl)
        return components


    def connected_components(self, mask):
        print "begin cc"
        to_visit = set(i for i,v in enumerate(mask) if v>0)
        components = np.zeros(self.vert_signal.shape, dtype='int')
        cc = 1
        s = set()
        A = self.connectivity
        while to_visit:
            s.add(iter(to_visit).next())
            while s:
                i = s.pop()
                to_visit.remove(i)
                components[i] = cc
                neighbours = A.indices[A.indptr[i]:A.indptr[i+1]]
                for j in neighbours:
                    if j in to_visit:
                        s.add(j)
            #print cc, len(to_visit)
            cc +=1
        return components

    def remove_small_cells(self, area_threshold=0.1):
        ca, mu, cv = self.calculate_cell_areas_aniso(return_cv=True)
        a = list(ca[i] for i in ca if i>0)
        mean_a = np.mean(a)
        threshold_a = area_threshold * mean_a
        print len(cv)
        removed = 0
        for i in ca:
            if ca[i] < threshold_a or len(cv[i])==1:
                self.vert_labels[cv[i]]=0
#                print self.vert_labels[cv[i]]
                removed += 1
        print 'removed', removed

    def split_disconnected_cells(self):

        A = self.connectivity

        for i in range(A.shape[0]):
            if A.indptr[i]==A.indptr[i+1]:
                print 'isolated', i

        cv = defaultdict(set)
        for i, l in enumerate(self.vert_labels):
            cv[l].add(i)

        new_cl = np.max(self.vert_labels)+1
        for l in cv:
            if l>0:
                cc = self.connected_components_l(cv[l])
                if len(cc)>1:
                    print cc
                for c in cc[1:]:
                    print 'split', l, new_cl, len(c), len(cc)
                    self.vert_labels[c] = new_cl
                    new_cl += 1

    def find_local_minima(self, s):
        minima = []
        A = self.connectivity
        for i in range(len(self.verts)):
#            print A.indptr[i], A.indptr[i+1], s[0, A.indices[A.indptr[i]:A.indptr[i+1]]].shape
            if A.indptr[i]<A.indptr[i+1]:
                min_other = np.amin(s[A.indices[A.indptr[i]:A.indptr[i+1]]])
                if s[i] < min_other:
                    minima.append(i)
        return minima


    def load_ply2(self, fn):
        descr, data = parse_ply2(fn)
        self.descr = descr
        self.data = data
        NV = len(data['vertex'][0])
        NF = len(data['face'][0])
        print 'NF', NF
        verts = []
        vert_norms = []
        vert_signal = []
        vert_labels = []
        print data['vertex'][1]
        x_idx = data['vertex'][1].index('x')
        y_idx = data['vertex'][1].index('y')
        z_idx = data['vertex'][1].index('z')

        # Does the surface have normal data?
        if 'nx' in data['vertex'][1]:
            nx_idx = data['vertex'][1].index('nx')
            ny_idx = data['vertex'][1].index('ny')
            nz_idx = data['vertex'][1].index('nz')
            has_normal = True
        else:
            has_normal = False
        if 'signal' in data['vertex'][1]:
            s_idx = data['vertex'][1].index('signal')
        else:
            s_idx = data['vertex'][1].index('red')
        if 'state' in data['vertex'][1]:
            l_idx = data['vertex'][1].index('state')
            has_label = True
        else:
            has_label = False
#        l_idx = data['vertex'][1].index('red')
        for v in data['vertex'][0]:
            verts.append((v[x_idx], v[y_idx], v[z_idx]))
            if has_normal:
                vert_norms.append(np.array((v[nx_idx], v[ny_idx], v[nz_idx])))
            vert_signal.append(v[s_idx])
            if has_label:
                vert_labels.append(int(v[l_idx]))
            else:
                vert_labels.append(0 if v[s_idx]>0 else 1)
        #del data['vertex']



        print 'done_vertex'

        v_array=np.array(verts,dtype='float32')
        v_array=v_array-np.sum(v_array,0)/len(v_array)
        bbox=(np.min(v_array,0),  np.max(v_array,0) )
        v_array=v_array-0.5*(bbox[0] + bbox[1])
        bbox=(np.min(v_array,0),  np.max(v_array,0) )
        zoom=1.0/la.norm(bbox[1])
        self.verts = [np.array(v) for v in v_array]
        self.vert_signal = np.array(vert_signal, dtype=np.float32)
        self.orig_vert_signal = np.array(vert_signal, dtype=np.float32)
        self.vert_labels = np.array(vert_labels, dtype=np.int32)
        self.tris = []
        for f in data['face'][0]:
            vv = f[0]
            tris = []
            for i in range(len(vv)-2):
                tris.append((vv[0], vv[i+1], vv[i+2]))

            self.tris.extend(tris)

        if not has_normal:
            print 'Calculate surface normals'
            # Area weighted surface normals (would prefer angle-weighted)
            self.vert_norms = calculate_vertex_normals(v_array, self.tris)
        else:
            self.vert_norms = np.array(vert_norms)
        
        print 'done_face'
        print 'make matrix'
        self.make_connectivity_matrix()
        self.calculate_vertex_areas()
        self.calculate_edge_perimeters()
#        print np.min(self.vert_signal), np.max(self.vert_signal), np.mean(self.vert_signal)
        print 'done matrix'
        return zoom


    def generate_arrays(self):
        print 'start_arrays'

        self.orig_verts = True
        npr.seed(1)
        tris = []
        v_out=np.array(self.verts,dtype=np.float32) 
        idx_out=np.array(self.tris,dtype=np.uint32)
        n_out=np.array(self.vert_norms,dtype=np.float32)
        nl = np.max(self.vert_labels)+1
        self.label_colmap=npr.random((nl,3))
        self.label_colmap[0,:] = 10.0/255.0
#        col_out = np.array(self.label_colmap[self.vert_labels, :], dtype=np.float32)
        nsignal = 0.1+ 0.9*(self.vert_signal / float(np.max(self.vert_signal)))
        col_out = np.tile(nsignal,(3,1)).T

        self.v_out = v_out
        self.idx_out = idx_out

        return v_out, n_out, col_out, idx_out

    def update_vertex_labels(self, newlabels):
        nl = np.max(newlabels)+1
        old_nl = self.label_colmap.shape[0]
        if nl>old_nl:
            new_colmap = np.zeros((nl, 3), dtype=np.float32)
            new_colmap[:old_nl,:] = self.label_colmap
            new_colmap[old_nl:] = npr.random((nl-old_nl, 3))
            self.label_colmap = new_colmap
        self.vert_labels = newlabels

    def update_vertex_signal(self, newsignal):
        self.vert_signal = newsignal
        
    def update_threshold(self, threshold):
        self.threshold = threshold
        
    def apply_mesh_cols_obj(self, obj, view_labels=True, colmap_labels=None, split_tris=False):
        if colmap_labels == None:
            colmap = self.label_colmap
            nl = np.max(self.vert_labels)+1
            old_nl = self.label_colmap.shape[0]
            if nl>old_nl:
                new_colmap = np.zeros((nl, 3), dtype=np.float32)
                new_colmap[:old_nl,:] = self.label_colmap
                new_colmap[old_nl:] = npr.random((nl-old_nl, 3))
                self.label_colmap = new_colmap
                colmap = new_colmap
        elif colmap_labels == 'area':
            colmap = self.gen_area_colmap()
        elif colmap_labels == 'aniso':
            colmap = self.gen_aniso_colmap()

        if split_tris:
            new_verts, new_tris, new_labels, new_signal = calc_split_tris(self.verts, self.tris, self.vert_labels, self.vert_signal)

            new_labels = np.array(new_labels)
            v_out=np.array(new_verts,dtype=np.float32) 
            idx_out=np.array(new_tris,dtype=np.uint32)

            n_out= calculate_vertex_normals(v_out, idx_out).astype(np.float32)

            new_signal = np.array(new_signal)

            nsignal = 0.1+ 0.9*(new_signal / float(np.max(new_signal)))
            nsignal = nsignal.astype(np.float32)

            mask = new_signal >= self.threshold
        
            col_signal = np.vstack((nsignal*mask, nsignal, nsignal*mask)).T

            if view_labels:
                col_out = np.array(colmap[new_labels, :], dtype=np.float32)
                row_mask = new_labels==0        
                col_out[row_mask, :] = col_signal[row_mask,:]
            else:
                col_out = col_signal

 #           v_out, n_out, col_out, idx_out_ = self.generate_arrays()
  
            print v_out.shape, v_out.dtype, n_out.shape, n_out.dtype, col_out.shape, col_out.dtype

            obj.vb = np.concatenate((v_out,n_out,col_out),axis=1)

            obj.elVBO.set_array(idx_out) 
            obj.elCount = len(idx_out.flatten())

#            glBindVertexArray(obj.vao)
            obj.vtVBO.bind()
            obj.vtVBO.set_array(obj.vb)
            obj.vtVBO.copy_data()
#            glBindVertexArray(0)
            obj.vtVBO.unbind()
            self.orig_verts=False
        else:

            nsignal = 0.1+ 0.9*(self.vert_signal / float(np.max(self.vert_signal)))

            if not self.orig_verts:
                v_out, n_out, col_out, idx_out = self.generate_arrays()
                obj.vb = np.concatenate((v_out,n_out,col_out),axis=1)
                obj.elVBO.set_array(idx_out) 
                obj.elCount = len(idx_out.flatten())
                self.orig_verts = True
            mask = self.vert_signal >= self.threshold
        
            col_signal = np.vstack((nsignal*mask, nsignal, nsignal*mask)).T

            if view_labels:
                col_out = np.array(colmap[self.vert_labels, :], dtype=np.float32)
                row_mask = self.vert_labels==0        
                col_out[row_mask, :] = col_signal[row_mask,:]
            else:
                col_out = col_signal

            obj.vb[:,6:9] = col_out
            obj.vtVBO.bind()
#            glBindVertexArray(obj.vao)
            obj.vtVBO.set_array(obj.vb)
            obj.vtVBO.copy_data()
#            glBindVertexArray(0)
            obj.vtVBO.unbind()

    def calculate_vertex_areas(self):
        n = len(self.verts)
        va = np.zeros((n,), dtype=np.float32)
        v_array=np.array(self.verts,dtype='float32') 
        tri_array=np.array(self.tris,dtype='i')
        tri_pts=v_array[tri_array]
        n = np.cross( tri_pts[:,1 ] - tri_pts[:,0], 
                   tri_pts[:,2 ] - tri_pts[:,0])
        ta = np.sqrt(n[:,0]**2+n[:,1]**2+n[:,2]**2)
        for t, a in zip(self.tris, ta):
            va[list(t)] += a/6.0
        self.vertex_area = va
        return va

    def calculate_cell_areas_aniso(self, return_cv=False):
        ca = {}
        mu = {}
        
        cv = defaultdict(list)
        for i, l in enumerate(self.vert_labels):
            cv[l].append(i)

        print '#cells', len(cv)

        for l, vl in cv.iteritems():
                va = [self.vertex_area[i] for i in vl]
                vx = [self.verts[i] for i in vl]
                ca[l] = sum(va)
                mu[l] = calc_aniso(np.array(vx), np.array(va))
        if return_cv:
            return ca, mu, cv
        else:
            return ca, mu

    def gen_area_colmap(self):
        ca, mu = self.calculate_cell_areas_aniso()

        a = list(ca[i] for i in ca if i>0)
        min_a = min(a)
        max_a = max(a)
        range_a = max(1e-12, max_a - min_a)
        print min_a, max_a
        
        col_map = np.zeros_like(self.label_colmap)
        for i, v in ca.iteritems():
            if i>0:
                col_map[i,:] = white_blue((v - min_a)/range_a)
        return col_map


    def gen_aniso_colmap(self):
        ca, mu = self.calculate_cell_areas_aniso()

        a = list(mu[i] for i in mu if i>0)
        print a

        min_a = min(a)
        max_a = max(a)
        range_a = max(1e-12, max_a - min_a)
        print min_a, max_a
            
        col_map = np.zeros_like(self.label_colmap)
        for i, v in mu.iteritems():
            if i>0:
                col_map[i,:] = white_blue((v - min_a)/range_a)
        return col_map
    
    

    def calculate_cell_max_z(self):
        mz = defaultdict(float)
        for x, l in zip(self.verts, self.vert_labels):
            mz[l] = max(mz[l], x[2])
        return mz


    def calculate_edge_lengths(self):
        try:
            A = self.connectity
        except AttributeError:
            self.make_connectivity_matrix
            A = self.connectivity
        n = len(self.verts)
        d = []
        dd = {}
        for i in range(self.n):
            for j in A.indicies[A.indptr[i]:A.indptr[i+1]]:
                l = la.norm(verts[i] - verts[j])
                dd[(i,j)] = l
                d.append(la.norm(verts[i] - verts[j]))
        self.edge_lengths = d
        return dd

    def calculate_edge_perimeters(self):
        ep = defaultdict(float)
        n = len(self.verts)
        E = dok_matrix((n,n), dtype=np.float32)
        for t0, t1, t2 in self.tris:
            v0, v1, v2 = self.verts[t0], self.verts[t1], self.verts[t2]
            h0 = la.norm(v0 - 0.5*(v1+v2))/3.0
            h1 = la.norm(v1 - 0.5*(v0+v2))/3.0
            h2 = la.norm(v2 - 0.5*(v0+v1))/3.0
            ep[(t0, t1)] += h2
            ep[(t1, t0)] += h2
            ep[(t1, t2)] += h0
            ep[(t2, t1)] += h0
            ep[(t0, t2)] += h1
            ep[(t2, t0)] += h1

        E.update(ep)
        E = E.tocsr()
        self.edge_perimeters = E
        self.edge_perimeter_sums = np.squeeze(np.asarray(E.sum(axis=1)))
        return E


    def save_ply(self, filename, state):
        data = self.data
        descr = self.descr
        vert_props = [u[0] for u in descr[0][2]]
        print 'vert_props', vert_props
        if 'state' in vert_props:
            i = vert_props.index('state')
        else:
            descr[0][2].append(('state',['int']))
            i = None
        if 'signal' in vert_props:
            j = vert_props.index('signal')
        else:
            descr[0][2].append(('signal',['float']))
            j = None


        n = []
        for el, s, v in zip(data['vertex'][0], state, self.vert_signal):
            m = list(el)
            if i:
                m[i] = s
            else:
                m.append(s)
            if j:
                m[j] = v
            else:
                m.append(v)
            n.append(tuple(m))
        data['vertex'] = (n, data['vertex'][1])
        
        write_ply2(filename, descr, data)


    def save_ply_rgb(self, filename, obj):
        
        write_ply2(filename, descr, data)

            


        
