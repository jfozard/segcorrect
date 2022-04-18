from __future__ import print_function

import numpy as np
import scipy.ndimage as nd
import math

from scipy.interpolate import interp1d

def translate(x,y,z):
    # Make 4x4 translation matrix
    a = np.eye(4)
    a[0,3]=x
    a[1,3]=y
    a[2,3]=z
    return a

def scale(s):
    # Make 4x4 scale matrix
    a = np.eye((4))
    a[0,0]=s
    a[1,1]=s
    a[2,2]=s
    a[3,3]=1.0
    return a

def perspective(fovy, aspect, zNear, zFar):
    # 4x4 Perspective matrix (from OpenGL specs)
    f = 1.0/math.tan(fovy/2.0/180*math.pi)
    return np.array(((f/aspect, 0, 0, 0), (0,f,0,0), (0,0,(zFar+zNear)/(zNear-zFar), 2*zFar*zNear/(zNear-zFar)), (0, 0, -1, 0)))


def process_mask(mask):
 #   mask = nd.binary_dilation(mask, iterations=2)
    mask = nd.binary_fill_holes(mask)
    mask = remove_spikes(mask)
 #   mask = nd.binary_erosion(mask, iterations=2, border_value=1)
    label, num_label = nd.label(mask)
    size = np.bincount(label.ravel())
    print(size, np.argmax(size[1:])+1)
    return (label == (np.argmax(size[1:]) + 1))#.astype(np.uint8)

def largest_cc(mask):
    label, num_label = nd.label(mask)
    size = np.bincount(label.ravel())
    return (label == (np.argmax(size[1:]) + 1))#.astype(np.uint8)
    


def remove_spikes(mask):
    ball = nd.generate_binary_structure(3, 2)
    mask2 = nd.binary_erosion(mask, ball, iterations=2)
    mask = np.logical_and(nd.binary_dilation(mask, ball, iterations=2), mask)
    return mask

def process_mask2(mask):
    ball = nd.generate_binary_structure(3, 2)
    mask = nd.binary_erosion(mask, ball, border_value=1)
    mask = nd.binary_dilation(mask, ball, iterations=2)
    mask = nd.binary_erosion(mask, ball, border_value=1)
    return mask

def smooth_mask(mask, radius):
    m = mask.astype(np.float32)
    m = m / np.max(m)
    mask = nd.gaussian_filter(m, radius) > 0.5

    return mask



def equalize(im, r1=10, r2=3):
    Ni = (im.shape[0]+r1-1)/r1
    Nj = (im.shape[1]+r1-1)/r1
    Nk = (im.shape[2]+r1-1)/r1
    med_val = np.zeros((Ni,Nj,Nk), dtype=im.dtype)
    max_val = np.zeros((Ni,Nj,Nk), dtype=im.dtype)
    min_val = np.zeros((Ni,Nj,Nk), dtype=im.dtype)
    
    for i in range(Ni):
        for j in range(Nj):
            for k in range(Nk):
                ri = slice(i*r1, min((i+1)*r1, im.shape[0]))
                rj = slice(j*r1, min((j+1)*r1, im.shape[1]))
                rk = slice(k*r1, min((k+1)*r1, im.shape[2]))
                med_val[i, j, k] = np.median(im[ri, rj, rk])
                max_val[i, j, k] = np.amax(im[ri, rj, rk])
                min_val[i, j, k] = np.amin(im[ri, rj, rk])

    mid_ri = [ 0.5*(i*r1 + min((i+1)*r1 -1, im.shape[0]-1)) for i in range(Ni)]
    mid_rj = [ 0.5*(i*r1 + min((i+1)*r1 -1, im.shape[1]-1)) for i in range(Nj)]
    mid_rk = [ 0.5*(i*r1 + min((i+1)*r1 -1, im.shape[2]-1)) for i in range(Nk)]

    i_map = interp1d(mid_ri, range(Ni), bounds_error=False, fill_value='extrapolate')
    j_coords = interp1d(mid_rj, range(Nj), bounds_error=False, fill_value='extrapolate')(range(im.shape[1]))
    j_coords = np.maximum(np.minimum(j_coords, Nj-1), 0)
    k_coords = interp1d(mid_rk, range(Nk), bounds_error=False, fill_value='extrapolate')(range(im.shape[2]))
    k_coords = np.maximum(np.minimum(k_coords, Nk-1), 0)

    print("calculated maximum and median")
                
    max_val = nd.gaussian_filter(max_val, r2)
    med_val = nd.gaussian_filter(med_val, r2)
    ave_max = np.amax(max_val)/8.0
    max_val = np.maximum(ave_max, max_val)
#    med_val = np.minimum(max_val - ave_max/2, np.maximum(ave_max/2.0, med_val))
    min_val = nd.gaussian_filter(min_val, r2)

    print("blurred max and median")

    coords_jk = np.array(np.meshgrid(k_coords, j_coords))
    rs = np.array(im, dtype=np.float32)
    for i in range(im.shape[0]):
        ir = min(max(0, i_map(i)), Ni-1)
        if ir<Ni-1:
            theta = ir - int(ir) 
            med_val_ir = (1-theta)*med_val[int(ir), :, :] + theta*med_val[int(ir)+1, :, :]
            max_val_ir = (1-theta)*max_val[int(ir), :, :] + theta*max_val[int(ir)+1, :, :]
            min_val_ir = (1-theta)*min_val[int(ir), :, :] + theta*min_val[int(ir)+1, :, :]
        else:
            med_val_ir = med_val[Ni-1, :, :]
            max_val_ir = max_val[Ni-1, :, :]
            min_val_ir = min_val[Ni-1, :, :]

        med_val_i = nd.map_coordinates(med_val_ir, coords_jk, order=1, mode='reflect')
        max_val_i = nd.map_coordinates(max_val_ir, coords_jk, order=1, mode='reflect')
        min_val_i = np.mean(min_val_ir)
#        rs[i] = np.where(rs[i]>med_val_i, 0.5*(1.0+(rs[i] - med_val_i)/(max_val_i-med_val_i)), 0.5*(rs[i]-min_val)/(med_val_i-min_val))
        rs[i] = (rs[i] - min_val_i)/(max_val_i-min_val_i)
        rs[i] = np.maximum(rs[i], 0.0)
        rs[i] = np.minimum(rs[i], 1.0)
        if im.dtype == np.uint8:
            rs[i] *= 255.0
        
    if im.dtype == np.uint8:
        return (rs).astype(np.uint8)
    else:
        return rs


def equalize_z(im, r1=10, r2=5):
    Ni = (im.shape[0]+r1-1)/r1
    Nj = (im.shape[1]+r1-1)/r1
    Nk = (im.shape[2]+r1-1)/r1
    med_val = np.zeros((Ni,Nj,Nk), dtype=np.uint8)
    max_val = np.zeros((Ni,Nj,Nk), dtype=np.uint8)
    
    for i in range(Ni):
        for j in range(Nj):
            for k in range(Nk):
                ri = slice(i*r1, min((i+1)*r1, im.shape[0]))
                rj = slice(j*r1, min((j+1)*r1, im.shape[1]))
                rk = slice(k*r1, min((k+1)*r1, im.shape[2]))
                med_val[i, j, k] = np.median(im[ri, rj, rk])
                max_val[i, j, k] = np.amax(im[ri, rj, rk])

    mid_ri = [ 0.5*(i*r1 + min((i+1)*r1 -1, im.shape[0]-1)) for i in range(Ni)]
    mid_rj = [ 0.5*(i*r1 + min((i+1)*r1 -1, im.shape[1]-1)) for i in range(Nj)]
    mid_rk = [ 0.5*(i*r1 + min((i+1)*r1 -1, im.shape[2]-1)) for i in range(Nk)]

    i_map = interp1d(mid_ri, range(Ni), bounds_error=False, fill_value='extrapolate')
    j_coords = interp1d(mid_rj, range(Nj), bounds_error=False, fill_value='extrapolate')(range(im.shape[1]))
    j_coords = np.maximum(np.minimum(j_coords, Nj-1), 0)
    k_coords = interp1d(mid_rk, range(Nk), bounds_error=False, fill_value='extrapolate')(range(im.shape[2]))
    k_coords = np.maximum(np.minimum(k_coords, Nk-1), 0)

    print("calculated maximum and median")
                
    max_val = nd.gaussian_filter(max_val, r2)
    med_val = nd.gaussian_filter(med_val, r2)
    ave_max = np.amax(max_val)/8.0
    max_val = np.maximum(ave_max, max_val)
    med_val = np.minimum(max_val - ave_max/2, np.maximum(ave_max/2.0, med_val))
    min_val = 0.0

    print("blurred max and median")

    coords_jk = np.array(np.meshgrid(k_coords, j_coords))
    rs = im.astype(float)
    for i in range(im.shape[0]):
        ir = min(max(0, i_map(i)), Ni-1)
        if ir<Ni-1:
            theta = ir - int(ir) 
            med_val_ir = (1-theta)*med_val[int(ir), :, :] + theta*med_val[int(ir)+1, :, :]
            max_val_ir = (1-theta)*max_val[int(ir), :, :] + theta*max_val[int(ir)+1, :, :]
        else:
            med_val_ir = med_val[Ni-1, :, :]
            max_val_ir = max_val[Ni-1, :, :]

        med_val_i = nd.map_coordinates(med_val_ir, coords_jk, order=1, mode='reflect')
        max_val_i = nd.map_coordinates(max_val_ir, coords_jk, order=1, mode='reflect')
        rs[i] = np.where(rs[i]>med_val_i, 0.5*(1.0+(rs[i] - med_val_i)/(max_val_i-med_val_i)), 0.5*(rs[i]-min_val)/(med_val_i-min_val))
        rs[i] = np.maximum(rs[i], 0.0)
        rs[i] = np.minimum(rs[i], 1.0)
        rs[i] *= 255.0
        
        
    return (rs).astype(np.uint8)




def make_iso_surface(level, ma, spacing):
    verts, tris = make_iso(ma, level)
    verts = verts * np.array(spacing, dtype=np.float32)[np.newaxis,:]
    m = ProjectionMesh.from_data(verts, tris)
    print("np.amin(m.degree), np.amax(m.degree)", np.amin(m.degree), np.amax(m.degree))

    return m

def triangulate_polygon(pts, p, n):
    # Really bad (fan around first vertex) triangulation of a polygon
    tris = []
    for i in range(1, len(pts)-1):
        tris.append((0, i, i+1))
    print('triangulate', pts, tris)
    return tris

def sorted_tuple(a, b):
    # Sort a pair of value and return tuple
    if a>b:
        return (b,a)
    else:
        return (a,b)

def slice_cell(p, n, verts, tris):
    """ 
    Cut a triangulated cell into two pieces, by intersecting
    it with the plane passing through point p and with normal n
    
    Cell defined by a list of triangles tris, each of which is
    an integer index to the list of vertex positions (numpy vectors)
    verts.

    We need to know which triangles lie on which side of the cut

    """

    cut_edges = {}
    over_tris = []
    under_tris = []

    new_verts = [v for v in verts]
    new_poly_edges = []


    def tri_next(i):
        return i+1 if i<2 else 0

    def tri_prev(i):
        return i-1 if i>0 else 2


    for t in tris:
        h = [np.dot(verts[i]-p,n) for i in t]
        s = [cmp(v,0) for v in h]
        under = sum([v<0 for v in h])
        over = sum([v>0 for v in h])

        if over==0:

            under_tris.append(t)
            if under==1:
                i = s.index(-1)
                new_poly_edges.append((t[tri_prev(i)], t[tri_next(i)]))
            continue


        if under==0:
            over_tris.append(t)
            continue


        if under==2:
            i = s.index(1)
            idx = t[i]
            i_prev = tri_prev(i)
            idx_prev = t[i_prev]
            i_next = tri_next(i)
            idx_next = t[i_next]
            
            st = sorted_tuple(idx_prev, idx)
            try:
                idx_c_prev = cut_edges[st]
            except KeyError:
                print(i_prev, h[i_prev], verts[idx], i, h[i], verts[idx_prev])
                c_prev = (h[i_prev]*verts[idx]-h[i]*verts[idx_prev]) \
                    /(h[i_prev]-h[i])
                idx_c_prev = len(new_verts)
                new_verts.append(c_prev)
                cut_edges[st] = idx_c_prev

            st = sorted_tuple(idx_next, idx)

            try:
                idx_c_next = cut_edges[st]
            except KeyError:
                c_next = (h[i_next]*verts[idx]-h[i]*verts[idx_next]) \
                /(h[i_next]-h[i])

                idx_c_next = len(new_verts)
                new_verts.append(c_next)
                cut_edges[st] = idx_c_next
            
            over_tris.append((idx_c_prev,idx,idx_c_next))
            under_tris.append((idx_c_next, idx_next, idx_prev))
            under_tris.append((idx_prev, idx_c_prev, idx_c_next))

            new_poly_edges.append((idx_c_next, idx_c_prev))

            continue

        if over==2:
            i = s.index(-1)
            idx = t[i]
            i_prev = tri_prev(i)
            idx_prev = t[i_prev]
            i_next = tri_next(i)
            idx_next = t[i_next]
            
            st = sorted_tuple(idx_prev, idx)
            try:
                idx_c_prev = cut_edges[st]
            except KeyError:
                c_prev = (h[i_prev]*verts[idx]-h[i]*verts[idx_prev]) \
                    /(h[i_prev]-h[i])
                idx_c_prev = len(new_verts)
                new_verts.append(c_prev)
                cut_edges[st] = idx_c_prev

            st = sorted_tuple(idx_next, idx)
            try:
                idx_c_next = cut_edges[st]
            except KeyError:
                c_next = (h[i_next]*verts[idx]-h[i]*verts[idx_next]) \
                /(h[i_next]-h[i])

                idx_c_next = len(new_verts)
                new_verts.append(c_next)
                cut_edges[st] = idx_c_next
        
            under_tris.append((idx_c_prev,idx,idx_c_next))
            over_tris.append((idx_c_next, idx_next, idx_prev))
            over_tris.append((idx_prev, idx_c_prev, idx_c_next))

            new_poly_edges.append((idx_c_prev, idx_c_next))
        
            continue

        if over+under==2:
            i = s.index(0)
            idx = t[i]
            i_prev = tri_prev(i)
            idx_prev = t[i_prev]
            i_next = tri_next(i)
            idx_next = t[i_next]

            c = (h[i_prev]*verts[idx]-h[i]*verts[idx_prev]) \
                /(h[i_prev]-h[i])
            idx_c = len(new_verts)
            new_verts.append(c)
            
            
            if s[i_prev]==1:
                over_tris.append((idx_prev, i, idx_c))
                under_tris.append((i, idx_next, idx_c))
                new_poly_edges.append((idx_c,i))
            else:
                under_tris.append((idx_prev, i, idx_c))
                over_tris.append((i, idx_next, idx_c))
                new_poly_edges.append((i,idx_c))
            continue
        raise 


    
    print(new_poly_edges)
    ordered_polys = []
    while new_poly_edges:
        ordered_poly = []
        p0 = new_poly_edges.pop()[1]
        while new_poly_edges:
            for q0, q1 in new_poly_edges:
                if q0==p0:
                    ordered_poly.append(p0)
                    new_poly_edges.remove((q0,q1))
                    p0 = q1
                    break
            else:
                break
        if ordered_poly:
            ordered_poly.append(p0)
            ordered_polys.append(ordered_poly)

    print(ordered_polys)
    cut_tris = []
    for ordered_poly in ordered_polys:
        tt = triangulate_polygon([new_verts[i] for i in ordered_poly], p, n)
        for a,b,c in tt:
            cut_tris.append((ordered_poly[a], ordered_poly[b], ordered_poly[c]))
    return new_verts, under_tris+cut_tris


def slice_cell_cut(p, n, verts, tris):
    """ 
    Cut a triangulated cell into two pieces, by intersecting
    it with the plane passing through point p and with normal n
    
    Cell defined by a list of triangles tris, each of which is
    an integer index to the list of vertex positions (numpy vectors)
    verts.

    We need to know which triangles lie on which side of the cut

    """

    cut_edges = {}
    over_tris = []
    under_tris = []

    new_verts = [v for v in verts]
    new_poly_edges = []


    def tri_next(i):
        return i+1 if i<2 else 0

    def tri_prev(i):
        return i-1 if i>0 else 2


    for t in tris:
        h = [np.dot(verts[i]-p,n) for i in t]
        s = [cmp(v,0) for v in h]
        under = sum([v<0 for v in h])
        over = sum([v>0 for v in h])

        if over==0:

            under_tris.append(t)
            if under==1:
                i = s.index(-1)
                new_poly_edges.append((t[tri_prev(i)], t[tri_next(i)]))
            continue


        if under==0:
            over_tris.append(t)
            continue


        if under==2:
            i = s.index(1)
            idx = t[i]
            i_prev = tri_prev(i)
            idx_prev = t[i_prev]
            i_next = tri_next(i)
            idx_next = t[i_next]
            
            st = sorted_tuple(idx_prev, idx)
            try:
                idx_c_prev = cut_edges[st]
            except KeyError:
                print(i_prev, h[i_prev], verts[idx], i, h[i], verts[idx_prev])
                c_prev = (h[i_prev]*verts[idx]-h[i]*verts[idx_prev]) \
                    /(h[i_prev]-h[i])
                idx_c_prev = len(new_verts)
                new_verts.append(c_prev)
                cut_edges[st] = idx_c_prev

            st = sorted_tuple(idx_next, idx)

            try:
                idx_c_next = cut_edges[st]
            except KeyError:
                c_next = (h[i_next]*verts[idx]-h[i]*verts[idx_next]) \
                /(h[i_next]-h[i])

                idx_c_next = len(new_verts)
                new_verts.append(c_next)
                cut_edges[st] = idx_c_next
            
            over_tris.append((idx_c_prev,idx,idx_c_next))
            under_tris.append((idx_c_next, idx_next, idx_prev))
            under_tris.append((idx_prev, idx_c_prev, idx_c_next))

            new_poly_edges.append((idx_c_next, idx_c_prev))

            continue

        if over==2:
            i = s.index(-1)
            idx = t[i]
            i_prev = tri_prev(i)
            idx_prev = t[i_prev]
            i_next = tri_next(i)
            idx_next = t[i_next]
            
            st = sorted_tuple(idx_prev, idx)
            try:
                idx_c_prev = cut_edges[st]
            except KeyError:
                c_prev = (h[i_prev]*verts[idx]-h[i]*verts[idx_prev]) \
                    /(h[i_prev]-h[i])
                idx_c_prev = len(new_verts)
                new_verts.append(c_prev)
                cut_edges[st] = idx_c_prev

            st = sorted_tuple(idx_next, idx)
            try:
                idx_c_next = cut_edges[st]
            except KeyError:
                c_next = (h[i_next]*verts[idx]-h[i]*verts[idx_next]) \
                /(h[i_next]-h[i])

                idx_c_next = len(new_verts)
                new_verts.append(c_next)
                cut_edges[st] = idx_c_next
        
            under_tris.append((idx_c_prev,idx,idx_c_next))
            over_tris.append((idx_c_next, idx_next, idx_prev))
            over_tris.append((idx_prev, idx_c_prev, idx_c_next))

            new_poly_edges.append((idx_c_prev, idx_c_next))
        
            continue

        if over+under==2:
            i = s.index(0)
            idx = t[i]
            i_prev = tri_prev(i)
            idx_prev = t[i_prev]
            i_next = tri_next(i)
            idx_next = t[i_next]

            c = (h[i_prev]*verts[idx]-h[i]*verts[idx_prev]) \
                /(h[i_prev]-h[i])
            idx_c = len(new_verts)
            new_verts.append(c)
            
            
            if s[i_prev]==1:
                over_tris.append((idx_prev, i, idx_c))
                under_tris.append((i, idx_next, idx_c))
                new_poly_edges.append((idx_c,i))
            else:
                under_tris.append((idx_prev, i, idx_c))
                over_tris.append((i, idx_next, idx_c))
                new_poly_edges.append((i,idx_c))
            continue
        raise 


    
    print(new_poly_edges)
    ordered_polys = []
    while new_poly_edges:
        ordered_poly = []
        p0 = new_poly_edges.pop()[1]
        while new_poly_edges:
            for q0, q1 in new_poly_edges:
                if q0==p0:
                    ordered_poly.append(p0)
                    new_poly_edges.remove((q0,q1))
                    p0 = q1
                    break
            else:
                break
        if ordered_poly:
            ordered_poly.append(p0)
            ordered_polys.append(ordered_poly)

    print(ordered_polys)
    cut_tris = []
    for ordered_poly in ordered_polys:
        tt = triangulate_polygon([new_verts[i] for i in ordered_poly], p, n)
        for a,b,c in tt:
            cut_tris.append((ordered_poly[a], ordered_poly[b], ordered_poly[c]))
    return new_verts, cut_tris

