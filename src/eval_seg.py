
from __future__ import print_function

import numpy as np
from scipy.sparse import coo_matrix
import scipy.ndimage as nd
import numpy.random as npr
from math import log, sqrt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.ndimage import distance_transform_edt, find_objects, binary_erosion

from PIL import Image
import sys

import matplotlib.pyplot as plt


def hausdorff(X, Y, label_pairs, boundary=False, average=False):
    # Calculate [average] [boundary] Hausdorff distances between pairs of labels in X and Y
    # X and Y should have sequential labels
    obj_X = find_objects(X)
    obj_Y = find_objects(Y)
    distances = []
    for x, y in label_pairs:
        assert(x>0)
        assert(y>0)
        bbox_x = obj_X[x-1]
        bbox_y = obj_Y[y-1]
        bbox_comb = [slice(min(a.start, b.start), max(a.stop, b.stop)) for a,b in zip(bbox_x, bbox_y)]
        mask_X = X[bbox_comb]==x
        mask_Y = Y[bbox_comb]==y
        if boundary:
            mask_X ^= binary_erosion(mask_X)
            mask_Y ^= binary_erosion(mask_Y)
        dist_X = distance_transform_edt(~mask_X)
        dist_Y = distance_transform_edt(~mask_Y)
        if average:
            d_X_Y = np.sum(mask_X*dist_Y)/np.sum(mask_X)
            d_Y_X = np.sum(mask_Y*dist_X)/np.sum(mask_Y)
            d = 0.5*(d_X_Y + d_Y_X)
        else:
            d_X_Y = np.max(mask_X*dist_Y)
            d_Y_X = np.max(mask_Y*dist_X)
            d = max(d_X_Y, d_Y_X)
        #print x, y, d
        distances.append(d)
    return np.array(distances)

def mean_hausdorff(X, Y, boundary=False, average=False):
    IoU_map = matching_IoU(X,Y)
    label_pairs = list(zip(IoU_map.keys(), IoU_map.values()))
#    print label_pairs
    return np.mean(hausdorff(X,Y,label_pairs, boundary=boundary, average=average))

def precision_recall_IoU(X, Y, return_M=False):
    IoU_map = matching_IoU(X,Y)
    M = len(IoU_map)
    lX = np.unique(X)
    lY = np.unique(Y)
    nX = len(lX[lX>0])
    nY = len(lY[lY>0])
    if return_M:
        return M/float(nX), M/float(nY), M
    else:
        return M/float(nX), M/float(nY)


def gen_ndata(X, Y):
    # Calculation of terms used in Rand / Jaccard
    
    w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX= np.unique(X, return_inverse=True)
    lY, iY = np.unique(Y, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

#    cX = np.array([np.sum(w[X==j]) for j in lX])
#    cY = np.array([np.sum(w[Y==j]) for j in lY])
 
    
    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]

    nX = lX.shape[0]
    nY = lY.shape[0]

    M = coo_matrix((w.flatten(), (I, J)), shape=(nX, nY))
#    r = M.todense().A/float(n)

    cX = np.ravel(M.sum(axis=1))
    cY = np.ravel(M.sum(axis=0))



#    M = np.array(M.todense())

    M2 = (M.multiply(M)).sum()
    cX2 = np.sum(cX*cX)
    cY2 = np.sum(cY*cY)
    
    n11 = 0.5*(M2-n)
    n10 = 0.5*(cX2 - M2)
    n01 = 0.5*(cY2 - M2)
    n00 = 0.5*n*(n-1)-n11-n10-n01

    return n00, n01, n10, n11


def rand_index(X,Y):
    n00, n01, n10, n11 = gen_ndata(X,Y)
    n = np.product(X.shape)
    R = 1 - (n11+n00)/(0.5*n*(n-1))
    return R
    
def jaccard_index(X,Y):
    n00, n01, n10, n11 = gen_ndata(X,Y)
    n = np.product(X.shape)
    J = 1 - n11/(n11+n01+n10)
    return J

    
def calc_cM(seg, exact, weights=None):

    X = seg
    Y = exact
    
    # Calculation of IoU
    if weights is None:
        w = np.ones(X.shape)
        n = np.product(X.shape)
    else:
        w = weights
        n = np.sum(w)
# 
    lX, iX = np.unique(X, return_inverse=True)
    lY, iY = np.unique(Y, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

#    cX = np.array([np.sum(w[X==j]) for j in lX])
#    cY = np.array([np.sum(w[Y==j]) for j in lY])
 
    cX = nd.sum(w, labels=X, index=lX)
    cY = nd.sum(w, labels=Y, index=lY)
    
    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]

#    print w.shape, I.shape, J.shape
    
    M = coo_matrix((w.flatten(), (I, J))).tocsr()

    # loop over all objects in seg

    return lX, lY, cX, cY, n, M

def calc_IoU(X, Y, w=None, thresholds=np.arange(0.5,1.0,0.05)):
    lX, lY, cX, cY, n, M = calc_cM(X, Y, w)
    return calc_IoU_pp(lX, lY, cX, cY, n, M, thresholds)


def calc_IoU2(X, Y, w=None, thresholds=np.arange(0.5,1.0,0.05)):
    lX, lY, cX, cY, n, M = calc_cM(X, Y, w)
    return calc_IoU_pp2(lX, lY, cX, cY, n, M, thresholds)


def calc_IoU_map(lX, lY, aX, aY, n, M):

    IoU_map = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]
            mask = lj>0
            j = j[mask]
            intersection_area = intersection_area[mask]
            if len(j)>0:
                j_area = aY[j]
                union_area = i_area + j_area - intersection_area
                IoU = intersection_area/union_area
                k = np.argmax(IoU)
                IoU_map[i] = (j[k], IoU[k])
    return IoU_map


def calc_acme_criterion(X, Y, w=None, threshold=0.75, return_criterion=False):
    # Max_g ((Ra ^ Rg) /Rg) for each a in X
    lX, lY, aX, aY, n, M = calc_cM(X, Y, w)
    acme_map = {}
    acme_vals = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]
            mask = lj>0
            lj = lj[mask]
            j = j[mask]
            intersection_area = intersection_area[mask]
            j_area = aY[j]
            union_area = i_area + j_area - intersection_area
            c = intersection_area/j_area
            if len(c>0):
                k = np.argmax(c)
                if c[k]>=threshold:
                    acme_map[li] = lj[k]
                acme_vals[li] = c[k]
            else:
                acme_vals[li] = 1.0
    if return_criterion:
        return acme_map, acme_vals
    else:
        return acme_map
        


def matching_IoU(X, Y, threshold=0.5, w=None, return_best=False):
    lX, lY, aX, aY, n, M = calc_cM(X, Y, w)
    IoU_map = {}
    IoU_best = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]
            mask = lj>0
            lj = lj[mask]
            j = j[mask]
            intersection_area = intersection_area[mask]
            if len(j)>0:
                j_area = aY[j]
                union_area = i_area + j_area - intersection_area
                IoU = intersection_area/union_area
                k = np.argmax(IoU)
                if IoU[k]>=threshold:
                    IoU_map[li] = lj[k]
                IoU_best[li] = IoU[k]
            else:
                IoU_best[li] = None
    if return_best:
        return IoU_map, IoU_best
    else:
        return IoU_map
            

#    print IoU_map

def calc_IoU(lX, lY, aX, aY, n, M, thresholds):    

    IoU_map = calc_IoU_map(lX, lY, aX, aY, n, M)

    ni = sum(lX>0)
    nj = sum(lY>0)

    total_IoU = 0.0
    for t in thresholds:
        matched_i = set()
        matched_j = set()
        for i, li in enumerate(lX):
            if li>0:
                j, v = IoU_map[i]
                if v > t:
                    matched_i.add(i)
                    matched_j.add(j)
        assert(len(matched_i)==len(matched_j))
        tp = len(matched_i)
        fp = ni - len(matched_i)
        fn = nj - len(matched_j)
        print("t, tp, fp, fn", t, tp, fp, fn)
        total_IoU += tp/float(tp + fn + fp)

    IoU = total_IoU/len(thresholds)
    return IoU

def calc_IoU_pp2(lX, lY, aX, aY, n, M, thresholds):

    IoU_map = calc_IoU_map(lX, lY, aX, aY, n, M)
    
    ni = sum(lX>0)
    nj = sum(lY>0)

    total_IoU = 0.0
    for t in thresholds:
        matched_i = set()
        matched_j = set()
        for i, li in enumerate(lX):
            if li>0:
                j, v = IoU_map[i]
                if v > t:
                    matched_i.add(i)
                    matched_j.add(j)
        assert(len(matched_i)==len(matched_j))
        tp = len(matched_i)
        fp = ni - len(matched_i)
        fn = nj - len(matched_j)
        print("t, tp, fp, fn", t, tp, fp, fn)
        total_IoU += tp/float(tp + fn)

    IoU = total_IoU/len(thresholds)
    return IoU


"""
def calc_IoU_pp2(lX, lY, aX, aY, n, M, thresholds):

    IoU_map = {}
    for i, li in enumerate(lX):
        i_area = aX[i]
        if li>0:
            j = M.indices[M.indptr[i]:M.indptr[i+1]]
            intersection_area = M.data[M.indptr[i]:M.indptr[i+1]]
            lj = lY[j]
            mask = lj>0
            j = j[mask]
            intersection_area[mask]
            j_area = aY[j]
            union_area = i_area + j_area - intersection_area
            IoU = intersection_area/union_area
            k = np.argmax(IoU)
            IoU_map[i] = (j[k], IoU[k])


#    print IoU_map

    
    ni = sum(lX>0)
    nj = sum(lY>0)

    total_IoU = 0.0
    for t in thresholds:
        matched_i = set()
        matched_j = set()
        for i, li in enumerate(lX):
            if li>0:
                j, v = IoU_map[i]
                if v > t:
                    matched_i.add(i)
                    matched_j.add(j)
        assert(len(matched_i)==len(matched_j))
        tp = len(matched_i)
        fp = ni - len(matched_i)
        fn = nj - len(matched_j)
        print "t, tp, fp, fn", t, tp, fp, fn
        total_IoU += tp/float(tp + fn)

    IoU = total_IoU/len(thresholds)
    return IoU
"""



def voi2(X, Y):
    n = np.product(X.shape)
    lX, cX = np.unique(X, return_counts=True)
    lY, cY = np.unique(Y, return_counts=True)
    nX = len(lX)
    nY = len(lY)
    p = cX/float(n)
    q = cY/float(n)


#    print lX, cX, p

    r = np.zeros((nX, nY))
    for i in range(nX):
        for j in range(nY):
            r[i,j] = np.sum((X==lX[i]) & (Y==lY[j]))/float(n)

#    print r
    VIJ = - np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]) + np.log(r/q[np.newaxis, :])))

    return VIJ


def voi_nw(X, Y):
    n = np.product(X.shape)

    lX, iX, cX = np.unique(X, return_counts=True, return_inverse=True)
    lY, iY, cY = np.unique(Y, return_counts=True, return_inverse=True)
#    nX = len(lX)
#    nY = len(lY)
    p = cX/float(n)
    q = cY/float(n)


    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]


    r = coo_matrix((np.ones(I.shape), (I, J)))
    r = r.todense().A/float(n)

#    print nX, nY, r.shape

#    print lX, cX, p

#    r = np.zeros((nX, nY))
#    for i in range(nX):
#        for j in range(nY):
#            r[i,j] = np.sum((X==lX[i]) & (Y==lY[j]))/float(n)

#    print r
    VIJ = - np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]) + np.log(r/q[np.newaxis, :])))

    return VIJ


def nmi(X, Y):

    w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX, cX = np.unique(X, return_counts=True, return_inverse=True)
    lY, iY, cY = np.unique(Y, return_counts=True, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

    cX = np.array([np.sum(w[X==j]) for j in lX])
    cY = np.array([np.sum(w[Y==j]) for j in lY])
    
    p = cX/float(n)
    q = cY/float(n)


    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]


    r = coo_matrix((w.flatten(), (I, J)))
    r = r.todense().A/float(n)

#    print nX, nY, r.shape

#    print lX, cX, p

#    r = np.zeros((nX, nY))
#    for i in range(nX):
#        for j in range(nY):
#            r[i,j] = np.sum((X==lX[i]) & (Y==lY[j]))/float(n)

#    print r
    #VIJ = - np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]) + np.log(r/q[np.newaxis, :])))

    MI = np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]/q[np.newaxis,:])))

    NMI = 1.0 - MI/(log(len(lX)) + log(len(lY)))
    
    return NMI


def nmi2(X, Y):

    w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX, cX = np.unique(X, return_counts=True, return_inverse=True)
    lY, iY, cY = np.unique(Y, return_counts=True, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

    cX = np.array([np.sum(w[X==j]) for j in lX])
    cY = np.array([np.sum(w[Y==j]) for j in lY])
    
    p = cX/float(n)
    q = cY/float(n)


    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]


    r = coo_matrix((w.flatten(), (I, J)))
    r = r.todense().A/float(n)

#    print nX, nY, r.shape

#    print lX, cX, p

#    r = np.zeros((nX, nY))
#    for i in range(nX):
#        for j in range(nY):
#            r[i,j] = np.sum((X==lX[i]) & (Y==lY[j]))/float(n)

#    print r
    #VIJ = - np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]) + np.log(r/q[np.newaxis, :])))

    HI = -np.sum(p*(np.nan_to_num(np.log(p))))
    HJ = -np.sum(q*(np.nan_to_num(np.log(q))))
    
    MI = np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]/q[np.newaxis,:])))

    NMI = MI/sqrt(HI*HJ)
    
    return NMI

    

def voi(X, Y, w=None):
    if w is None:
        w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX, cX = np.unique(X, return_counts=True, return_inverse=True)
    lY, iY, cY = np.unique(Y, return_counts=True, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

    cX = np.array([np.sum(w[X==j]) for j in lX])
    cY = np.array([np.sum(w[Y==j]) for j in lY])
    
    p = cX/float(n)
    q = cY/float(n)


    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]


    r = coo_matrix((w.flatten(), (I, J)))
    r = r.todense().A/float(n)

#    print nX, nY, r.shape

#    print lX, cX, p

#    r = np.zeros((nX, nY))
#    for i in range(nX):
#        for j in range(nY):
#            r[i,j] = np.sum((X==lX[i]) & (Y==lY[j]))/float(n)

#    print r
    VIJ = - np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]) + np.log(r/q[np.newaxis, :])))

    return VIJ


def voi_different(X, Y, w=None):
    if w is None:
        w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX, cX = np.unique(X, return_counts=True, return_inverse=True)
    lY, iY, cY = np.unique(Y, return_counts=True, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

    cX = np.array([np.sum(w[X==j]) for j in lX])
    cY = np.array([np.sum(w[Y==j]) for j in lY])
    
    p = cX/float(n)
    q = cY/float(n)


    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]


    r = coo_matrix((w.flatten(), (I, J)))
    r = r.todense().A/float(n)

#    print nX, nY, r.shape

#    print lX, cX, p

#    r = np.zeros((nX, nY))
#    for i in range(nX):
#        for j in range(nY):
#            r[i,j] = np.sum((X==lX[i]) & (Y==lY[j]))/float(n)

#    print r
    #VIJ = - np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]) + np.log(r/q[np.newaxis, :])))

    VIJ = -np.sum(p*np.nan_to_num(np.log(p))) -np.sum(q*np.nan_to_num(np.log(q))) - 2*np.sum(r*np.nan_to_num(np.log(r/p[:, np.newaxis]/q[np.newaxis,:])))
    
    return VIJ


def corresponding_cells3(X, Y):
    
    w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX= np.unique(X, return_inverse=True)
    lY, iY = np.unique(Y, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

#    cX = np.array([np.sum(w[X==j]) for j in lX])
#    cY = np.array([np.sum(w[Y==j]) for j in lY])
 
    
    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]

    nX = lX.shape[0]
    nY = lY.shape[0]

    M = coo_matrix((w.flatten(), (I, J)), shape=(nX, nY))
#    r = M.todense().A/float(n)

    M2 = M.tocsr()
    links_R = {}
    overlap_R = {}
    for i, li in enumerate(lX):
        j = M2.indices[M2.indptr[i]:M2.indptr[i+1]]
        c = M2.data[M2.indptr[i]:M2.indptr[i+1]]
        k = np.argmax(c)
        links_R[li] = lY[j[k]]
        overlap_R[li] = k/np.sum(c)

    M2 = (M.T).tocsr()
    links_S = {}
    overlap_S = {}
    for i, li in enumerate(lY):
        j = M2.indices[M2.indptr[i]:M2.indptr[i+1]]
        c = M2.data[M2.indptr[i]:M2.indptr[i+1]]
        k = np.argmax(c)
        links_S[li] = lX[j[k]]
        overlap_S[li] = k/np.sum(c)

    return links_R, links_S, overlap_R, overlap_S



def corresponding_cells2(R, S):
    l_R = np.unique(R)
    l_R = l_R[l_R>0]
    com_R = nd.center_of_mass(R>0, R, l_R)
    links_R = {}


    for l, c in zip(l_R, com_R):
        links_R[l] = int(S[tuple(map(int, c))])

    l_S = np.unique(S)
    l_S = l_S[l_S>0]


    com_S = nd.center_of_mass(S>0, S, l_S)
    links_S = {}
    for l, c in zip(l_S, com_S):
        links_S[l] = int(R[tuple(map(int, c))])

    return links_R, links_S


def corresponding_cells(R, S, w=None):
    if w is None:
        w = np.ones(R.shape)

    l_R = np.unique(R)
#    l_R = l_R[l_R>0]
    links_R = {}


    for l in l_R:
        u = S[R==l]
        w2 = w[R==l]


        if len(u)>0:
            links_R[l] = max(map(lambda val: (np.sum((u==val)*w2), val), set(u)))[1]
        else:
            links_R[l] = 0


    l_S = np.unique(S)
#    l_S = l_S[l_S>0]

    links_S = {}
    for l in l_S:
        u = R[S==l]
        w2 = w[S==l]
        if len(u)>0:
            links_S[l] = max(map(lambda val: (np.sum((u==val)*w2), val), set(u)))[1]
        else:
            links_S[l] = 0

    

    
    return links_R, links_S

"""
def per_cell_JI(X, Y, w=None):

    # X is gold standard, Y is segmentation
    
    if w is None:
        w = np.ones(X.shape)

    n = np.sum(w)

    lX, iX, cX = np.unique(X, return_counts=True, return_inverse=True)
    lY, iY, cY = np.unique(Y, return_counts=True, return_inverse=True)

    # have lX = [0, 2, 3] iX = [0, 1, 1, 2, 2]
    # want cX = [np.sum(w[iX==j]) for j in range(len(lX))]

    cX = np.array([np.sum(w[X==j]) for j in lX])
    cY = np.array([np.sum(w[Y==j]) for j in lY])
    
    I = iX.flatten() #[X.flatten()]
    J = iY.flatten() #[Y.flatten()]


    r = coo_matrix((w.flatten(), (I, J)))

    # r is now | (X==i) ^ (Y==j)|_w

    # cX is | (X==i) |_w

    #  ind matching cell in Y for each cell in X 

    r = r.to_csr()
    """

def max_freq(u, w):
    # Find most frequent (weighted) value in u

    v, indices = np.unique(u, return_inverse=True)

    c = np.bincount(indices, weights=w)
    i = np.argmax(c)
    overlap = c[i]
    
    s = v[i]

    return s, overlap

def test_max_freq():
    print(max_freq([2,3,4],[0.1, 0.2, 0.3]),  (4, 0.3))

    print(max_freq([2,3,4,2,2],[0.1, 0.2, 0.3, 0.1, 0.15]),  (2, 0.35))
    

def calc_overlaps(u, w):
    # Find most frequent (weighted) value in u

    v, indices = np.unique(u, return_inverse=True)

    c = np.bincount(indices, weights=w)

    return v, c

    
def per_cell_JI(R, S, w=None):
    if w is None:
        w = np.ones(R.shape)

    l_R = np.unique(R)
    per_cell_JI = {}

    seg = 0.0
    
    for l in l_R:
        
        u = S[R==l]
        w2 = w[R==l]
        
        area_l = np.sum(w2)



        s, overlap = max_freq(u, w2)
                
        if overlap>0.5*area_l:
            union_area = np.sum(w[np.logical_or(R==l, S==s)])
            seg += overlap/union_area

        v, c = calc_overlaps(u, w2)

        unions = [ np.sum(w[np.logical_or(R==l, S==t)]) for t in v ]

        JI = np.array(c)/np.array(unions)
        i = np.argmax(JI)
        per_cell_JI[l] = JI[i]
        
    seg = seg/len(l_R)
            
    return np.mean(per_cell_JI.values()), per_cell_JI
            
def cell_detection(R, S, w=None):
    # percentage of cells in R that have a reciprocal link to a cell in S; 
    # both cells only having one incomming link
    links_R, links_S, _, _ = corresponding_cells3(R, S)

    count_R = dict((r, 0) for r in links_R)
    count_S = dict((s, 0) for s in links_S)

    for r in links_S.itervalues():
        if r in count_R:
            count_R[r] += 1

    for s in links_R.itervalues():
        if s in count_S:
            count_S[s] += 1
        
    correct = 0


    for r in links_R:
        if count_R[r]==1:
            s = links_R[r]
            if count_S[s]==1 and links_S[s]==r:
                correct += 1

    return float(correct) / len(links_R)

def cell_detection_relabel(R0, S0, w=None):
    # percentage of cells in R that have a reciprocal link to a cell in S; 
    # both cells only having one incomming link
    # ignore cells in R0 mapping to zero in S0

    lR, R = np.unique(R0, return_inverse=True)

    if 0 not in lR:
        R +=1
    lS, S = np.unique(S0, return_inverse=True)
    if 0 not in lS:
        S+=1

    links_R, links_S, _, _ = corresponding_cells3(R, S)

    print(0 in links_R)
    print(0 in links_S)
    print(sum(v==0 for v in links_R.values()), links_R.get(0))

    links2_R = dict((i,v) for i,v in links_R.iteritems() if i>0 and v>0)
    links2_S = dict((i,v) for i,v in links_S.iteritems() if i>0)

    count_R = dict((r, 0) for r in links2_R)
    count_S = dict((s, 0) for s in links2_S)

    
    for r in links2_S.itervalues():
        if r in count_R:
            count_R[r] += 1

    for s in links2_R.itervalues():
        if s in count_S:
            count_S[s] += 1
        
    correct = 0

    matched_R = []
    matched_S = []
    for r in links2_R:
        if count_R[r]==1:
            s = links2_R[r]
            if count_S[s]==1 and links_S[s]==r:
                correct += 1
                matched_R.append(r)
                matched_S.append(s)

    unmatched_R = [r for r in links_R if r not in matched_R]
    unmatched_S = [s for s in links_S if s not in matched_S]
    
    # relabel these images
    # map matched cells to low indices


    index_map_R = np.zeros((np.max(lR)+1,), dtype=np.int32)
    index_map_S = np.zeros((np.max(lS)+1,), dtype=np.int32)

    index_map_R[matched_R] = np.arange(1, len(matched_R)+1)
    index_map_S[matched_S] = np.arange(1, len(matched_R)+1)
    index_map_R[0] = 0
    index_map_S[0] = 0

    
    i = len(matched_R)+1
    for r in unmatched_R:
        if r>0:
#        print r, r in index_map_R
            if links_R[r]==0:
                index_map_R[r] = 0
            else:
                index_map_R[r] = i
                i+= 1

    i = len(matched_S)+1
    for s in unmatched_S:
        if s>0:
            index_map_S[s] = i
            i += 1

    #new_R = np.vectorize(index_map_R.get)(R)
    #new_S = np.vectorize(index_map_R.get)(S)
    new_R = index_map_R[R].reshape(R0.shape)
    new_S = index_map_S[S].reshape(S0.shape)
    

    return float(correct) / len(links2_S), new_R, new_S, len(matched_R)+1

def boundary_detection(R, S, w=None):
    if w is None:
        w = np.ones(R.shape)

    links_R, links_S = corresponding_cells(R, S)
    links_S[0] = 0
    mapped_S = np.vectorize(links_S.get)(S)

#    print 'R', R[:100]
#    print 'S', S[:100]
#    print 'links_R', links_R
#    print 'links_S', links_S
#    print 'mapped_S', mapped_S[:100]

    correct = np.sum((R == mapped_S)*w)
    return float(correct) / np.sum(w)

def test_cell_detection():
    R = np.array([3]*9+[4]*9+[5]*9+[6]*7+[7]*11)
    S = np.array([1]*9+[2]*15+[3]*3+[4]*5+[5]*7+[6]*6)

    
    print(cell_detection(R,S), 0.4)

def test_boundary_detection():
    R = np.array([1]*9+[3]*9+[2]*9+[6]*7+[9]*11)
    S = np.array([1]*9+[2]*15+[3]*3+[4]*5+[5]*7+[6]*6)
    
    print(boundary_detection(R,S), 37./45)


def test_2D():
    R = np.zeros((10,10))
    R[2:6,2:6] = 1
    R[6:9, 6:9] = 2

    
    #S = npr.randint(10, size=(10,10))
    S = np.array(R)
    
    S[S>0] += 1

#    S[0,0] = 10
    
    print('cd', cell_detection(R,S), 1.0)
    print('bd', boundary_detection(R,S), 1.0)

    print('nmi', nmi(R,S))
    print('nmi2', nmi2(R,S))

    
    
    print('nmi sklearn', normalized_mutual_info_score(R.flatten(), S.flatten()))
    print('anmi sklearn', adjusted_mutual_info_score(R.flatten(), S.flatten()))

    print('rand', rand_index(R,S))
    print('nrand', adjusted_rand_score(R.flatten(),S.flatten()))

    print('jaccard', jaccard_index(R,S))

    print('SEG', per_cell_JI(R.flatten(), S.flatten())[0])
    
    print('voi', voi(R,S))
    print('voi2', voi2(R,S))
    print('voi_different', voi_different(R,S))


def remove_outside(A, seg):
    A = np.array(A)
    l = np.unique(A)
    overlap = nd.mean(seg==0, labels=A, index=l)
    for i, v in zip(l, overlap):
        if v>0.5:
            A[A==i] = 0
    return A



def test_random():
    R = npr.randint(1, 5, 50)
    S = npr.randint(1, 5, 50)
#    R[:40] = S[:40]

    print(cell_detection(R,S), 1.0)
    print(boundary_detection(R,S), 1.0)


def test():    
    test_cell_detection()
    test_boundary_detection()
    test_2D()
    test_random()




if __name__ == "__main__":
    test()
    im1 = Image.open(sys.argv[1])
    im1 = np.asarray(im1)
    im2 = Image.open(sys.argv[2])
    im2 = np.asarray(im2)

    if len(im1.shape)>2:
        print(im1.shape)
        im1 = im1.astype(np.int32)
        im1 = im1[:,:,0] + 256*im1[:,:,1]+256*256*im1[:,:,2]
    if len(im2.shape)>2:
        print(im2.shape)
        im2 = im2.astype(np.int32)
        im2 = im2[:,:,0] + 256*im2[:,:,1]+256*256*im2[:,:,2]

    R = im1
    S = im2
        
    print('cd >', cell_detection(R,S))
    print('bd >', boundary_detection(R,S))

    print('nmi >', nmi(R,S))
    print('nmi2 >', nmi2(R,S))
    
    print('nmi sklearn >', normalized_mutual_info_score(R.flatten(), S.flatten()))
    print('anmi sklearn >', adjusted_mutual_info_score(R.flatten(), S.flatten()))

    print('rand <', rand_index(R,S))
    print('arand >', adjusted_rand_score(R.flatten(),S.flatten()))

    print('jaccard <', jaccard_index(R,S))

    seg, per_cell_JI = per_cell_JI(R.flatten(), S.flatten())

    JI_img = np.vectorize(per_cell_JI.get, otypes=[np.float64])(R)

#    plt.imshow(JI_img)
#    plt.show()
    
    print('SEG >', seg)

    
    print('voi <', voi(R,S))
    print('voi2 <' , voi2(R,S))
    print('voi_different <', voi_different(R,S))

    im3 = Image.fromarray((255.0*JI_img).astype(np.uint8))
    im3.save(sys.argv[3])
