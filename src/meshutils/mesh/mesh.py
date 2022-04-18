

import numpy as np
import numpy.random as npr

from meshutils.ply.ply_parser import *

from scipy.sparse import coo_matrix

def calculate_vertex_normals(verts, tris):
    v_array = np.asarray(verts)
    tri_array = np.asarray(tris, dtype=int)
    tri_pts = v_array[tri_array]
    n = np.cross( tri_pts[:,1] - tri_pts[:,0], 
                  tri_pts[:,2] - tri_pts[:,0])

#    K = n[:,0]**2 + n[:,1]**2 + n[:,2]**2
#    print np.where(K < 1e-12)

    v_normals = np.zeros(v_array.shape)

    

    for i in range(tri_array.shape[0]):
#        for j in tris[i]:
            v_normals[tris[i],:] += n[np.newaxis, i, :]

#    print n.shape, tri_array.shape, n[tri_array,:].shape
#    v_normals = np.sum(n[tri_array,:], axis=1)
#    print v_normals.shape, v_array.shape


    nrms = np.sqrt(v_normals[:,0]**2 + v_normals[:,1]**2 + v_normals[:,2]**2)

#    print np.where(nrms < 1e-12)
    
    v_normals = v_normals / nrms.reshape((-1,1))

    return v_normals



class Mesh(object):
    """
    Very simple triangulated mesh class
    Stored as triangle soup
    """

    def __init__(self):
        self.verts = np.zeros((0,3), dtype=float)                 # List of vertex positions
        self.tris = np.zeros((0,3), dtype=int)                  # List of triangle vertex indices
        self.vert_props = {}            # Dictionary of arrays - per vertex properties
        self.tri_props = {}             # Dictionary of arrays - per face properties

    def load_ply(self, fn):
        descr, data = parse_ply(fn)
        #self.descr = descr
        #self.data = data
        
        print descr
#        print data['face'][0]
#        print data['face'][1]

        NV = len(data['vertex'][0])
        NF = len(data['face'][0])
        print 'NF', NF

        verts = []

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
            vert_norm = []
        else:
            has_normal = False

        if 'signal' in data['vertex'][1]:
            s_idx = data['vertex'][1].index('signal')
            has_signal = True
            vert_signal = []
        else:
            has_signal = False
        
        if 'red' in data['vertex'][1]:
            r_idx = data['vertex'][1].index('red')
            b_idx = data['vertex'][1].index('red')
            g_idx = data['vertex'][1].index('green')
            vert_color = []
            has_color = True
        else:
            has_color = False

        if 'label' in data['vertex'][1]:
            l_idx = data['vertex'][1].index('label')
            vert_label = []
            has_label = True
        elif 'state' in data['vertex'][1]:
            l_idx = data['vertex'][1].index('state')
            vert_label = []
            has_label = True
        else:
            has_label = False

        for v in data['vertex'][0]:
            verts.append((v[x_idx], v[y_idx], v[z_idx]))
            if has_normal:
                vert_norm.append(np.array((v[nx_idx], v[ny_idx], v[nz_idx])))
            if has_label:
                vert_label.append(int(v[l_idx]))
            if has_signal:
                vert_signal.append(v[s_idx])
            if has_color:
                vert_color.append(np.array((v[r_idx], v[g_idx], v[b_idx])))
 
        print 'done_vertex'

        self.verts = np.array(verts)

        tris = []
        for f in data['face'][0]:
            vv = f[0]
            tt = []
            for i in range(len(vv)-2):
                tt.append((vv[0], vv[i+1], vv[i+2]))

            tris.extend(tt)

        self.tris = np.array(tris)

        if has_normal:
            self.vert_props['normal'] = np.array(vert_norm)
        else:
            print 'Calculate surface normals'
            # Area weighted surface normals (would prefer angle-weighted)
            self.vert_props['normal'] = calculate_vertex_normals(self.verts, self.tris)
        
        if has_signal:
            self.vert_props['signal'] = np.array(vert_signal)

        if has_color:
            self.vert_props['color'] = np.array(vert_color)

        if has_label:
            self.vert_props['label'] = np.array(vert_label)


    def save_ply(self, filename):

        descr = [('vertex', None, [('x', ['float']),
                                   ('y', ['float']),
                                   ('z', ['float'])]),
                 ('face', None, [('vertex_index', 
                                  ['list', 'int', 'int'])])]

        vp_list = ['x', 'y', 'z']
        fp_list = ['vertex_index']
        v_data = []
        f_data = []
        
        if 'normal' in self.vert_props:
            descr[0][2].extend([('nx', ['float']),
                                ('ny', ['float']),
                                ('nz', ['float'])])
            vp_list.extend(['nx', 'ny', 'nz'])
            normal = self.vert_props['normal']
            has_normal = True
        else:
            has_normal = False
        
        if 'color' in self.vert_props:
            descr[0][2].extend([('red', ['uchar']),
                                ('green', ['uchar']),
                                ('blue', ['uchar'])])
            vp_list.extend(['red', 'green', 'blue'])
            color = self.vert_props['color']
            has_color = True
        else:
            has_color = False


        has_signal = []
        for sn in self.vert_props:
            if 'signal' in sn:
                descr[0][2].append((sn, ['float']))
                vp_list.extend(sn)
                has_signal.append(sn)

        if 'label' in self.vert_props:
            descr[0][2].append(('label', ['int']))
            vp_list.extend('label')
            label = self.vert_props['label']
            has_label = True
        else:
            has_label = False


        if 'label' in self.tri_props:
            descr[1][2].extend((['label', ['int']],)) # ['red',['uchar']],['green',['uchar']],['blue',['uchar']]   ))
            tri_label = self.tri_props['label']
            fp_list.extend(['label']) #,'red','green','blue'])
#            nl = np.max(tri_label)+1
#            red = npr.randint(255, size=(nl,))
#            green =  npr.randint(255, size=(nl,))
#            blue = npr.randint(255, size=(nl,))
            has_tri_label = True
        else:
            has_tri_label = False

        for i in range(self.verts.shape[0]):
            v = self.verts[i,:].tolist()
            if has_normal:
                v.extend(normal[i,:])
            if has_color:
                v.extend(color[i,:])
            for sn in has_signal:
                v.append(self.vert_props[sn][i])
            if has_label:
                v.append(label[i])
            v_data.append(v)

    

        for i in range(self.tris.shape[0]):
            t = [self.tris[i,:].tolist()]
            if has_tri_label:
                t.append(tri_label[i])
#                t.append(red[tri_label[i]])
#                t.append(green[tri_label[i]])
#                t.append(blue[tri_label[i]])
            f_data.append(t)
            

        data = { 'vertex': (v_data, vp_list),
                 'face': (f_data, fp_list) }
            
        write_ply(filename, descr, data)



            


        
