

import sys
import ast
import numpy as np

def process(l):
    l = l.lstrip()
    if l[0]=='\'':
        s = l[1:].index('\'') + 2
        v = l[1:s-1]    
    else:
        try:
            s = l.index(' ')
            v = l[0:s]
        except ValueError:
            s = -1
            v = l[0:s]
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass

    if l[s:].lstrip():
        return [v] + process(l[s:])
    else:
        return [v]


def read_msr(filename):
    with open(filename, 'r') as f:
        header = []
        objects = []
        for l in f:
            u = l.split('=')
            k = u[0].strip()
            header.append((k, process(u[1])))
            if k=='OBJECTCOUNT':
                break
                nobj = int(u[1])
        for l in f:
            u = l.split('=')
            k = u[0].strip()
            if k=='OBJECT':
                obj_name = u[1][2:].strip('\'\r\n')
                obj = []
                objects.append((obj_name, obj))
            else:
                obj.append((k, process(u[1])))
                
    return dict(header), objects

def read_msr_points(filename):
    header, objects = read_msr(filename)

    print header, objects
    
#    size = np.array(header['IMAGESIZE'])
#    scale = np.array(header['SCALE'])

#    print 'size', size
#    print 'scale', scale

    point_arrays = []
    for o in objects:
        name = o[0]
        points = []
        for k, v in o[1]:
            if k=='VERT':
                points.append(v)
#        points = np.array(points)*scale + 0.5*size*scale
        points = np.array(points)
        point_arrays.append((o[0], points))
    return np.vstack([v[1] for v in point_arrays])



if __name__=='__main__':
    print read_msr_points(sys.argv[1])
