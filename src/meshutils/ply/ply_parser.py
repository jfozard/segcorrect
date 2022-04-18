
"""
Very simple functions to help with reading and writing
ASCII format PLY files
"""

def parse_ply(filename):
    f = open(filename, 'r')
    #
    l = f.readline()
    if l[0:3]!='ply':
        raise RuntimeError('No ply magic string at beginning of file')
    l = f.readline()
    sl = l.split()
    if sl[0]!='format' or sl[1]!='ascii':
        raise RuntimeError('File not in ascii format')
    l = f.readline()
    while l.split()[0]=='comment':
        print l[l.index(' ')+1:]
        l = f.readline()
    element_list = []
    while l.strip()!='end_header':
        el_name = l.split()[1]
        el_count = int(l.split()[2])
        property_list = []
        l = f.readline()
        sl = l.split()
        while sl[0]=='property':
            prop_name = sl[-1]
            prop_type = l.split()[1:-1]
            property_list.append((prop_name, prop_type))
            l = f.readline()
            sl = l.split()
        element_list.append((el_name, el_count, property_list))
    data = {}
    for el in element_list:
        eltype_name = el[0]
        eltype_count = el[1]
        eltype_data = []
        for i in range(el[1]):
            el_data = []
            l = f.readline()
            sl = l.split()
            buf = sl
            for prop in el[2]:
                prop_name = prop[0]
                prop_type = prop[1]
                if prop_type[0] == 'list':
                    count = int(buf[0])
                    if prop_type[2]=='float' or prop_type[2]=='double':
                        prop_data = map(float, buf[1:count+1])
                    else:
                        prop_data = map(int, buf[1:count+1])
                    buf = buf[count+1:]
                else:
                    if prop_type[0]=='float' or prop_type[0]=='double':
                        prop_data = float(buf[0])
                    else:
                        prop_data = int(buf[0])
                    buf = buf[1:]
                el_data.append(prop_data)
            eltype_data.append(tuple(el_data))
        data[eltype_name] = (eltype_data, [x[0] for x in el[2]])
    print data['face'][-1]

    print 'done_parse'
    return element_list, data


def write_ply(filename, descr, data):
    """ Write a mesh to an ascii PLY file
    """

    f = open(filename, 'w')
    f.write('ply\nformat ascii 1.0\ncomment mesh project\n')
    for e in descr:
        count = len(data[e[0]][0])
        f.write('element '+e[0]+' '+str(count)+'\n')
        for p in e[2]:
            f.write('property '+' '.join(p[1])+' '+p[0]+'\n')
    f.write('end_header\n')
    for e in descr:
        edata = data[e[0]][0]
        for el in edata:
            op = []
            for i, p in enumerate(e[2]):
                v = el[i]
                if p[1][0]!='list':
                    op.append(v)
                else:
                    op.append(len(v))
                    op.extend(v)
            f.write(' '.join(map(str, op))+'\n')
    f.close()
