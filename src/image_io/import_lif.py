
"""
Load TIFF stack (exported from lif microscopy file by imagej)
"""

import sys, os
import numpy as np
import javabridge
import bioformats as bf
from xml import etree as et
import matplotlib.pyplot as plt

init_javabridge = False

def get_number(max_num):
    """Input integer from console, in range [0, (max-num-1)]"""
    return int(raw_input("select stack> "))



def parse_xml_metadata(xml_string, array_order='zyx'):
    """Get interesting metadata from the LIF file XML string.
    Parameters
    ----------
    xml_string : string
        The string containing the XML data.
    array_order : string
        The order of the dimensions in the multidimensional array.
        Valid orders are a permutation of "tzyxc" for time, the three
        spatial dimensions, and channels.
    Returns
    -------
    names : list of string
        The name of each image series.
    sizes : list of tuple of int
        The dimensions of the image in the specified order of each image.
    resolutions : list of tuple of float
        The resolution of each series in the order given by
        `array_order`. Time and channel dimensions are ignored.
    """
    array_order = array_order.upper()
    names, sizes, resolutions = [], [], []
    spatial_array_order = [c for c in array_order if c in 'XYZ']
    size_tags = ['Size' + c for c in array_order]
    res_tags = ['PhysicalSize' + c for c in spatial_array_order]
    metadata_root = et.ElementTree.fromstring(xml_string)
    for child in metadata_root:
        if child.tag.endswith('Image'):
            names.append(child.attrib['Name'])
            for grandchild in child:
                if grandchild.tag.endswith('Pixels'):
                    att = grandchild.attrib
                    sizes.append(tuple([int(att[t]) for t in size_tags]))
                    resolutions.append(tuple([float(att[t]) for t in res_tags]))
    return names, sizes, resolutions


def load_lif_stack(filename, im_num=None, verbose=True):
    """Load 3D TIFF from file

    Args:
        filename (string): Path of image file
        im_num (int): Im_Num of stack in lif file.

    Returns:
        array (numpy.ndarray): Array of image data
        spacing (tuple): Voxel spacing (in microns) in the i, j, k directions.
    """

    global init_javabridge
    if not init_javabridge:
        javabridge.start_vm(class_path=bf.JARS, run_headless=True)
        init_javabridge = True

    md = bf.get_omexml_metadata(filename)
    mdo = bf.OMEXML(md)
    rdr = bf.ImageReader(filename, perform_init=True)

    names, sizes, resolutions = parse_xml_metadata(md)
        
    if verbose or im_num is None:
        print '%s contains:' % filename
        for i in range(mdo.image_count):
            print '%d, %s, %s' % (i,names[i],(sizes[i],))
            
    	if im_num is None:
		im_num = get_number(mdo.image_count)

    if verbose:
	print 'Importing image with im_num ', im_num
        print 'Image size', sizes[im_num]
        print 'Image resolution ', resolutions[im_num]

    im_size = sizes[im_num]
    z_size = im_size[0]
    image3d = np.zeros(im_size, np.uint8)
    spacing = resolutions[im_num]

    for z in range(z_size):
        image3d[z, :, :] = rdr.read(z=z, series=im_num, rescale=False)[:,:,0]

        
        
	
    return image3d, spacing


def get_lif_info(filename):
    """Load lif info
    Args:
        filename (string): Path of image file

    Returns:
    """

    global init_javabridge
    if not init_javabridge:
        javabridge.start_vm(class_path=bf.JARS, run_headless=True)
        init_javabridge = True

    md = bf.get_omexml_metadata(filename)
    mdo = bf.OMEXML(md)
    rdr = bf.ImageReader(filename, perform_init=True)

    names, sizes, resolutions = parse_xml_metadata(md)
    return names, sizes, resolutions


def stop_javabridge():
    global init_javabridge
    if init_javabridge:
        javabridge.kill_vm()
        init_javabridge = False





if __name__=="__main__":
    image3d, spacing = load_lif_stack(sys.argv[1])
    plt.figure()
    plt.imshow(np.max(image3d, axis=0))
    plt.show()
    stop_javabridge()
