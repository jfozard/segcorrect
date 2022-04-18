
import sys
import numpy as np
import scipy.ndimage as nd

from import_tiff import load_tiff, write_tiff

from import_image_dir import load_image_dir

from find_tiff_offset import find_offset_2d, find_offset_3d

import matplotlib.pyplot as plt

spacing = np.array([1,1,1])

cube = np.zeros((16,16,16), dtype = np.uint8)
cube[3:12,3:12,3:12] = 255
cube[3:8,3:8,3:8] = 0
cube[4:8,4:8,4:8] = 127

cube_intensity = np.zeros((16,16,16), dtype = np.uint8)
cube_intensity[11,3:12,3:12] = 255
cube_intensity[3:12,11,3:12] = 255
cube_intensity[3:12,3:12,11] = 255

write_tiff(sys.argv[1], cube.astype(np.uint8), spacing)

write_tiff(sys.argv[2], cube_intensity.astype(np.uint8), spacing)


