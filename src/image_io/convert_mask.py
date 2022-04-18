
import sys
import numpy as np
import scipy.ndimage as nd

from import_tiff import load_tiff, write_tiff

from import_image_dir import load_image_dir

from find_tiff_offset import find_offset_2d

# Original, uncropped stack
im_orig, orig_spacing = load_tiff(sys.argv[1])

im_crop, crop_spacing = load_tiff(sys.argv[2])

print 'im_orig im_crop spacing', orig_spacing, crop_spacing

u = find_offset_2d(im_orig, im_crop)

print 'crop offset', u

im_dir, dir_spacing = load_image_dir(sys.argv[3])

print 'spacing', orig_spacing, crop_spacing, dir_spacing

rescaling = np.array(im_orig.shape).astype(float)/np.array(im_dir.shape)

print rescaling

offset_z = 17 # 3
rescale_z = min((im_dir.shape[0]-offset_z)*2-1, im_crop.shape[0])
dir_z = (rescale_z+1)/2

start_z = 0
start_x = 4
start_y = 4

dir_x = 510
dir_y = 510
rescale_x = dir_x*2-1
rescale_y = dir_y*2-1

im_dir_clip = im_dir[offset_z:offset_z+dir_z, :dir_x, :dir_y]

x_sc = float(rescale_x)/(dir_x)
y_sc = float(rescale_y)/(dir_y)
z_sc = float(rescale_z)/(dir_z)

im_dir_rescale = np.zeros(im_orig.shape)

print im_dir_clip.shape, dir_z, rescale_z

im_dir_rescale[start_z:rescale_z+start_z, start_y:start_y+rescale_y, start_x:start_x+rescale_x] = nd.zoom(im_dir_clip, (z_sc, y_sc, x_sc), order=1)[:min(rescale_z, im_orig.shape[0]-start_z), :min(rescale_y, im_orig.shape[1]-start_y), :min(rescale_x, im_orig.shape[2]-start_x)]

# Find shift?




write_tiff(sys.argv[4], im_dir_rescale.astype(np.uint8), crop_spacing)

mask = (255*(im_dir_rescale[:im_crop.shape[0], u[0]:u[0]+im_crop.shape[1], u[1]:u[1]+im_crop.shape[2]]>0)).astype(np.uint8)

write_tiff(sys.argv[5], mask.astype(np.uint8), crop_spacing)


