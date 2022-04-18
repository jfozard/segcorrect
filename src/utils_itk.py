
from __future__ import print_function

import SimpleITK as sitk
import numpy as np

def equalize_itk(stack):
    im = sitk.GetImageFromArray(stack)
    im = sitk.AdaptiveHistogramEqualization(im)
    return sitk.GetArrayFromImage(stack)

def itk_edt(mask, spacing):
    im = sitk.GetImageFromArray((~mask).astype(np.uint16))
    im.SetSpacing(spacing)
    im = sitk.SignedMaurerDistanceMap(im)
    return np.clip(sitk.GetArrayFromImage(im), 0, 255).astype(np.uint8)

def itk_blur_stack(stack, spacing, r):
    im = sitk.GetImageFromArray(stack)
    im.SetSpacing(spacing)
    im = sitk.RecursiveGaussian(im, r)
    im = sitk.RecursiveGaussian(im, r, direction=1)
    if stack.shape[0]>1:
        im = sitk.RecursiveGaussian(im, r, direction=2)
    s = sitk.GetArrayFromImage(im)
    s = np.maximum(s, 0.0)
    print(s.dtype, s.shape, stack.dtype, stack.shape)
    return s

