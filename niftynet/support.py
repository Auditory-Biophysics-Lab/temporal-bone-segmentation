#!/usr/bin/env python3

import ctypes
from functools import partial
import os
import subprocess as subp

import numpy as np
import SimpleITK as sitk

print = partial(print, flush=True)

def has_cuda():
    ## FIXME: Something more robust that nvidia-smi?
    ## Find nvidia-smi
    paths = (
        "/usr/bin/nvidia-smi",
        "/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
    )

    for p in paths:
        if os.path.isfile(p):
            try:
                subp.check_call((p,), stderr=subp.DEVNULL, stdout=subp.DEVNULL)
            except:
                return False
            else:
                return True
    return False

cudaresize = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libcudaresize.so"))
def fix_spacing(im, desired_spacing=0.15388000011444092, use_sitk=False):
    width, height, depth = im.GetSize()

    ws, hs, ds = im.GetSpacing()

    new_width, new_height, new_depth = (round(i*(j/desired_spacing)) for i,j in zip(im.GetSize(), im.GetSpacing()))

    if (width, height, depth) == (new_width, new_height, new_depth):
        print("Nothing to do")
        return im

    if not has_cuda() or use_sitk: ## Resize using SimpleITK instead
        print("Using SimpleITK for resizing, this may take a while...")
        new_size = (new_width, new_height, new_depth)
        output_spacing = tuple((i*(j/k) for i,j,k in zip(im.GetSpacing(), im.GetSize(), new_size)))
        rsf = sitk.ResampleImageFilter()
        rsf.SetDefaultPixelValue(-2000)
        rsf.SetOutputSpacing(output_spacing)
        rsf.SetInterpolator(sitk.sitkBSpline)
        rsf.SetOutputDirection(im.GetDirection())
        rsf.SetOutputOrigin(im.GetOrigin())
        rsf.SetSize(new_size)
        return rsf.Execute(im)

    print("Using CUDA for resizing")
    return resize(im, new_width, new_height, new_depth)

def resize(im, new_width, new_height, new_depth):
    width, height, depth = im.GetSize()
    ## Allocate the output array here
    output = np.zeros(new_width*new_height*new_depth, dtype=ctypes.c_float)
    output_ct = output.ctypes

    im_array = sitk.GetArrayFromImage(im)
    ## This cast is necessary
    dtype = im_array.dtype
    orig = im_array
    im_array = im_array.astype(ctypes.c_float).reshape(width*height*depth)

    im_array_ct = im_array.ctypes

    cudaresize.interpolate(im_array_ct.data_as(ctypes.POINTER(ctypes.c_float)), output_ct.data_as(ctypes.POINTER(ctypes.c_float)), width, height, depth, new_width, new_height, new_depth)
    ## Numpy indexing is reversed, we have to reshape it like this
    output = output.astype(dtype).reshape((new_depth, new_height, new_width))

    ## Finally, load it into the output image and copy over metadata
    output_im = sitk.GetImageFromArray(output)
    ## Don't set the desired spacing, set the actual spacing
    output_im.SetSpacing(tuple((i*(j/k) for i,j,k in zip(im.GetSpacing(), im.GetSize(), output_im.GetSize()))))
    output_im.SetDirection(im.GetDirection())
    output_im.SetOrigin(im.GetOrigin())

    return output_im

def island_removal(im, keep=None):
    """Run island removal.

    :params keep: A dictionary mapping label value to the number of largest islands to keep. If 
                  a label is missing, the largest island is kept.
    """
    im = sitk.Cast(im, sitk.sitkUInt8)

    #print("Unique")
    ## Figure out the unique labels. This way, we can also be sure that there will be at least one 
    ## connected component with that specific label
    #unique = np.unique(sitk.GetArrayViewFromImage(im))
    unique = [1,2,3,4,5,6,7,8,9]

    ## Edge case in case there's no background
    if unique[0] == 0:
        unique = unique[1:]

    if keep is None:
        keep = {}
        
    output = None
    ## Split the image into the labels
    for value in unique:
        value = int(value)
        print("=== VALUE",value)

        ## Do this by thresholding for the specific value we want
        print("First thres")
        filt = sitk.ThresholdImageFilter()
        filt.SetUpper(value)
        filt.SetLower(value)
        res = filt.Execute(im)

        print("Conn")
        conn_filt = sitk.ConnectedComponentImageFilter()
        res = conn_filt.Execute(res)

        print("Relabel")
        ## Next, relabel them in order of decreasing size (0 is background, 1 is largest, etc.)
        label_filt = sitk.RelabelComponentImageFilter()
        #label_filt.InPlaceOn()
        ## Ignore very small objects if necessary
        label_filt.SetMinimumObjectSize(1)
        res = label_filt.Execute(res)

        print("2nd Thres")
        ## Now remove all islands except the ones we want (usually just the biggest)
        thresh_filt = sitk.ThresholdImageFilter()
        #thresh_filt.InPlaceOn()
        thresh_filt.SetLower(1)
        ## Use the user-specified number if given, otherwise keep just the largest
        thresh_filt.SetUpper(keep.get(value, 1))
        res = thresh_filt.Execute(res)

        print("3rd Thres")
        ## Lastly, we have to fix the island labels to match the input labels. The threshold filter 
        ## lets us choose what value to assign pixels outside the range, so just set the range to [0,0]
        ## and set the pixels outside it to the appropriate label value
        thresh_filt = sitk.ThresholdImageFilter()
        #thresh_filt.InPlaceOn()
        thresh_filt.SetLower(0)
        thresh_filt.SetUpper(0)
        thresh_filt.SetOutsideValue(value)
        res = thresh_filt.Execute(res)

        print("Add")
        if output is None:
            output = res
        else:
            add_filt = sitk.AddImageFilter()
            #add_filt.InPlaceOn()
            output = add_filt.Execute(output, res)

        res = None

    return output
