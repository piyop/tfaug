# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:41:06 2020

@author: okuda
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

from tfaug import AugmentImg

def test_aug_prm(prm, name, testimg, testlbl, dstdir):    

    BATCH_SIZE = 10

    with Image.open(testimg).convert('RGB') as img:
        image = np.asarray(img)
    image = np.tile(image, (BATCH_SIZE, 1, 1, 1))

    with Image.open(testlbl).convert('RGB') as label:
        label = np.asarray(label)
    if label.data.ndim == 2:
        # if label image have no channel, add channel axis
        label = label[:, :, np.newaxis]
    label = np.tile(label, (BATCH_SIZE, 1, 1, 1))
    
    func = AugmentImg(**prm._asdict())

    img, lbl = func(image, label)

    if prm.resize and not prm.random_crop:
        assert img.shape == [BATCH_SIZE] + \
            list(prm.resize) + [3]
        assert lbl.shape == [BATCH_SIZE] + \
            list(prm.resize) + [3]
    elif prm.random_crop or prm.central_crop:
        shape = prm.central_crop or prm.random_crop
        assert img.shape == [BATCH_SIZE] + \
            list(shape) + [3]
        assert lbl.shape == [BATCH_SIZE] + \
            list(shape) + [3]
    else:
        assert img.shape == image.shape
        assert lbl.shape == label.shape

    # adjust value range to display images : canceling standardize effect.
    # this cause color change
    img = img.numpy()
    lbl = lbl.numpy()
    if prm.standardize:
        img = adjust_img_range(img)
        lbl = adjust_img_range(lbl)
    else:
        img = img.astype(np.uint8)
        lbl = lbl.astype(np.uint8)

    plot_dsresult(((img, lbl),), BATCH_SIZE,
                       1, dstdir+name+'.png', 
                       plot_label=True)


def plot_dsresult(dataset, batch_size, num_batch, path_fig, plot_label=False):

    num_row = num_batch * 2 if plot_label else num_batch
    fig, axs = plt.subplots(num_row, batch_size,
                            figsize=(batch_size, num_row), dpi=200)
    if num_row == 1:
        axs = np.expand_dims(axs, 0)
    if batch_size == 1:
        axs = np.expand_dims(axs, 1)
    fig.suptitle(os.path.basename(path_fig))
    for n_batch, (ims, lbs) in enumerate(dataset):        
        pltrow = n_batch
        if plot_label:
            pltrow = pltrow*2

        for batch in range(batch_size):
            if plot_label:
                axs[pltrow+1, batch].axis("off")
                axs[pltrow+1, batch].imshow(lbs[batch])
                
            axs[pltrow, batch].axis("off")
            axs[pltrow, batch].imshow(ims[batch])
            
                
    plt.savefig(path_fig)


def adjust_img_range(img):
    if img.dtype == np.float16:
        img = img.astype(np.float32)
    max_axis = np.max(img, axis=(1, 2, 3))[:, None, None, None]
    min_axis = np.min(img, axis=(1, 2, 3))[:, None, None, None]
    return (img - min_axis) / (max_axis - min_axis)

def _dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v


def _tensor_spec_to_dtype(v):
    return v.dtype if isinstance(v, tf.TensorSpec) else v


def new_py_function(func, inp, Tout, name=None):
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp,
                                                     expand_composites=True)
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)
    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name)
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout,
                                     expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out
