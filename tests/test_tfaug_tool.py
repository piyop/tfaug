# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:41:06 2020

@author: okuda
"""

import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


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
