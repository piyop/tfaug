# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:41:06 2020

@author: 005869
"""

import io
from PIL import Image
import numpy as np

import tensorflow as tf


tfexample_format = {"image":tf.io.FixedLenFeature([], dtype=tf.string),
                    "msk":tf.io.FixedLenFeature([], dtype=tf.string)}


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



def np_to_pngstr(npary):
    with io.BytesIO() as output:
        Image.fromarray(npary).save(output, format="PNG")
        stimg = output.getvalue()
    return stimg

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_imgs(path_imgs, mode='RGB'):
    ret_imgs = []
    for path_img in path_imgs:
        with Image.open(path_img) as im:
            ret_imgs.append(np.array(im.convert(mode)))
    return ret_imgs