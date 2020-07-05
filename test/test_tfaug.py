# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:42:29 2020

@author: 005869
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), r'src')))    
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tfaug import augment_img


def test_tfdata():
    """
    test tf.data using class augment_img

    Returns
    -------
    None.

    """    
    
    testimg='img.png'
    testlbl='lbl.png'
    
    BATCH_SIZE=10
    
    with Image.open(testimg) as img:
        image=np.asarray(img)
    image=np.tile(image, (BATCH_SIZE,1,1,1))
    
    with Image.open(testlbl) as label:
        label=np.asarray(label)
    if label.data.ndim == 2: 
        #if label image have no channel, add channel axis
        label=label[:,:,np.newaxis]
    label=np.tile(label, (BATCH_SIZE,1,1,1))

    random_zoom=.1
    random_shift=(.1,.1)
#    random_shift=None
    # random_saturation=(5,10),
    random_saturation=None
    training=True
    arg_fun=augment_img(rotation=0, 
                      standardize=False,
                      random_flip_left_right=True,
                      random_flip_up_down=True, 
                      random_shift=random_shift, 
                      random_zoom=random_zoom,
                      random_brightness=0.2,
                      random_saturation=random_saturation,
                      training=training)
    
    ds=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(image),
                          tf.data.Dataset.from_tensor_slices(label))) \
                        .shuffle(BATCH_SIZE*10).batch(BATCH_SIZE).map(arg_fun)

    #if you want adjust sampling weight on classes, use sample_from_datasets like below
    sample_weights=[1.0, 9.0]
    ds=tf.data.experimental.sample_from_datasets([ds, ds], weights=sample_weights)
    
    img, lbl=next(ds.take(1).__iter__())
    
    #adjust value range to display images : canceling standardize effect.
    img=adjust_range(img)
    lbl=adjust_range(lbl)    
    
    fig, axs=plt.subplots(BATCH_SIZE,2, figsize=(3,BATCH_SIZE), dpi=300)
    for i, (im, lb) in enumerate(zip(img, lbl)):
    
        axs[i,0].axis("off")
        axs[i,0].imshow(im)
        
        axs[i,1].axis("off")
        axs[i,1].imshow(lb)
        i += 1
    plt.savefig('test_tfdata.png')
    
    
def adjust_range(img):
    img=img.numpy()
    max_axis=np.max(img, axis=(1,2,3))[:,None,None,None]
    min_axis=np.min(img, axis=(1,2,3))[:,None,None,None] 
    return (img - min_axis) / (max_axis - min_axis)
    

def test_augmentation():
    """
    test class augment_img

    Returns
    -------
    None.

    """
    

    #image and lbl which you want to test
    testimg='img.png'
    testlbl='lbl.png'
    
    BATCH_SIZE=10
    
    with Image.open(testimg) as img:
        image=np.asarray(img)
    image=np.tile(image, (BATCH_SIZE,1,1,1))
    
    with Image.open(testlbl) as label:
        label=np.asarray(label)
    if label.data.ndim == 2: 
        #if label image have no channel, add channel axis
        label=label[:,:,np.newaxis]
    label=np.tile(label, (BATCH_SIZE,1,1,1))
    
    random_zoom=.1
    random_shift=(.1,.1)
#    random_shift=None
    training=True
    random_saturation=False
    func=augment_img(rotation=0, 
                      standardize=True,
                      random_flip_left_right=True,
                      random_flip_up_down=True, 
                      random_shift=random_shift, 
                      random_zoom=random_zoom,
                      random_brightness=0.2,
                      random_saturation=random_saturation,
                      training=training)
    
    img, lbl=func(image, label)
    
    #adjust value range to display images : canceling standardize effect.
    img=adjust_range(img)
    lbl=adjust_range(lbl)    
    
    fig, axs=plt.subplots(BATCH_SIZE,2, figsize=(3,BATCH_SIZE), dpi=300)
    for i, (im, lb) in enumerate(zip(img, lbl)):
        axs[i,0].axis("off")
        axs[i,0].imshow(im)
        
        axs[i,1].axis("off")
        axs[i,1].imshow(lb)
        i += 1
    plt.savefig('test_augmentation.png')
        
if __name__ == '__main__':
    pass
    test_augmentation()
    test_tfdata()