# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:42:29 2020

@author: 005869
"""

import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), r'src')))    
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import unittest


import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data.experimental import AUTOTUNE

from tfaug import augment_img
import test_tfaug_tool as tool

DATADIR = r'testdata\tfaug'+os.sep


class test_tfaug(unittest.TestCase):
    
    def aug_from_filepath(self):
        
        batch_size = 2
        filepaths = [DATADIR+'Lenna.png'] * 10
        
        # define tf.data.Dataset
        ds = tf.data.Dataset.from_tensor_slices(tf.range(10)).repeat().batch(batch_size)
        
        # construct preprocessing function
        dtype = tf.int32
        class tf_img_preproc():            
            def __init__(self, filepaths):
                self.filepaths = filepaths            
            
            def preproc(self, image_nos):
                
                return tf.convert_to_tensor(tool.read_imgs([self.filepaths[no] 
                                                       for no 
                                                       in image_nos]),
                                            dtype=dtype)
                        
        preproc_obj = tf_img_preproc(filepaths)        
        func = lambda x:tf.py_function(preproc_obj.preproc, [x], dtype)
        
        # define augmentation
        aug_fun=augment_img(standardize=False,random_rotation=90,training=True)
        # map augmentation
        ds_aug = ds.map(func).map(aug_fun, num_parallel_calls=AUTOTUNE)
            
        # check augmented image
        fig, axs=plt.subplots(batch_size, 10, figsize=(10, batch_size), dpi=300)
        for i, imgs in enumerate(iter(ds_aug.take(10))):            
            axs[0,i].axis("off")
            axs[0,i].imshow(imgs[0])
            axs[1,i].axis("off")
            axs[1,i].imshow(imgs[1])
            
        plt.savefig(DATADIR+'aug_from_filepath.png')
            
            
    def aug_from_tfrecord(self):
        
        batch_size = 2
        
        #test file
        filepaths = [DATADIR+'Lenna.png'] * 10
        
        #function for generate tfExample      
        def image_example(iimg, imsk):                    
            feature = {'image': tool._bytes_feature(tool.np_to_pngstr(iimg)),
                       'msk': tool._bytes_feature(tool.np_to_pngstr(imsk))}            
            return tf.train.Example(features=tf.train.Features(feature=feature))
        
        path_tfrecord = DATADIR+r'sample.tfrecords'
        #save tfrecord
        with tf.io.TFRecordWriter(path_tfrecord) as writer:
            for filepath in filepaths:                      
                img = np.array(Image.open(filepath).convert('RGB') )     
                #use same image as msk                
                writer.write(image_example(img, img).SerializeToString())
    
        # construct preprocessing function
        dtype = tf.uint8
               
        def preproc(tfexamples):                
            return (tf.map_fn(tf.image.decode_png,tfexamples['image'], dtype=dtype),
                    tf.map_fn(tf.image.decode_png,tfexamples['msk'], dtype=dtype))
                                   
        # define augmentation
        aug_fun=augment_img(standardize=False,random_zoom=[0.2,0.8],training=True)
        
        #define dataset
        ds_aug = (tf.data.TFRecordDataset([path_tfrecord]).repeat().batch(batch_size)
                      .apply(tf.data.experimental.parse_example_dataset(tool.tfexample_format))
                      .map(preproc,num_parallel_calls=AUTOTUNE)
                      .map(aug_fun,num_parallel_calls=AUTOTUNE))
        
        # check augmented image
        fig, axs=plt.subplots(batch_size*2, 10, figsize=(10, batch_size*2), dpi=300)
        for i, (imgs, msks) in enumerate(iter(ds_aug.take(10))):       
            for row in range(batch_size):
                axs[row*2,i].axis("off")
                axs[row*2,i].imshow(imgs[row])
                axs[row*2+1,i].axis("off")
                axs[row*2+1,i].imshow(msks[row])
            
        plt.savefig(DATADIR+'aug_from_tfrecord.png')
        
        # to learn a model
        # model.fit(ds_aug)
        
    
    def aug_multi_input_and_pyfunc(self):
        
        batch_size = 1
        filepaths = [DATADIR+'Lenna.png'] * 10
        
        # define tf.data.Dataset
        ds = tf.data.Dataset.from_tensor_slices(tf.range(10)).repeat().batch(batch_size)
        
        # define augmentation
        aug_fun=augment_img(standardize=False,random_rotation=90,training=True)
        
        # construct preprocessing function
        dtype = tf.int32
        # class tf_img_preproc():            
        #     def __init__(self, filepaths1, filepaths2, aug_func):
        #         self.filepaths1 = filepaths1           
        #         self.filepaths2 = filepaths2            
        #         self.aug_func = aug_func
            
        def tf_img_preproc(filepaths1, filepaths2, aug_func):
            def preproc(self, image_nos):
                
                #augment only image here
                return (aug_func(tf.convert_to_tensor(tool.read_imgs([filepaths1[no] 
                                                       for no 
                                                       in image_nos]),
                                            dtype=dtype)),
                        tf.convert_to_tensor(tool.read_imgs([filepaths2[no] 
                                                                   for no 
                                                                   in image_nos]),
                                            dtype=dtype))
            return preproc
            
        #pass aug_fun to 3rd arg
        func = tf_img_preproc(filepaths, filepaths.copy(), aug_fun)    
        #py_function for multiple output
        func = lambda x:tool.new_py_function(func, [x], {'image1':dtype,'image2':dtype})
        
        # map preprocess and augmentation
        ds_aug = ds.map(func, num_parallel_calls=AUTOTUNE)
            
        # check augmented image
        fig, axs=plt.subplots(batch_size*2, 10, figsize=(10, batch_size*2), dpi=300)
        for i, in_dict in enumerate(iter(ds_aug.take(10))):
            for row in range(batch_size):
                axs[row*2,i].axis("off")
                axs[row*2,i].imshow(in_dict['image1'][row])
                axs[row*2+1,i].axis("off")
                axs[row*2+1,i].imshow(in_dict['image2'][row])
            
        plt.savefig(DATADIR+'aug_multi_input_and_pyfunc.png')
            
        
        # to learn a model
        # model.fit(ds_aug)
        
        
    
    def test_tfdata_vertual(self):
        
        BATCH_SIZE=10
        image = np.arange(5**3).reshape(5,5,5)
        image=np.tile(image,(BATCH_SIZE,1,1,1))
        
        random_zoom=(.1,.1)
        random_shift=(.1,.1)
        random_saturation=None
        training=True
        aug_fun=augment_img(
                          standardize=False,
                          random_flip_left_right=True,
                          random_flip_up_down=True, 
                          random_shift=random_shift, 
                          random_zoom=random_zoom,
                          random_brightness=0.2,
                          random_saturation=random_saturation,
                          training=training)
        
        image = image.astype(np.float32)
        
        def py_function(x):
            tf.print('py_fuction',output_stream=sys.stderr)
            tf.print(x.shape,output_stream=sys.stderr)
            return x
        
        def aug_fun(x):
            tf.print('aug_fun',output_stream=sys.stderr)
            tf.print(x.shape,output_stream=sys.stderr)
            return x
        
        func = lambda x:tf.py_function(py_function, [x], tf.float32)
        
        ds = tf.data.Dataset.from_tensors(image).map(func).map(aug_fun)                      
        
        tf.print('get data')
        img=next(ds.take(1).__iter__())
        # print(img.shape)
        
        
    def adjust_img_range(self, img):
        max_axis=np.max(img, axis=(1,2,3))[:,None,None,None]
        min_axis=np.min(img, axis=(1,2,3))[:,None,None,None] 
        return (img - min_axis) / (max_axis - min_axis)
    
    
        
    
    def test_augmentation(self):
        """
        test class augment_img
    
        Returns
        -------
        None.
    
        """
        params = namedtuple('params', '''
                                  standardize,
                                  random_rotation, 
                                  random_flip_left_right,
                                  random_flip_up_down, 
                                  random_shift, 
                                  random_zoom,
                                  random_shear,
                                  random_brightness,
                                  random_saturation,
                                  random_hue,
                                  random_contrast,
                                  random_crop,
                                  interpolation
                                  training''')
    
                
        #image and lbl which you want to test
        testimg=DATADIR+'Lenna.png'
        testlbl=DATADIR+'Lenna.png'
        
        BATCH_SIZE=10
        
        with Image.open(testimg).convert('RGB') as img:
            image=np.asarray(img)
        image=np.tile(image, (BATCH_SIZE,1,1,1))
        
        with Image.open(testlbl).convert('RGB') as label:
            label=np.asarray(label)
        if label.data.ndim == 2: 
            #if label image have no channel, add channel axis
            label=label[:,:,np.newaxis]
        label=np.tile(label, (BATCH_SIZE,1,1,1))
        
        cases = [
                params(#test y_shift
                standardize=True,           
                random_rotation=0, 
                random_flip_left_right=True,
                random_flip_up_down=False, 
                random_shift=[256,0], 
                random_zoom=None,
                random_shear=None,
                random_brightness=False,
                random_saturation=None,
                random_hue=None,
                random_contrast=None,
                random_crop = None,
                interpolation = 'nearest',
                training=True),
                
                params(#test x_shift and rotation
                standardize=True,                
                random_rotation=45, 
                random_flip_left_right=False,
                random_flip_up_down=True, 
                random_shift=(0,256), 
                random_zoom=None,
                random_shear=None,
                random_brightness=False,
                random_saturation=(5, 10),
                random_hue=None,
                random_contrast=None,
                random_crop = None,
                interpolation = 'nearest',
                training=True),
                
                params(#test crop and zoom
                standardize=True,
                random_rotation=45, 
                random_flip_left_right=False,
                random_flip_up_down=False, 
                random_shift=None, 
                random_zoom=(0.8, 0.1),
                random_shear=None,
                random_brightness=None,
                random_saturation=None,
                random_hue=None,
                random_contrast=None,
                random_crop = (256, 512),
                interpolation = 'bilinear',
                training=True),
                
                params(#test shear and color
                standardize=True,
                random_rotation=None, 
                random_flip_left_right=False,
                random_flip_up_down=False, 
                random_shift=None, 
                random_zoom=None,
                random_shear=(10,10),
                random_brightness=0.5,
                random_saturation=[5,10],
                random_hue=0.3,
                random_contrast=[.1,.5],
                random_crop = None,
                interpolation = 'bilinear',
                training=True),
                
                params(#test train = False
                standardize=True,
                random_rotation=45, 
                random_flip_left_right=True,
                random_flip_up_down=True, 
                random_shift=None, 
                random_zoom=(0.8, 0.1),
                random_shear=(50,50),
                random_brightness=0.5,
                random_saturation=[5,10],
                random_hue=0.3,
                random_contrast=[.1,.5],
                random_crop = (256, 256),
                interpolation = 'nearest',
                training=False),
                 ]
        
        
        for no, case in enumerate(cases):
            with self.subTest(case=case):
                
                func=augment_img(*case)
                
                img, lbl=func(image, label)
                
                if case.random_crop:
                    assert img.shape == [BATCH_SIZE] + list(case.random_crop) + [3]
                    assert lbl.shape == [BATCH_SIZE] + list(case.random_crop) + [3]
                else:
                    assert img.shape == image.shape
                    assert lbl.shape == label.shape
                    
                
                # plt.imsave('imgraw_'+case.interpolation+'.png', img[0].numpy())
                
                #adjust value range to display images : canceling standardize effect.
                img=self.adjust_img_range(img.numpy())
                lbl=self.adjust_img_range(lbl.numpy())    
                
                fig, axs=plt.subplots(BATCH_SIZE,2, figsize=(3,BATCH_SIZE), dpi=300)
                for i, (im, lb) in enumerate(zip(img, lbl)):
                    axs[i,0].axis("off")
                    axs[i,0].imshow(im.squeeze())
                    
                    axs[i,1].axis("off")
                    axs[i,1].imshow(lb.squeeze())
                    i += 1
                plt.savefig(DATADIR+'test_augmentation_caseno'+str(no)+'.png')
        
        
    def test_valid(self):
                
        #image and lbl which you want to test
        testimg=DATADIR+'Lenna.png'
        testlbl=DATADIR+'Lenna.png'
        
        BATCH_SIZE=10
        
        with Image.open(testimg).convert('RGB') as img:
            image=np.asarray(img)
        image=np.tile(image, (BATCH_SIZE,1,1,1))
        
        with Image.open(testlbl).convert('RGB') as label:
            label=np.asarray(label)
        if label.data.ndim == 2: 
            #if label image have no channel, add channel axis
            label=label[:,:,np.newaxis]
        label=np.tile(label, (BATCH_SIZE,1,1,1))   
        
        training = False
        
        func=augment_img(standardize=False,
                          random_flip_left_right=True,
                          random_flip_up_down=True, 
                          random_shift=(0.1,0.1), 
                          random_zoom=(0.1, 0.1),
                          random_brightness=False,
                          random_saturation=False,
                          random_crop = [256, 128],
                          training=training)
        
        img, lbl=func(image, label)
        lbl_offset_y = (label.shape[1] - 256) // 2
        lbl_offset_x = (label.shape[1] - 128) // 2
                
        self.assertEqual(img.shape, (10, 256, 128, 3))
        self.assertEqual(lbl.shape, (10, 256, 128, 3))
        
        self.assertTrue(np.allclose(lbl.numpy(),
                               label[:,lbl_offset_y:lbl_offset_y+256,lbl_offset_x:lbl_offset_x+128,:]))
        self.assertTrue(np.allclose(img.numpy(),
                               image[:,lbl_offset_y:lbl_offset_y+256,lbl_offset_x:lbl_offset_x+128,:]))
        
        
    """
    below code is each transformation test
    """
    def batch_transform(self):
        
        # interpolation='bilinear'
        BATCH_SIZE = 10
        interpolation='nearest'
        
        testimg=DATADIR+'Lenna.png'
        with Image.open(testimg).convert('RGB') as img:
            image=np.asarray(img)
        image=np.tile(image, (BATCH_SIZE,1,1,1))
                    
        batch_size, size_y, size_x, depth = tf.shape(image)
        size_y, size_x = tf.cast((size_y, size_x), dtype=tf.float32)
            

        trans_matrix = tf.eye(3, 3, [tf.shape(image)[0]], dtype=tf.float32)
        
        shift_y = tf.zeros([batch_size], dtype=tf.float32)
        shift_x = tf.zeros([batch_size], dtype=tf.float32)        
        shear_y = tf.zeros([batch_size], dtype=tf.float32)
        shear_x = tf.zeros([batch_size], dtype=tf.float32)
        zoom_y = tf.zeros([batch_size], dtype=tf.float32)
        zoom_x = tf.zeros([batch_size], dtype=tf.float32)
        
        shift_size = np.array([0, 0], dtype=np.float32)
        if shift_size is not None:            
            shift_y += tf.random.uniform([batch_size], -shift_size[0], shift_size[0])
            shift_x += tf.random.uniform([batch_size], -shift_size[1], shift_size[1])
            
        shear_theta = np.array([0, 0], dtype=np.float32)
        if shear_theta is not None:
            shear_tan = tf.tan(shear_theta / 180 * math.pi)
            shear_y += tf.random.uniform([batch_size], -shear_tan[0], shear_tan[0])
            shear_x += tf.random.uniform([batch_size], -shear_tan[1], shear_tan[1])
                 
            shift_y += -(size_y * shear_y) / 2
            shift_x += -(size_x * shear_x) / 2            
        
        
        zoom = np.array([0, 0], dtype=np.float32)
        if zoom is not None:
            zoom_y = tf.random.uniform([batch_size], -zoom[0], zoom[0])
            zoom_x = tf.random.uniform([batch_size], -zoom[1], zoom[1])
            
            shift_y += -(size_y * zoom_y) / 2
            shift_x += -(size_x * zoom_x) / 2            
            
        
        trans_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                                    [[x[5]+1, x[3], x[1]],
                                    [x[2], x[4]+1, x[0]],
                                    [0,0,1]], tf.float32),        
                          tf.transpose([shift_y, shift_x, shear_y, shear_x, zoom_y, zoom_x]))
            
        rot_theta = 90
        if rot_theta is not None:
            rad_theta = rot_theta / 180 * math.pi
            rot = tf.random.uniform([batch_size], -rad_theta, rad_theta)
            h11 = tf.cos(rot)
            h12 = -tf.sin(rot)
            h21 = tf.sin(rot)
            h22 = tf.cos(rot)
            shift_rot_y = ( (size_y - size_y * tf.cos(rot)) - (size_x * tf.sin(rot)) ) / 2
            shift_rot_x =( (size_x - size_x * tf.cos(rot)) + (size_y * tf.sin(rot)) ) / 2
            
            rot_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                                    [[x[0], x[1], x[5]],
                                    [x[2], x[3], x[4]],
                                    [0,0,1]], tf.float32),        
                          tf.transpose([h11, h12, h21, h22, shift_rot_y, shift_rot_x]))
            
            trans_matrix = tf.keras.backend.batch_dot(trans_matrix, rot_matrix)
                                
        #get matrix
        M = tfa.image.transform_ops.matrices_to_flat_transforms(tf.linalg.inv(trans_matrix))
        
        #execute
        retimg = tfa.image.transform(image,M,interpolation=interpolation)     
        
        #crop
        random_crop = (256, 512)
        retimg = tf.image.random_crop(image, 
                                      size=tf.concat((tf.expand_dims(batch_size,-1),
                                                      random_crop,
                                                      tf.expand_dims(depth,-1)),axis=0))

        fig, axs=plt.subplots(BATCH_SIZE,1, figsize=(3,BATCH_SIZE), dpi=300)
        for i, im in enumerate(retimg):
            axs[i].axis("off")
            axs[i].imshow(np.squeeze(im.numpy()))
            
        plt.savefig(DATADIR+'test_batch_transform.png')
        
        
    def single_transform(self):
        
        interpolation='nearest'
        
        testimg=DATADIR+'Lenna.png'
        with Image.open(testimg).convert('RGB') as img:
            image=np.asarray(img)
        size_y, size_x = image.shape[:2]
            

        trans_matrix = tf.eye(3, 3, dtype=tf.float32)
        
        shift_ratio = np.array([0, 0])
        if shift_ratio is not None:            
            shift_val = image.shape[:2]  * shift_ratio
            trans_matrix += np.array([[0, 0, shift_val[1]],
                                    [0, 0, shift_val[0]],
                                    [0,0,0]], np.float32)
            
        shear_theta = np.array([0, 0])
        if shear_theta is not None:
            shear_rad = shear_theta / 180 * math.pi
            shift_shear = -(image.shape[:2] * np.tan(shear_rad)) / 2
            
            trans_matrix += np.array([[0, math.tan(shear_rad[1]), shift_shear[1]],
                                              [math.tan(shear_rad[0]), 0, shift_shear[0]],
                                              [0,0,0]], np.float32)
        
        
        zoom = np.array([0, 0])
        if zoom is not None:
            shift_zoom = -(image.shape[:2] * zoom) / 2
            trans_matrix += np.array([[zoom[1], 0, shift_zoom[1]],
                                    [0, zoom[0], shift_zoom[0]],
                                    [0,0,0]], np.float32)
            
            
        rot_theta = 0
        rad_theta = rot_theta / 180 * math.pi
        if rot_theta is not None:
            shift_rot_y = ( (size_y - size_y * math.cos(rad_theta)) - (size_x * math.sin(rad_theta)) ) / 2
            shift_rot_x =( (size_x - size_x * math.cos(rad_theta)) + (size_y * math.sin(rad_theta)) ) / 2
            
            trans_matrix = tf.tensordot(trans_matrix, np.array([[math.cos(rad_theta), -math.sin(rad_theta), shift_rot_x],
                                              [math.sin(rad_theta), math.cos(rad_theta), shift_rot_y],
                                              [0,0,1]], np.float32),axes = [[1], [0]])
                                
        #get matrix
        M = tfa.image.transform_ops.matrices_to_flat_transforms(tf.linalg.inv(trans_matrix))
        
        #transform
        retimg = tfa.image.transform(image,M,interpolation=interpolation)    
        plt.imshow(retimg.numpy())
        
        """color change"""
        # random_brightness = 0.1
        random_brightness = None
        # random_saturation = (0.5,2)
        random_saturation = None
        # random_hue = .03
        random_hue = None
        # random_contrast = (0.3, 0.5)
        random_contrast = None
        if random_brightness:
            retimg = tf.image.random_brightness(image, random_brightness)
        if random_saturation:
            retimg = tf.image.random_saturation(image, *random_saturation)
        if random_hue:
            retimg = tf.image.random_hue(image, random_hue)
        if random_contrast:
            retimg = tf.image.random_contrast(image, *random_contrast)
        
        
        plt.imshow(retimg.numpy())
        
        
if __name__ == '__main__':
    pass
    unittest.main()
    # obj = test_tfaug()
    # obj.test_augmentation()