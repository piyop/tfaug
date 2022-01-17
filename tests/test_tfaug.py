# -*- coding: utf-8 -*-
"""
@license: MIT
@author: t.okuda
"""

import unittest
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from glob import glob

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow_addons as tfa
import tensorflow as tf


import test_tfaug_tool as tool
from tfaug import AugmentImg, DatasetCreator, TfrecordConverter
from test_aug_types import TestAugTypes


DATADIR = 'testdata/'

    
class TestTfrecordConverter(unittest.TestCase):
    
    
    def test_get_patch(self):
        im = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)

        patches = TfrecordConverter().split_to_patch(im, [1, 2], [1, 1])

        assert patches.shape == (4*3, 3, 4, 3), "invalid patch shape"
        tmp = np.zeros((3, 4, 3))
        tmp[1:, 1:, :] = im[0:2, 0:3, :]
        assert (patches[0, :, :, :] == tmp).all(), "invalid patch values"
        tmp = np.zeros((3, 4, 3))
        tmp[0:2, 0:2, :] = im[-2:, -2:, :]
        assert (patches[-1, :, :, :] == tmp).all(), "invalid patch values"
        
    def test_concat_patch(self):
        
        x_size, y_size = 20, 16
        patch_size = 3,5
        img = np.arange(y_size*x_size).reshape(y_size,x_size,1)
        splitted = TfrecordConverter().split_to_patch(img, patch_size, 0)
        
        cy_size = -(-y_size//patch_size[0])
        cx_size = -(-x_size//patch_size[1])
        
        ret = TfrecordConverter().concat_patch(splitted, cy_size, cx_size)        
        
        assert (img == ret[:img.shape[0],:img.shape[1]]).all(), 'value is changed'



class TestDatasetCreator(unittest.TestCase):
    
    def test_set_inputs_shapes(self):
        
        BATCH_SIZE = 2
        flist_imgs = [DATADIR+'Lenna.png'] * 10
        flist_imgs_small = [DATADIR+'Lenna_crop.png'] * 10
        clslabels = list(range(10))        
        
        dc = DatasetCreator(False, BATCH_SIZE, training=True)
        
        ds1 = DatasetCreator(1, BATCH_SIZE).from_path(flist_imgs)
        ds2 = DatasetCreator(1, BATCH_SIZE).from_path(flist_imgs_small)
        lbl = DatasetCreator(1, BATCH_SIZE).from_path(flist_imgs)
        lbl_cls = tf.data.Dataset.from_tensor_slices(clslabels).batch(BATCH_SIZE)       
        
        # segmentation
        test_ds1 = tf.data.Dataset.zip((ds1, lbl))
        
        shapes_in, shapes_lbl = dc._get_inputs_shapes(test_ds1, 'segmentation')
        
        assert shapes_in == [(BATCH_SIZE, 512,512,3)], 'invalid shape'
        assert shapes_lbl == (BATCH_SIZE, 512,512,3), 'invalid label shape'
        
        # classification
        test_ds2 = tf.data.Dataset.zip((ds1, ds2, lbl_cls))
        shapes_in, shapes_lbl = dc._get_inputs_shapes(test_ds2, 'class')
                
        assert shapes_in[0] == (BATCH_SIZE, 512,512,3)
        assert shapes_in[1] == (BATCH_SIZE, 256,512,3)
        assert shapes_lbl == (BATCH_SIZE)
        
    
    def test_tf_function(self):

        BATCH_SIZE = 5
        # data augmentation configurations:
        DATAGEN_CONF = {'standardize': False,
                        'resize': [100,100],
                        # 'random_rotation': 5,
                        'random_flip_left_right': True,
                        'random_flip_up_down': False,
                        # 'random_shift': [25, 25],
                        'random_zoom': [0.2, 0.2],
                        # 'random_shear': [5, 5],
                        # 'random_brightness': 0.2,
                        # 'random_hue': 0.00001,
                        'random_contrast': [0.6, 1.4],
                        'random_crop': None,  # what to set random_crop
                        'random_noise': 5,
                        # 'random_saturation': [0.5, 1.5],
                        'input_shape':[BATCH_SIZE, 512, 512, 3],
                        'num_transforms':10}

        flist = [DATADIR+'Lenna.png'] * 10 * BATCH_SIZE
        # test for ratio_samples
        labels = [0] * 10 * BATCH_SIZE
        
        dc = DatasetCreator(BATCH_SIZE,
                            BATCH_SIZE,
                            **DATAGEN_CONF,
                            training=True)
        ds = dc.from_path(flist, labels)            
                    
        taked = iter(ds.take(10))
        
        @tf.function
        def one_step():
            img, lbl = next(taked)
            # print('imgshape:', img.shape, 'lblshape:', lbl.shape)
            return img, lbl

        img, lbl = one_step()        
        assert img.shape[0] == BATCH_SIZE, 'invalid batch size'
        
        zipped = zip(img, lbl)
        piyo0, piyo1 = next(zipped)

        tool.plot_dsresult(((img, lbl),), BATCH_SIZE, 1,
                           DATADIR+'test_tf_function.png')
        
    def test_from_path(self):

        # data augmentation configurations:
        DATAGEN_CONF = {'standardize': True,
                        'resize': None,
                        'random_rotation': 5,
                        'random_flip_left_right': True,
                        'random_flip_up_down': False,
                        'random_shift': [.1, .1],
                        'random_zoom': [0.2, 0.2],
                        'random_shear': [5, 5],
                        'random_brightness': 0.2,
                        'random_hue': 0.01,
                        'random_contrast': [0.6, 1.4],
                        'random_crop': None,  # what to set random_crop
                        'random_noise': 100,
                        'random_saturation': [0.5, 2],
                        'num_transforms':10}

        BATCH_SIZE = 2
        flist = [DATADIR+'Lenna.png'] * 10 * BATCH_SIZE
        labels = [0] * 10 * BATCH_SIZE

        ds = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE, label_type='class',
                            **DATAGEN_CONF,
                            training=True).\
            from_path(flist, labels)

        tool.plot_dsresult(ds.take(10), BATCH_SIZE, 10,
                           DATADIR+'test_from_path.png')

    def test_from_tfrecord(self):

        random_crop_size = [100, 254]
        # data augmentation configurations:
        DATAGEN_CONF = {'standardize': True,
                        'resize': None,
                        'random_rotation': 5,
                        'random_flip_left_right': True,
                        'random_flip_up_down': False,
                        'random_shift': [.1, .1],
                        'random_zoom': [0.2, 0.2],
                        'random_shear': [5, 5],
                        'random_brightness': 0.2,
                        'random_hue': 0.01,
                        'random_contrast': [0.6, 1.4],
                        'random_crop': random_crop_size,
                        'random_noise': 100,
                        'random_saturation': [0.5, 2],
                        'num_transforms':10}

        BATCH_SIZE = 2
        flist = [DATADIR+'Lenna.png'] * 10 * BATCH_SIZE
        labels = [0] * 10 * BATCH_SIZE

        # test for classification
        path_tfrecord_0 = DATADIR+'ds_from_tfrecord_0.tfrecord'
        TfrecordConverter().from_path_label(flist,
                                                     labels,
                                                     path_tfrecord_0)

        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE, label_type='class',
                            **DATAGEN_CONF, training=True)
        ds, cnt = dc.from_tfrecords([path_tfrecord_0])

        rep_cnt = 0
        for img, label in iter(ds):
            rep_cnt += 1

        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"

        tool.plot_dsresult(ds.take(10), BATCH_SIZE, 10,
                           DATADIR+'test_ds_from_tfrecord.png')

        # test for segmentation
        path_tfrecord = DATADIR+'ds_from_tfrecord.tfrecord'
        TfrecordConverter().from_path_label(flist,
                                                     flist,
                                                     path_tfrecord)

        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE,
                            label_type='segmentation',
                            **DATAGEN_CONF,  training=True)
        ds, cnt = dc.from_tfrecords([path_tfrecord])

        rep_cnt = 0
        for img, label in iter(ds):
            rep_cnt += 1

        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"
        assert label.shape[1:3] == random_crop_size, "crop size is invalid"

        tool.plot_dsresult(ds.take(10), BATCH_SIZE, 10,
                           DATADIR+'test_ds_from_tfrecord.png')

    def test_from_tfrecord_sample_ratio(self):

        random_crop_size = [100, 254]
        # data augmentation configurations:
        DATAGEN_CONF = {'standardize': True,
                        'resize': None,
                        'random_rotation': 5,
                        'random_flip_left_right': True,
                        'random_flip_up_down': False,
                        'random_shift': [.1, .1],
                        'random_zoom': [0.2, 0.2],
                        'random_shear': [5, 5],
                        'random_brightness': 0.2,
                        'random_hue': 0.01,
                        'random_contrast': [0.6, 1.4],
                        'random_crop': random_crop_size,
                        'random_noise': 100,
                        'random_saturation': [0.5, 2],
                        'num_transforms':10}

        BATCH_SIZE = 5
        flist = [DATADIR+'Lenna.png'] * 10 * BATCH_SIZE
        # test for ratio_samples
        labels = [0] * 10 * BATCH_SIZE
        path_tfrecord_0 = DATADIR+'ds_from_tfrecord_0.tfrecord'
        TfrecordConverter().from_path_label(flist,
                                                     labels,
                                                     path_tfrecord_0)
        labels = [1] * 10 * BATCH_SIZE
        path_tfrecord_1 = DATADIR+'ds_from_tfrecord_1.tfrecord'
        TfrecordConverter().from_path_label(flist,
                                                     labels,
                                                     path_tfrecord_1)

        dc = DatasetCreator(5, 10,
                            label_type='class',
                            repeat=False,
                            **DATAGEN_CONF,  training=True)
        ds, cnt = dc.from_tfrecords([[path_tfrecord_0], [path_tfrecord_1]],
                                            ratio_samples=np.array([0.1, 1000], dtype=np.float32))

        img, label = next(iter(ds.take(1)))
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"
        assert all(label == 1), "sampled label is invalid this sometimes happen"

        ds, cnt = DatasetCreator(5, 50,
                                 label_type='class',
                                 repeat=False,
                                 **DATAGEN_CONF,
                                 training=True).from_tfrecords([[path_tfrecord_0], [path_tfrecord_1]],
                                                                       ratio_samples=np.array([1, 1], dtype=np.float32))
        rep_cnt = 0
        for img, label in iter(ds):
            rep_cnt += 1
        assert rep_cnt == 2, "repetition count is invalid"
        assert any(label == 1) and any(label == 0), "sampled label is invalid"

        # check for sampling ratio
        dc = DatasetCreator(5, 10,
                            label_type='class',
                            repeat=True,
                            **DATAGEN_CONF,  training=True)
        ds, cnt = dc.from_tfrecords([[path_tfrecord_0], [path_tfrecord_1]],
                                            ratio_samples=np.array([1, 10], dtype=np.float32))
        ds = ds.take(200)
        cnt_1, cnt_0 = 0, 0
        for img, label in ds:
            cnt_0 += (label.numpy() == 0).sum()
            cnt_1 += (label.numpy() == 1).sum()

        assert 1/10 - 1/100 < cnt_0 / cnt_1 < 1/10 + 1/100,\
            "sampling ratio is invalid. this happen randomely. please retry:"\
                + str(cnt_0/cnt_1)

    def test_from_path(self):

        BATCH_SIZE = 2
        flist_imgs = [DATADIR+'Lenna.png'] * 10
        flist_seglabels = flist_imgs.copy()
        img_org = np.array(Image.open(flist_imgs[0]))
        clslabels = list(range(10))

        path_tfrecord = DATADIR+'test_from_path.tfrecord'
        TfrecordConverter().from_path_label(flist_imgs,
                                                     flist_seglabels,
                                                     path_tfrecord)

        # check segmentation label
        dc = DatasetCreator(1, BATCH_SIZE, training=True)
        ds, imgcnt = dc.from_tfrecords([path_tfrecord])

        for i, (img, label) in enumerate(ds):
            assert (img == img_org).numpy().all(), 'image is changed'
            assert (label == img_org).numpy().all(), 'labels is changed'

        TfrecordConverter().from_path_label(flist_imgs,
                                                     clslabels,
                                                     path_tfrecord)

        # check class label
        dc = DatasetCreator(False, BATCH_SIZE, training=True)
        ds, datacnt = dc.from_tfrecords([path_tfrecord])

        for i, (img, label) in enumerate(ds):
            assert (img == img_org).numpy().all(), 'image is changed'
            assert all(label.numpy() == 
                       clslabels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]), '''label is
                                                a                    changed'''            
        
        
        
    def test_from_3inputs(self):

        BATCH_SIZE = 2
        img_org = np.array(Image.open(DATADIR+'Lenna.png'))
        clslabels = list(range(10))
        
        # test uint8 and float32 tiff
        # save as float32 tiff
        fp32 = np.array(Image.open(DATADIR+'Lenna.png')).astype(np.float32) // 256
        Image.fromarray(fp32[:,:,0]).save(DATADIR+'Lenna.tif')
        
        flist_imgs = [(DATADIR+'Lenna.png',DATADIR+'Lenna.tif',DATADIR+'Lenna.png')
                      for i in range(10)]
        # test 3 images in tfrecord and classification
        path_tfrecord = DATADIR+'test_3_inimgs.tfrecord'
        TfrecordConverter().from_path_label(flist_imgs,
                                            clslabels,
                                            path_tfrecord,
                                            n_imgin=3)
        
        dc = DatasetCreator(False, BATCH_SIZE, num_transforms=20, training=True)
        ds, datacnt = dc.from_tfrecords([path_tfrecord])
        
        for i, inputs in enumerate(ds):
            assert (inputs[0]['image_in0'] == img_org).numpy().all(), 'in_image0 is changed'
            assert (inputs[0]['image_in1'] == fp32).numpy().all(), 'in_image1 is changed'
            assert (inputs[0]['image_in2'] == img_org).numpy().all(), 'in_image2 is changed'
            assert all(inputs[1].numpy() == \
                       clslabels[i*BATCH_SIZE:(i+1)*BATCH_SIZE]), 'label is changed'
                
        # test 3 images in tfrecord and segmentation
        path_tfrecord = DATADIR+'test_3_inimgs_seg.tfrecord'
        TfrecordConverter().from_path_label(flist_imgs,
                                                     [DATADIR+'Lenna.png'] * 10,
                                                     path_tfrecord,
                                                     n_imgin=3)
        
        dc = DatasetCreator(False, BATCH_SIZE, num_transforms=20, training=True)
        ds, datacnt = dc.from_tfrecords([path_tfrecord])
        
        for i, inputs in enumerate(ds):
            assert (inputs[0]['image_in0'] == img_org).numpy().all(), 'in_image0 is changed'
            assert (inputs[0]['image_in1'] == fp32).numpy().all(), 'in_image1 is changed'
            assert (inputs[0]['image_in2'] == img_org).numpy().all(), 'in_image2 is changed'
            assert (inputs[1] == img_org).numpy().all(), 'label is changed'
            
        
    def test_private_functions_in_DatasetCreator(self):
        
        BATCH_SIZE = 2
        img_org = Image.open(DATADIR+'Lenna.png')
        shape = list(np.array(img_org).shape)
        fp32 = np.array(Image.open(DATADIR+'Lenna.png')).astype(np.float32) // 256
        Image.fromarray(fp32[:,:,0]).save(DATADIR+'Lenna.tif')     
        
        clslabels = list(range(10))        
        flist_imgs = [(DATADIR+'Lenna.png',DATADIR+'Lenna.tif',DATADIR+'Lenna.png') 
                      for i in range(10)]
        
        path_tfrecord = DATADIR+'test_3_inimgs.tfrecord'
        TfrecordConverter().from_path_label(flist_imgs,
                                                     clslabels,
                                                     path_tfrecord,
                                                     n_imgin=3)
        
        dc = DatasetCreator(1, BATCH_SIZE, training=True)        
        path_tfrecords = [path_tfrecord, path_tfrecord]
        (ds, num_img, label_type, imgs_dtype, 
         imgs_shape, label_shape) = dc._get_ds_tfrecord(1, path_tfrecords)
        
        # test _set_formats        
        example_formats = dc._gen_example(label_type, imgs_dtype, imgs_shape)
        decoders = dc._decoder_creator(label_type, imgs_dtype, imgs_shape)
        
        
        assert example_formats['image_in0'].dtype == tf.string
        assert example_formats['image_in1'].dtype == tf.string
        assert example_formats['image_in2'].dtype == tf.string
        assert example_formats['label'].dtype == tf.int64
        
        ds_decoded = (ds.batch(BATCH_SIZE)
                  .apply(tf.data.experimental.parse_example_dataset(example_formats))
                  .map(decoders))
                
        # define augmentation
        datagen_confs = {'random_rotation':5, 
                         'num_transforms':5}
        
        inputs_shape, input_label_shape = dc._get_inputs_shapes(ds_decoded, label_type)
        
        seeds = np.random.uniform(0, 2**32, (int(1e6)))  
        if len(imgs_dtype) > 1:# multiple input    
            aug_funs = []
            for shape in inputs_shape:
                datagen_confs['input_shape'] = shape
                aug_funs.append(AugmentImg(**datagen_confs,
                                     seeds = seeds))
            if label_type == 'segmentation':
                datagen_confs['input_shape'] = input_label_shape
                aug_funs.append(AugmentImg(**datagen_confs,
                                     seeds = seeds))
            elif label_type == 'class':
                aug_funs.append(lambda x: x)
            
            aug_fun = dc._apply_aug(aug_funs)

        ds_aug = ds_decoded.map(aug_fun)
                    
        ds_out = ds_aug.map(dc._ds_to_dict(example_formats.keys()))
        test_ret = next(iter(ds_out))
        
        assert test_ret[0]['image_in0'].shape == [BATCH_SIZE, *imgs_shape[0]], "invalid image 0 size"
        assert test_ret[0]['image_in1'].shape == [BATCH_SIZE, *imgs_shape[1]], "invalid image 1 size"
        assert test_ret[0]['image_in2'].shape == [BATCH_SIZE, *imgs_shape[2]], "invalid image 2 size"
        assert test_ret[1].shape == BATCH_SIZE, "invalid label size"            
        
        

    def test_sharded_from_path(self):

        flist_imgs = [DATADIR+'Lenna.png'] * 10
        flist_seglabels = flist_imgs.copy()
        img_org = np.array(Image.open(flist_imgs[0]))
        clslabels = list(range(10))

        path_tfrecord = DATADIR+'test_shards_from_path.tfrecord'
        TfrecordConverter().from_path_label(flist_imgs,
                                                     flist_seglabels,
                                                     path_tfrecord,
                                                     image_per_shard=3)

        path_tfrecords = glob(DATADIR+'test_shards_from_path_?.tfrecord')
        assert len(path_tfrecords) == 4, 'num of shards is invalid'

        # check segmentation label
        dc = DatasetCreator(1, 1, label_type='segmentation', training=True)
        ds, imgcnt = dc.from_tfrecords(path_tfrecords)

        for i, (img, label) in enumerate(ds):
            assert (img == img_org).numpy().all(), 'image is changed'
            assert (label == img_org).numpy().all(), 'labels is changed'

        path_tfrecord = DATADIR+'test_shards_from_path_seg.tfrecord'
        TfrecordConverter().from_path_label(flist_imgs,
                                                     clslabels,
                                                     path_tfrecord,
                                                     image_per_shard=2)

        path_tfrecords = glob(DATADIR+'test_shards_from_path_seg_?.tfrecord')
        assert len(path_tfrecords) == 5, 'num of shards is invalid'

        # check class label
        dc = DatasetCreator(False, 1, label_type='class', training=True)
        ds, datacnt = dc.from_tfrecords(path_tfrecords)

        list_label = []
        for i, (img, label) in enumerate(ds):
            list_label.append(label.numpy())
            assert (img == img_org).numpy().all(), 'image was changed'

        label_all = np.concatenate(sorted(list_label))
        assert all(label_all == clslabels), 'label was changed'
        
        
        #check from_ary_label()        

    def test_from_ary_label(self):

        random_crop_size = [100, 254]
        # data augmentation configurations:
        DATAGEN_CONF = {'standardize': True,
                        'resize': None,
                        'random_rotation': 5,
                        'random_flip_left_right': True,
                        'random_flip_up_down': False,
                        'random_shift': [.1, .1],
                        'random_zoom': [0.2, 0.2],
                        'random_shear': [5, 5],
                        'random_brightness': 0.2,
                        'random_contrast': [0.6, 1.4],
                        'random_crop': random_crop_size,
                        'random_noise': 100,
                        'num_transforms':10}

        BATCH_SIZE = 2
        with Image.open(DATADIR+'Lenna.png').convert('RGB') as img:
            image = np.asarray(img)
        image = np.tile(image, (10 * BATCH_SIZE, 1, 1, 1))
        # add channel 4
        image = np.concatenate([image, np.zeros(image.shape[:3], dtype=np.uint8)[
                               :, :, :, np.newaxis]], axis=3)

        labels = [0] * 10 * BATCH_SIZE

        # test for classification
        path_tfrecord = DATADIR+'ds_from_tfrecord.tfrecord'
        TfrecordConverter().from_ary_label(image,
                                                    labels,
                                                    path_tfrecord)

        # for preproc, set input dimension
        DATAGEN_CONF['input_shape'] = [BATCH_SIZE,*image.shape[1:3], 3]
        
        def preproc(img, lbl): return (img[:, :, :, :3], lbl)

        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE,
                            label_type='class', preproc=preproc,
                            **DATAGEN_CONF, training=True)
        ds, cnt = dc.from_tfrecords([path_tfrecord])

        rep_cnt = 0
        test = next(iter(ds))
        for img, label in iter(ds):
            rep_cnt += 1

        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"
        assert img.shape[3] == 3, "data shape is invalid"

        #test for segmentation
        path_tfrecord = DATADIR+'ds_from_tfrecord.tfrecord'
        TfrecordConverter().from_ary_label(image,
                                                    image,
                                                    path_tfrecord)

        def preproc(img, lbl): return (img, lbl[:, :, :, :3])

        DATAGEN_CONF['input_shape'] = [BATCH_SIZE,*image.shape[1:]]
        DATAGEN_CONF['input_label_shape'] = [BATCH_SIZE,*image.shape[1:3], 3]
        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE,
                            label_type='segmentation', preproc=preproc,
                            **DATAGEN_CONF, training=True)
        ds, cnt = dc.from_tfrecords([path_tfrecord])

        rep_cnt = 0
        for img, label in iter(ds):
            rep_cnt += 1

        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"
        assert img.shape[3] == 4, "data shape is invalid"
        assert label.shape[3] == 3, "data shape is invalid"

        # test for no labels
        path_tfrecord = DATADIR+'ds_from_tfrecord.tfrecord'
        DATAGEN_CONF['input_shape'] = [BATCH_SIZE,*image.shape[1:]]
        TfrecordConverter().from_ary_label(image,
                                                    None,
                                                    path_tfrecord)

        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE,
                            label_type=None,
                            **DATAGEN_CONF, training=True)
        ds, cnt = dc.from_tfrecords([path_tfrecord])

        rep_cnt = 0
        for img in iter(ds):
            rep_cnt += 1
            
        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape == [BATCH_SIZE, *random_crop_size, 4], "crop size is invalid"



    
class TestAugmentImg(unittest.TestCase):
    
    def test_standardization(self):

        im = np.arange(3*4*5*3, dtype=np.uint8).reshape(3, 4, 5, 3)
        tn = tf.Variable(im)
        
        ret = AugmentImg()._standardization(tn, 3).numpy()

        mean_axis = np.mean(ret, axis=(1, 2, 3))
        std_axis = np.std(ret, axis=(1,2,3))

        assert np.allclose(mean_axis, 0), 'standardization failed'
        assert np.allclose(std_axis, 1), 'standardization failed'
        
        
    def test_set_seed(self):
        
        BATCH_SIZE = 2
        img = np.array(Image.open(DATADIR+'Lenna.png'))
        img1 = np.tile(img, (BATCH_SIZE,1,1,1))
        img2 = img1.copy()
                
        # data augmentation configurations:
        DATAGEN_CONF = {'standardize': False,
                        'resize': [100,100],
                        'random_rotation': 5,
                        'random_flip_left_right': True,
                        'random_flip_up_down': False,
                        'random_shift': [25, 25],
                        'random_zoom': [0.2, 0.2],
                        'random_shear': [5, 5],
                        'random_brightness': 0.2,
                        'random_hue': 0.00001,
                        'random_contrast': [0.6, 1.4],
                        'random_crop': None,  
                        'random_noise': 5,
                        'random_saturation': [0.5, 1.5],
                        'input_shape':[BATCH_SIZE, 512, 512, 3],
                        'num_transforms':10}
        
        
        seeds = np.random.uniform(0, 100, (int(1e6)))
        aug1 = AugmentImg(**DATAGEN_CONF, seeds = seeds, training=True)        
        
        aug2 = AugmentImg(**DATAGEN_CONF, seeds = seeds, training=True)
        
        ret1 = aug1(img1)
        ret2 = aug2(img2)
        
        assert (ret1 == ret2).numpy().all(), "aug is not same"        
    

    def test_add_blur(self):

        imgs = np.random.rand(2*5*4*3).reshape(2, 5, 4, 3) * 255
        sigmas = (1.0, 0.5)
        kernel_size = 3

        plt.imshow(imgs[1].astype(np.int8))

        padimg = np.pad(imgs, [[0, 0], [1, 1], [1, 1],
                        [0, 0]], mode='symmetric')
        ret = AugmentImg()._add_blur(imgs, sigmas, 3, 3)
        plt.imshow(ret.numpy()[1].astype(np.int8))

        kernels = AugmentImg()._get_gaussian_kernels(sigmas, kernel_size)
        kernels = tf.repeat(tf.expand_dims(kernels, 3), 3, 3)

        testval = tf.reduce_sum(padimg[:, 0:3, 0:3, :] * kernels, axis=(1, 2))
        assert np.allclose(ret[:, 0, 0, :], testval), 'invalid blur'
        testval = tf.reduce_sum(padimg[:, 4:7, 3:6, :] * kernels, axis=(1, 2))
        assert np.allclose(ret[:, 4, 3, :], testval), 'invalid blur'

    def test_get_gaussian_kernels(self):

        sigmas = (1.0, 1.5)
        kernel_size = 3

        # check kernel
        kernel_tfaug = AugmentImg()._get_gaussian_kernels(sigmas, kernel_size)

        kernel_tfa = tfa.image.filters._get_gaussian_kernel(
            sigmas[0], kernel_size)
        gaussian_kernel_x = tf.reshape(kernel_tfa, [1, kernel_size])
        gaussian_kernel_y = tf.reshape(kernel_tfa, [kernel_size, 1])
        kernel_tfa = tfa.image.filters._get_gaussian_kernel_2d(
            gaussian_kernel_y, gaussian_kernel_x)

        assert (kernel_tfa == kernel_tfaug[0]).numpy(
        ).all(), 'invalid blur kernel1'

        kernel_tfa = tfa.image.filters._get_gaussian_kernel(
            sigmas[1], kernel_size)
        gaussian_kernel_x = tf.reshape(kernel_tfa, [1, kernel_size])
        gaussian_kernel_y = tf.reshape(kernel_tfa, [kernel_size, 1])
        kernel_tfa = tfa.image.filters._get_gaussian_kernel_2d(
            gaussian_kernel_y, gaussian_kernel_x)

        assert (kernel_tfa == kernel_tfaug[1]).numpy(
        ).all(), 'invalid blur kernel2'

    def test_cnv2d_minibatchwise(self):

        imgs = np.arange(2*5*4*3, dtype=np.float32).reshape(2, 5, 4, 3)

        kernels = np.arange(2*3*3*1, dtype=np.float32).reshape(2, 3, 3, 1)
        kernels = np.repeat(kernels, 3, axis=3)

        ret = AugmentImg()._cnv2d_minibatchwise(imgs, kernels)

        assert np.allclose((imgs[:, 1:4, 1:4, :] * kernels).sum(axis=(1, 2)),
                           ret[:, 2, 2, :].numpy()), 'calc error1'

        assert np.allclose((imgs[:, 2:5, 0:3, :] * kernels).sum(axis=(1, 2)),
                           ret[:, 3, 1, :].numpy()), 'calc error2'


    def test_tfdata_vertual(self):

        BATCH_SIZE = 10
        image = np.arange(5**3).reshape(5, 5, 5).astype(np.float32)
        image = np.tile(image, (BATCH_SIZE, 1, 1, 1))

        random_zoom = (.1, .1)
        random_shift = (.1, .1)
        random_saturation = None
        training = True
        aug_fun = AugmentImg(
            standardize=True,
            random_flip_left_right=True,
            random_flip_up_down=True,
            random_shift=random_shift,
            random_zoom=random_zoom,
            random_brightness=0.2,
            random_saturation=random_saturation,
            training=training,
            num_transforms=100)

        image = image.astype(np.float32)

        test_cases = {'4dim': image, '3dim': image[:, :, :, 0]}

        for no, case in enumerate(test_cases):
            with self.subTest(case=case):
                image = test_cases[case]

                def py_function(x):
                    return x

                def aug_fun(x):
                    return x

                def func(x): return tf.py_function(
                    py_function, [x], tf.float32)

                ds = tf.data.Dataset.from_tensors(image).map(func).map(aug_fun)

                tf.print('get data')
                img = next(ds.take(1).__iter__())
                # print(img.shape)

    def test_central_crop(self):

        # image and lbl which you want to test
        testimg = DATADIR+'Lenna.png'
        testlbl = DATADIR+'Lenna.png'

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

        training = False

        func = AugmentImg(standardize=False,
                          random_flip_left_right=False,
                          random_flip_up_down=False,
                          random_shift=None,
                          random_zoom=None,
                          random_brightness=False,
                          random_saturation=False,
                          central_crop=[256, 128],
                          training=training,
                          num_transforms=10)

        img, lbl = func(image, label)
        lbl_offset_y = (label.shape[1] - 256) // 2
        lbl_offset_x = (label.shape[1] - 128) // 2

        self.assertEqual(img.shape, (10, 256, 128, 3))
        self.assertEqual(lbl.shape, (10, 256, 128, 3))

        self.assertTrue(np.allclose(lbl.numpy(),
                                    label[:, lbl_offset_y:lbl_offset_y+256, lbl_offset_x:lbl_offset_x+128, :]))
        self.assertTrue(np.allclose(img.numpy(),
                                    image[:, lbl_offset_y:lbl_offset_y+256, lbl_offset_x:lbl_offset_x+128, :]))

    """
    below code is each transformation test
    """

    def batch_transform(self):

        # interpolation='bilinear'
        BATCH_SIZE = 10
        interpolation = 'nearest'

        testimg = DATADIR+'Lenna.png'
        with Image.open(testimg).convert('RGB') as img:
            image = np.asarray(img)
        image = np.tile(image, (BATCH_SIZE, 1, 1, 1))

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
            shift_y += tf.random.uniform([batch_size], -
                                         shift_size[0], shift_size[0])
            shift_x += tf.random.uniform([batch_size], -
                                         shift_size[1], shift_size[1])

        shear_theta = np.array([0, 0], dtype=np.float32)
        if shear_theta is not None:
            shear_tan = tf.tan(shear_theta / 180 * math.pi)
            shear_y += tf.random.uniform([batch_size], -
                                         shear_tan[0], shear_tan[0])
            shear_x += tf.random.uniform([batch_size], -
                                         shear_tan[1], shear_tan[1])

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
             [0, 0, 1]], tf.float32),
            tf.transpose([shift_y, shift_x, shear_y, shear_x, zoom_y, zoom_x]))

        rot_theta = 90
        if rot_theta is not None:
            rad_theta = rot_theta / 180 * math.pi
            rot = tf.random.uniform([batch_size], -rad_theta, rad_theta)
            h11 = tf.cos(rot)
            h12 = -tf.sin(rot)
            h21 = tf.sin(rot)
            h22 = tf.cos(rot)
            shift_rot_y = ((size_y - size_y * tf.cos(rot)) -
                           (size_x * tf.sin(rot))) / 2
            shift_rot_x = ((size_x - size_x * tf.cos(rot)) +
                           (size_y * tf.sin(rot))) / 2

            rot_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                [[x[0], x[1], x[5]],
                 [x[2], x[3], x[4]],
                 [0, 0, 1]], tf.float32),
                tf.transpose([h11, h12, h21, h22, shift_rot_y, shift_rot_x]))

            trans_matrix = tf.keras.backend.batch_dot(trans_matrix, rot_matrix)

        # get matrix
        M = tfa.image.transform_ops.matrices_to_flat_transforms(
            tf.linalg.inv(trans_matrix))

        # execute
        retimg = tfa.image.transform(image, M, interpolation=interpolation)

        # crop
        random_crop = (256, 512)
        retimg = tf.image.random_crop(image,
                                      size=tf.concat((tf.expand_dims(batch_size, -1),
                                                      random_crop,
                                                      tf.expand_dims(depth, -1)), axis=0))

        fig, axs = plt.subplots(
            BATCH_SIZE, 1, figsize=(3, BATCH_SIZE), dpi=300)
        for i, im in enumerate(retimg):
            axs[i].axis("off")
            axs[i].imshow(np.squeeze(im.numpy()))

        plt.savefig(DATADIR+'test_batch_transform.png')

    def single_transform(self):

        interpolation = 'nearest'

        testimg = DATADIR+'Lenna.png'
        with Image.open(testimg).convert('RGB') as img:
            image = np.asarray(img)
        size_y, size_x = image.shape[:2]

        trans_matrix = tf.eye(3, 3, dtype=tf.float32)

        shift_ratio = np.array([0, 0])
        if shift_ratio is not None:
            shift_val = image.shape[:2] * shift_ratio
            trans_matrix += np.array([[0, 0, shift_val[1]],
                                      [0, 0, shift_val[0]],
                                      [0, 0, 0]], np.float32)

        shear_theta = np.array([0, 0])
        if shear_theta is not None:
            shear_rad = shear_theta / 180 * math.pi
            shift_shear = -(image.shape[:2] * np.tan(shear_rad)) / 2

            trans_matrix += np.array([[0, math.tan(shear_rad[1]), shift_shear[1]],
                                      [math.tan(shear_rad[0]), 0,
                                       shift_shear[0]],
                                      [0, 0, 0]], np.float32)

        zoom = np.array([0, 0])
        if zoom is not None:
            shift_zoom = -(image.shape[:2] * zoom) / 2
            trans_matrix += np.array([[zoom[1], 0, shift_zoom[1]],
                                      [0, zoom[0], shift_zoom[0]],
                                      [0, 0, 0]], np.float32)

        rot_theta = 0
        rad_theta = rot_theta / 180 * math.pi
        if rot_theta is not None:
            shift_rot_y = ((size_y - size_y * math.cos(rad_theta)
                            ) - (size_x * math.sin(rad_theta))) / 2
            shift_rot_x = ((size_x - size_x * math.cos(rad_theta)
                            ) + (size_y * math.sin(rad_theta))) / 2

            trans_matrix = tf.tensordot(trans_matrix,
                                        np.array([[math.cos(rad_theta), -math.sin(rad_theta), shift_rot_x],
                                                  [math.sin(rad_theta), math.cos(
                                                      rad_theta), shift_rot_y],
                                                  [0, 0, 1]], np.float32), axes=[[1], [0]])

        # get matrix
        M = tfa.image.transform_ops.matrices_to_flat_transforms(
            tf.linalg.inv(trans_matrix))

        # transform
        retimg = tfa.image.transform(image, M, interpolation=interpolation)
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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAugTypes)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # TestDatasetCreatoar().test_from_3inputs()
    # TestDatasetCreator().test_from_tfrecord()
    # TestDatasetCreator().test_from_ary_label()
    # TestDatasetCreator().test_from_path()
    # TestAugmentImg().test_augmentation()
