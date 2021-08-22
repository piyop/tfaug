# -*- coding: utf-8 -*-
"""
@license: MIT
@author: t.okuda
"""

import math
import sys
import io
import os
import json
import numpy as np
from tqdm import tqdm

from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data.experimental import AUTOTUNE
from typing import Tuple, Callable, List, Union



class DatasetCreator():
    """
    creator for tf.data.dataset
    
    if create dataset from tfrecord files, use dataset_from_tfrecords
    else if create dataset from image filepaths, use dataset_from_path
    
    dataset_from_tfrecords() can adjust sampling ratios by specifying ratio_samples
    
    
    Methods
    ----------------
    
    
    """

    def __init__(self,
                 shuffle_buffer: int,
                 batch_size: int,
                 label_type: str = None,
                 repeat: bool = False,
                 preproc: Callable = None,
                 augmentation=True,
                 **kwargs: dict):
        """
        creator for tf.data.dataset
    
        Parameters
        ----------
        shuffle_buffer : int
            shuffle buffer size. if you don't need shuffle, use shuffle_buffer=None
        batch_size : int
            batch size.
        label_type : str, optional
            'segmentation' or 'class'.
            automatically load from json file after ver.0.1.0.5
        repeat : bool, optional
            repeat every datasets or use once.
        preproc : Callable, optional
            preprocess callback function before augment image. The default is None
            proprocess function must take args(img, label) and return (img, label)
        **kwargs : dict
            parameters for AugmentImg()
            
        Returns
        -------
        None.
    
        """

        self._shuffle_buffer = shuffle_buffer
        self._batch_size = batch_size
        self._datagen_confs = kwargs
        self._repeat = repeat
        self._preproc = preproc
        self._augmentation = augmentation

        self._labeltype = label_type
        if self._labeltype:
            self._set_labeltype()

    def _set_labeltype(self):

        if self._labeltype == 'class':
            label_dtype = tf.int64
            self._clslabel = True
        elif self._labeltype == 'segmentation':
            label_dtype = tf.string
            self._clslabel = False
        elif self._labeltype is None or self._labeltype == 'None':
            self._tfexample_format = {
                "image": tf.io.FixedLenFeature([], dtype=tf.string)}
            return
        else:
            assert False, 'undifined labeltype:'+self._labeltype

        self._tfexample_format = {"image": tf.io.FixedLenFeature([], dtype=tf.string),
                                  "label": tf.io.FixedLenFeature([], dtype=label_dtype)}

    def dataset_from_path(self, img_paths: List[str],
                          labels: Union[List[str], List[int], np.ndarray] = None,
                          imgtype: Union['png', 'jpg'] = 'png'):

        if self._labeltype is None:
            self._labeltype = check_label_type(labels)
            self._set_labeltype()

        def decode_jpeg(path_img):
            return tf.image.decode_jpeg(tf.io.read_file(path_img))

        def decode_png(path_img):
            return tf.image.decode_png(tf.io.read_file(path_img))

        if imgtype == 'png':
            decode_func = decode_png
        elif imgtype == 'jpeg':
            decode_func = decode_jpeg
        else:
            raise NotImplementedError('imgtype must specify jpeg or png')

        ds_img = tf.data.Dataset.from_tensor_slices(
            img_paths).map(decode_func, num_parallel_calls=AUTOTUNE)

        # TODO: get input shape from ds_aug
        if labels:
            ds_labels = tf.data.Dataset.from_tensor_slices(labels)
            if self._labeltype == 'segmentation':
                ds_labels = ds_labels.map(
                    decode_png, num_parallel_calls=AUTOTUNE)
            zipped = tf.data.Dataset.zip((ds_img, ds_labels))
        else:
            zipped = ds_img
        if self._shuffle_buffer:
            zipped = zipped.shuffle(self._shuffle_buffer)
        if self._repeat:
            zipped = zipped.repeat()

        batch = zipped.batch(self._batch_size)

        if self._preproc:
            batch = batch.map(self._preproc)

        if self._augmentation:
            aug = AugmentImg(**self._datagen_confs, clslabel=self._clslabel)
            batch = batch.map(aug, num_parallel_calls=AUTOTUNE)

        return batch.prefetch(AUTOTUNE)


    def dataset_from_tfrecords(self,
                               path_tfrecords: Union[List[str], List[List[str]]]
                               ratio_samples: List[float] = None):
        """


        Parameters
        ----------
        path_tfrecords : Union[List[str], List[List[str]]]
            filepaths for tfrecords.
        ratio_samples : List[float], optional
            sampling ratios from each tfrecord groups.
            if use this option, path_tfrecords must be 2d List
            if 0 in ratio_samples and not use repat=True, cause hung up while learning

        Returns
        -------
        ds_aug : iterator
            dataset iterator.
        num_img : int
            the number of imgs in tfrecords

        """

        if ratio_samples is None:
            ds, num_img = self._get_ds_tfrecord(
                self._shuffle_buffer, path_tfrecords)
        else:
            assert len(
                path_tfrecords[0][0]) > 1, "if use ratio_samples, you must use 2d list"
            dss = []
            for path_tfrecord in path_tfrecords:
                dss.append(self._get_ds_tfrecord(
                    self._shuffle_buffer, path_tfrecord))

            ds = tf.data.experimental.sample_from_datasets(
                list(zip(*dss))[0], ratio_samples)
            num_img = sum(list(zip(*dss))[1])

        ds_aug = (ds.batch(self._batch_size)
                  .apply(tf.data.experimental.parse_example_dataset(self._tfexample_format))
                  .map(self._decoder, num_parallel_calls=AUTOTUNE))

        if self._preproc:
            ds_aug = ds_aug.map(self._preproc)

        # TODO: if input shape is fixed, get shape from ds_aug
        if self._augmentation:
            # define augmentation
            aug_fun = AugmentImg(**self._datagen_confs,
                                 clslabel=self._clslabel)
            ds_aug = ds_aug.map(aug_fun, num_parallel_calls=AUTOTUNE)

        return ds_aug.prefetch(AUTOTUNE), num_img

    def _get_ds_tfrecord(self, shuffle_buffer, path_tfrecords):

        num_img, label_type = 0, None
        for path_tfrecord in path_tfrecords:
            with open(os.path.splitext(path_tfrecord)[0]+'.json') as fp:
                fileds = json.load(fp)
                num_img += fileds['imgcnt']

                #check label compatibility
                if label_type:
                    assert fileds['label_type'] == label_type, (
                        "label type incompatible:"+path_tfrecord)
                if 'label_type' in fileds.keys():
                    label_type = fileds['label_type']

        if label_type is not None:
            self._labeltype = label_type
            self._set_labeltype()

        # define dataset
        ds = tf.data.TFRecordDataset(
            path_tfrecords, num_parallel_reads=len(path_tfrecords))
        if self._repeat:
            ds = ds.repeat()

        if shuffle_buffer:
            ds = ds.shuffle(shuffle_buffer)

        return ds, num_img

    def _decoder(self, tfexamples):
        return [tf.map_fn(tf.image.decode_png, tfexamples[key], dtype=tf.uint8)
                if value.dtype == tf.string else tfexamples[key]
                for key, value in self._tfexample_format.items()]


def check_label_type(labels):
    """
    check label type in labels[0]

    Parameters
    ----------
    labels : list or str or np.ndarray
        label data source.

    Returns
    -------
    label_type : str
        segmentation or class

    """

    label_type = None
    if labels is not None:
        #check label type
        label_type = 'segmentation' if isinstance(labels[0], str) or \
            (isinstance(labels, np.ndarray) and labels.ndim >= 3) or \
            (isinstance(labels, list) and isinstance(labels[0], np.ndarray)) else 'class'

    return label_type


class TfrecordConverter():

    def __init__(self):
        """
        converter of images to tfrecords

        Parameters
        ----------

        Returns
        -------
        None.

        """

    def np_to_pngstr(self, npary):
        with io.BytesIO() as output:
            Image.fromarray(npary).save(output, format="PNG")
            stimg = output.getvalue()
        return stimg

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def tfrecord_from_path_label(self, path_imgs, labels, path_out,
                                 image_per_shard=None):
        """
        generate tfrecord from image and label paths

        Parameters
        ----------
        path_imgs : list of str
            list of image paths 
        labels : list of str
            list of label paths 
            use None if label isn't need
        path_out : TYPE
            output tfrecord path.

        Returns
        -------
        None.

        """
        self.save_png_label_tfrecord(path_imgs, path_out,
                                     lambda x: np.array(Image.open(x)),
                                     labels,
                                     image_per_shard)

    def _writejson(self, path_record, imgcnt, label_type):
        with open(os.path.splitext(path_record)[0]+'.json', 'w') as file:
            json.dump({'imgcnt': imgcnt,
                       'label_type': str(label_type)}, file)

    def tfrecord_from_ary_label(self, ary, labels, path_out,
                                image_per_shard=None):
        """
        generate tfrecord from image and label arrays

        Parameters
        ----------
        ary : np.ndarray or list of np.ndarray
            image array
        labels : np.ndarray or list of np.ndarray
            for classification labels, dims of labels is 1
            for segmentation labels, dims of labels is 4
            use None if label isn't need
        path_out : TYPE
            output tfrecord path.

        Returns
        -------
        None.

        """
        self.save_png_label_tfrecord(ary,
                                     path_out,
                                     lambda x: np.array(x),
                                     labels,
                                     image_per_shard)

    def save_png_label_tfrecord(self, imgs, path_out, reader_func,
                                labels=None,
                                image_per_shard=None):

        label_type = check_label_type(labels)
        if labels is not None:
            # define label feature
            if label_type == 'segmentation':
                def label_feature(ilabel): return self._bytes_feature(
                    self.np_to_pngstr(ilabel))
            else:
                def label_feature(ilabel): return self._int64_feature(ilabel)

        def image_example(iimg, ilabel=None):
            feature = {'image': self._bytes_feature(self.np_to_pngstr(iimg))}
            if ilabel is not None:
                feature['label'] = label_feature(ilabel)
            return tf.train.Example(features=tf.train.Features(feature=feature))

        prefix, suffix = os.path.splitext(path_out)

        if image_per_shard:
            path_record = prefix+'_0'+suffix
            last_cnt = len(imgs) % image_per_shard
        else:
            path_record = path_out
            last_cnt = len(imgs)

        try:
            writer = tf.io.TFRecordWriter(path_record)
            # save tfrecord
            for i, img in tqdm(enumerate(imgs),
                               total=len(imgs),
                               leave=False):

                if image_per_shard and i != 0 and i % image_per_shard == 0:
                    # use same image as msk
                    writer.close()
                    writer = None
                    self._writejson(path_record, image_per_shard, label_type)
                    path_record = prefix+f'_{i//image_per_shard}'+suffix
                    writer = tf.io.TFRecordWriter(path_record)

                img = reader_func(img)
                if labels is not None:
                    label = labels[i]
                    if label_type == 'segmentation':
                        if isinstance(labels[0], str):
                            ext = os.path.splitext(label)[1]
                            assert ext in ['.jpg', '.JPG', '.png', '.PNG', '.bmp'],\
                                "file extention is imcompatible:"+label
                        label = reader_func(label)
                else:
                    label = None

                writer.write(image_example(img, label).SerializeToString())

            if writer:
                writer.close()
                writer = None
                # save datalen to json
                self._writejson(path_record, last_cnt, label_type)

        finally:
            if writer is not None:
                writer.close()

    def _check_patch_axis(self, patch_size):
        if isinstance(patch_size, list) and len(patch_size) > 1:
            patch_x, patch_y = patch_size[1], patch_size[0]
        else:
            patch_x, patch_y = patch_size, patch_size
        return patch_x, patch_y

    def get_patch_axis(self, len_x, patch_x, len_y, patch_y):
        return ([x for x in range(0, len_x, patch_x)],
                [y for y in range(0, len_y, patch_y)])

    def split_to_patch(self, npimg, patch_size, buffer_size, dtype=np.uint8):
        """
        split npimgs to patch by axies xx and yy

        Parameters
        ----------
        npimg : 3d np.ndarray
            input numpy images shape[y_dim, x_dim, channel_dim]
        patch_size : int or Tuple[int, int]
            patch size split to
        buffer_size : int or Tuple[int, int]
            buffer area size around patch
        dtype : np.typeDict
            output dtype. The default is np.uint8.

        Returns
        -------
        allnp : 4d np.ndarray
            splitted numpy array. splitted images packed to first dimension

        """

        patch_x, patch_y = self._check_patch_axis(patch_size)

        xx, yy = self.get_patch_axis(
            npimg.shape[1], patch_x, npimg.shape[0], patch_y)

        return self.get_patch(npimg, patch_size, buffer_size, xx, yy, dtype=np.uint8)

    def get_patch(self, npimg, patch_size, buffer_size, xx, yy, dtype=np.uint8):
        """
        split npimgs to patch by axies xx and yy

        Parameters
        ----------
        npimg : 3d np.ndarray
            input numpy images shape[y_dim, x_dim, channel_dim]
        patch_size : int or Tuple[int, int]
            patch size split to
        buffer_size : int or Tuple[int, int]
            buffer area size around patch
        xx : List[int, int, ...]
            left-top point of x axies
        yy : List[int, int, ...]
            left-top point of y axies.
        dtype : np.typeDict
            output dtype. The default is np.uint8.

        Returns
        -------
        allnp : 4d np.ndarray
            splitted numpy array. splitted images packed to first dimension

        """
        org_ndim = npimg.ndim
        if org_ndim == 2:
            npimg = np.expand_dims(npimg, 2)

        patch_x, patch_y = self._check_patch_axis(patch_size)

        if isinstance(buffer_size, list) and len(buffer_size) > 1:
            buffer_x, buffer_y = buffer_size[1], buffer_size[0]
        else:
            buffer_x, buffer_y = buffer_size, buffer_size

        padded = np.pad(npimg,
                        [[buffer_y, yy[-1]+1 - npimg.shape[0] + buffer_y + patch_y],
                         [buffer_x, xx[-1]+1 - npimg.shape[1] + buffer_x + patch_x],
                         [0, 0]], 'constant')

        img_size_y = 2*buffer_y + patch_y
        img_size_x = 2*buffer_x + patch_x
        # get paded array
        all_image = [padded[y:y + img_size_y, x:x + img_size_x, :]
                     for y in yy for x in xx]

        allnp = np.array(all_image, dtype=dtype)
        del all_image, padded
        if org_ndim == 2:
            allnp = allnp[:, :, :, 0]

        return allnp


class AugmentImg():
    """

    image augmentation class for tf.data

    """

    def __init__(self,
                 standardize: bool = False,
                 resize: Tuple[int, int] = None,
                 random_rotation: float = 0,
                 random_flip_left_right: bool = False,
                 random_flip_up_down: bool = False,
                 random_shift: Tuple[float, float] = None,
                 random_zoom: Tuple[float, float] = None,
                 random_shear: Tuple[float, float] = None,
                 random_brightness: float = None,
                 random_saturation: Tuple[float, float] = None,
                 random_hue: float = None,
                 random_contrast: Tuple[float, float] = None,
                 random_crop: Tuple[int, int] = None,
                 central_crop: Tuple[int, int] = None,
                 random_noise: float = None,
                 random_blur: float = None,
                 random_blur_kernel: float = 3,
                 interpolation: str = 'nearest',
                 inshape: Tuple[int, int, int, int] = None,
                 clslabel: bool = False,
                 dtype: type = None,
                 input_shape: Tuple[int, int, int, int] = None,
                 input_shape_label: Tuple[int, int, int, int] = None,
                 num_transforms: int = 10000,
                 training: bool = False) \
            -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        __init__() sets the parameters for augmantation.

        When __call__() can take not only input image but also label image.
        Image and label image will be augmented with the same transformation at the same time.
        However, label image is not augmented by random_brightness, random_saturation, standardize

        This augmentation is executed on a batch of images. 
        Input image should be 4d Tensor(batch, x, y, channel)
        The width and height of the images must be the same.

        If training == False, this class will not augment images except standardize,
        resize, random_crop or central_crop.

        Parameters
        ----------
        standardize : bool, optional
            image standardization. The default is True.
            if true, returned image dtype will be float automatically.
        resize : Tuple[int, int], optional
            specify resize image size [y_size, x_size]
            this resize operation is done before below augmentations
        random_rotation : float, optional
            rotation angle(degree). The default is 0.
        random_flip_left_right : bool, optional
            The default is False.
        random_flip_up_down : bool, optional
            The default is False.
        random_shift : Tuple[float, float], optional
            random shift images.
            vartical direction (-list[0], list[0])
            holizontal direction  (-list[1], list[1])
            Each values shows ratio of image size.
            The default is None.
        random_zoom : Tuple[float, float], optional
            random zoom range -random_zoom to random_zoom.
            random_zoom[0] : y-direction, random_zoom[1] : x-direction
            value of random_zoom is ratio of image size
            The default is None.
        random_shear : Tuple[float, float], optional
            randomly adjust y and x directional shear degree
            random_shear[0] : y-direction, random_shear[1] : x-direction
            The default is None.
        random_brightness : float, optional
            randomly adjust image brightness range
            [-max_delta, max_delta).
             The default is None.
        random_saturation : Tuple[float, float], optional
            randomly adjust image brightness range between [lower, upper).
            The default is None.
        random_hue : float, optional
            randomly adjust hue of RGB images between [-random_hue, random_hue]
        random_contrast : Tuple[float, float], optional
            randomly adjust contrast of RGB images by contrast factor 
            which lies between [random_contrast[0], random_contrast[1])
            result image is calculated by (x - mean) * contrast_factor + mean
        random_crop : int, optional
            randomly crop image with size [x,y] = [random_crop_height, random_crop_width].
            The default is None.
        central_crop: int, optional
            crop center of image with size [x,y] = [crop_height, crop_width].
            The default is None.
        random_noise : float, optional
            add random gausian noise. random_noise value mean sigma param of gaussian.
        random_blur : float, optional
            add random gausian blur. This value means sigma param of gaussian.    
            random_blur generate sigma as uniform(0, random_blur) for every mini-batch
            random blur convert integer images to float images.
        random_blur_kernel : int, optional
            kernel size of gaussian random blur . The default is 3
        interpolation : str, The default is nearest.
            interpolation method. nearest or bilinear
        clslabel : bool, The default is False
            If false, labels are presumed to be the same dimensions as the image and 
            apply the same geometric transformations to labels
        dtype : tf.Dtype, The default is None
            tfaug cast input images to this dtype after geometric transformation.
        input_shape : Tuple(int, int, int, int), The default is None
            input image (batch,y,x,channels) dimensions. 
            To reduce CPU load by generating all transform matrices at first, 
            you mus use this.
        input_shape_label : Tuple(int, int, int, int), The default is None
            input label (batch,y,x,channels) dimensions. 
            To reduce CPU load by generating all transform matrices at first, 
            you must use this.
        num_transforms : int, The default is 10,000
            The number of transformation matrixes generated in advance. 
            this must specify input_shape.
        training : bool, optional
            If false, this class don't augment image except standardize.
            The default is False.

        Returns
        -------
        class instance : Callable[[tf.Tensor, tf.Tensor, bool], Tuple[tf.Tensor,tf.Tensor]]

        """

        self._standardize = standardize

        (self._resize, self._random_rotation, self._random_shift, self._random_zoom,
         self._random_shear, self._random_crop, self._central_crop) = (
             None, None, None, None, None, None, None)

        if resize:
            self._resize = tf.cast(resize, tf.int32)

        if random_rotation:
            self._random_rotation = tf.cast(random_rotation, tf.float32)
        self._random_flip_left_right = random_flip_left_right
        self._random_flip_up_down = random_flip_up_down
        if random_shift:
            self._random_shift = tf.cast(random_shift, tf.float32)
        if random_zoom:
            self._random_zoom = tf.cast(random_zoom, tf.float32)
        if random_shear:
            self._random_shear = tf.cast(random_shear, tf.float32)
        self._random_brightness = random_brightness
        self._random_saturation = random_saturation
        self._random_hue = random_hue
        self._random_contrast = random_contrast
        if random_crop:
            self._random_crop = tf.cast(random_crop, tf.int32)
        if central_crop:
            self._central_crop = tf.cast(central_crop, tf.int32)
        self._random_noise = random_noise
        self._random_blur = random_blur
        self._random_blur_kernel = random_blur_kernel
        if interpolation is None:
            self._interpolation = 'nearest'
        else:
            self._interpolation = interpolation
        self._clslabel = clslabel
        self._dtype = dtype
        self._input_shape = input_shape
        self._input_shape_label = input_shape_label
        self._num_transforms = num_transforms
        self._training = training

        self._transform_active = (isinstance(self._random_rotation, tf.Tensor) or
                                  isinstance(self._random_zoom, tf.Tensor) or
                                  isinstance(self._random_shift, tf.Tensor) or
                                  isinstance(self._random_shear, tf.Tensor) or
                                  isinstance(self._resize, tf.Tensor))

        if self._input_shape and self._num_transforms and self._transform_active:
            print('generating transform matrix...', flush=True)
            rep_cnt = math.ceil(self._num_transforms/self._input_shape[0])
            self._Ms = []
            for i in tqdm(range(rep_cnt)):
                self._Ms.append(self._get_transform(self._input_shape))
            self._Ms = tf.constant(
                np.array(self._Ms).reshape(
                    rep_cnt*self._input_shape[0], 8)[:self._num_transforms])

        self._resize_shape = None
        if isinstance(self._resize, tf.Tensor):
            self._resize_shape = self._resize

    @tf.function
    def __call__(self, image: tf.Tensor,
                 label: tf.Tensor = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """


        Parameters
        ----------
        image : 4d tf.Tensor (batch, x, y, channel)
            image to be augment.
        label : 4d tf.Tensor (batch, x, y, channel), optional
            label image to be augment.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            augmented images and labels.

        """

        return self._augmentation(image, label, self._training)

    def _get_transform(self, imgshape):

        if self._transform_active:

            in_size = imgshape[1:3]
            batch_size = imgshape[0]

            size_fl = tf.cast(in_size, dtype=tf.float32)

            trans_matrix = tf.eye(3, 3, [batch_size], dtype=tf.float32)

            shift_y = tf.zeros([batch_size], dtype=tf.float32)
            shift_x = tf.zeros([batch_size], dtype=tf.float32)
            shear_y = tf.zeros([batch_size], dtype=tf.float32)
            shear_x = tf.zeros([batch_size], dtype=tf.float32)
            zoom_y = tf.zeros([batch_size], dtype=tf.float32)
            zoom_x = tf.zeros([batch_size], dtype=tf.float32)
            resize_factor_y = tf.ones([batch_size], dtype=tf.float32)
            resize_factor_x = tf.ones([batch_size], dtype=tf.float32)

            if isinstance(self._resize, tf.Tensor):
                resize_fl = tf.cast(self._resize, tf.float32)
                # update scalling of zoom
                resize_factor_y = (resize_fl[0] / size_fl[0])
                resize_factor_x = (resize_fl[1] / size_fl[1])

            if self._training:
                if isinstance(self._random_shift, tf.Tensor):
                    shift_y += tf.random.uniform([batch_size], -
                                                 self._random_shift[0], self._random_shift[0])
                    shift_x += tf.random.uniform([batch_size], -
                                                 self._random_shift[1], self._random_shift[1])

                if isinstance(self._random_shear, tf.Tensor):
                    shear_tan = tf.tan(self._random_shear / 180 * math.pi)
                    shear_y += tf.random.uniform([batch_size], -
                                                 shear_tan[0], shear_tan[0])\
                        * resize_factor_y
                    shear_x += tf.random.uniform([batch_size], -
                                                 shear_tan[1], shear_tan[1])\
                        * resize_factor_x

                    shift_y += -(size_fl[0] * shear_y) / 2
                    shift_x += -(size_fl[1] * shear_x) / 2

                if isinstance(self._random_zoom, tf.Tensor):
                    zoom_y = tf.random.uniform(
                        [batch_size], -self._random_zoom[0], self._random_zoom[0]) \
                        * resize_factor_y
                    zoom_x = tf.random.uniform(
                        [batch_size], -self._random_zoom[1], self._random_zoom[1])\
                        * resize_factor_x

                    shift_y += -(size_fl[0] * zoom_y) / 2
                    shift_x += -(size_fl[1] * zoom_x) / 2

            if isinstance(self._resize, tf.Tensor):
                zoom_y += resize_factor_y - 1
                zoom_x += resize_factor_x - 1

            trans_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                [[x[5]+1, x[3], x[1]],
                 [x[2], x[4]+1, x[0]],
                 [0, 0, 1]], tf.float32),
                tf.transpose([shift_y, shift_x, shear_y, shear_x, zoom_y, zoom_x]))

            if self._training:
                if isinstance(self._random_rotation, tf.Tensor):
                    rad_theta = self._random_rotation / 180 * math.pi
                    rot = tf.random.uniform(
                        [batch_size], -rad_theta, rad_theta)
                    h11, h12, h21, h22 = tf.cos(
                        rot), -tf.sin(rot), tf.sin(rot), tf.cos(rot)

                    shift_rot_y = (
                        (size_fl[0] - size_fl[0] * tf.cos(rot)) - (size_fl[1] * tf.sin(rot))) / 2
                    shift_rot_x = (
                        (size_fl[1] - size_fl[1] * tf.cos(rot)) + (size_fl[0] * tf.sin(rot))) / 2

                    rot_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                        [[x[0], x[1], x[5]],
                         [x[2], x[3], x[4]],
                         [0, 0, 1]], tf.float32),
                        tf.transpose([h11, h12, h21, h22, shift_rot_y, shift_rot_x]))

                    trans_matrix = tf.keras.backend.batch_dot(
                        trans_matrix, rot_matrix)

            # get matrix
            M = tfa.image.transform_ops.matrices_to_flat_transforms(
                tf.linalg.inv(trans_matrix))

            return M

    def _augmentation(self, image, label=None, training=False):
        """

        Parameters
        --------------------
        image : 4d tensor
            input image
        label : 4d tensor or 1d tensor, optional,
            true segmentation label or classification label
        training : bool, optional,
            training or not
        """
        if self._input_shape:
            image = tf.ensure_shape(image, self._input_shape)

        shape = tf.shape(image)
        batch_size = shape[0]

        # keep image channel dims
        last_image_dim = shape[-1]
        # last_label_dim = tf.shape(label)[-1]

        if label is not None and not self._clslabel:
            if self._input_shape_label:
                label = tf.ensure_shape(label, self._input_shape_label)
            image = tf.concat([image, label], axis=3)

        if training:
            if self._random_flip_left_right:
                image = tf.image.random_flip_left_right(image)

            if self._random_flip_up_down:
                image = tf.image.random_flip_up_down(image)

        if self._transform_active:

            if hasattr(self, "_Ms"):
                M = tf.gather(self._Ms, tf.cast(tf.random.uniform(
                    [batch_size])*self._num_transforms, tf.int32), axis=0)
            else:
                M = self._get_transform(tf.shape(image))

            # geometric transform
            image = tfa.image.transform(
                image, M, interpolation=self._interpolation,
                output_shape=self._resize_shape)

        if isinstance(self._random_crop, tf.Tensor):
            depth = tf.shape(image)[3]
            image = tf.image.random_crop(image,
                                         size=tf.concat((tf.expand_dims(batch_size, -1),
                                                         self._random_crop,
                                                         tf.expand_dims(depth, -1)), axis=0))

        if isinstance(self._central_crop, tf.Tensor):
            # central crop
            shape = tf.shape(image)
            offset_height = (shape[1] - self._central_crop[0]) // 2
            target_height = self._central_crop[0]
            offset_width = (shape[2] - self._central_crop[1]) // 2
            target_width = self._central_crop[1]
            image = tf.image.crop_to_bounding_box(image,
                                                  offset_height,
                                                  offset_width,
                                                  target_height,
                                                  target_width)

        # separete image and label
        if label is not None and not self._clslabel:
            image, label = (image[:, :, :, :last_image_dim],
                            image[:, :, :, last_image_dim:])

        if self._dtype is not None:
            # cast dtype
            image = tf.cast(image, self._dtype)

        if training:
            if self._random_brightness:
                image = tf.image.random_brightness(
                    image, self._random_brightness)
            if self._random_saturation:
                image = tf.image.random_saturation(
                    image, *self._random_saturation)
            if self._random_hue:
                image = tf.image.random_hue(image, self._random_hue)
            if self._random_contrast:
                image = tf.image.random_contrast(image, *self._random_contrast)
            if self._random_noise:
                noise_source = tf.random.normal(tf.shape(image),
                                                mean=0,
                                                stddev=1)
                noise_scale = tf.random.uniform([batch_size, 1, 1, 1],
                                                -self._random_noise, self._random_noise)
                noise = tf.cast(noise_scale * noise_source, image.dtype)
                image += noise
            if self._random_blur:
                sigmas = tf.random.uniform([batch_size], 0, self._random_blur)
                image = self._add_blur(
                    image, sigmas, self._random_blur_kernel, last_image_dim)

        if self._standardize:
            # tf.image.per_image_standardization have a bug #33892
            # image = tf.image.per_image_standardization(image)
            image = self._standardization(image, batch_size)

        if label is not None:
            return image, label
        else:
            return image

    def _standardization(self, image, batch_size):
        # num_dims
        # axs = tf.constant((1, 2, 3))
        # ndim = tf.rank(image)
        # image = tf.cast(image, tf.float32)
        # max_axis = tf.math.reduce_max(image, axs[:ndim-1], keepdims=True)
        # min_axis = tf.math.reduce_min(image, axs[:ndim-1], keepdims=True)
        image = tf.cast(image, tf.keras.backend.floatx())
        axs = tf.constant((1, 2, 3))
        mean = tf.math.reduce_mean(image, axis=axs, keepdims=True)
        stddev = tf.math.reduce_std(image, axis=axs)
        adj_factor = tf.repeat(1.0/tf.math.sqrt(tf.cast(batch_size,
                                                        tf.keras.backend.floatx())),
                               batch_size, axis=0)

        adjusted_stddev = tf.math.maximum(stddev, adj_factor)
        adjusted_stddev = tf.reshape(adjusted_stddev, (batch_size, 1, 1, 1))
        # adjusted_stddev = tf.expand_dims(adjusted_stddev, axis=axs)

        return (image - mean) / adjusted_stddev

    def _add_blur(self, images, sigmas, kernel_size, channel_size):

        kernel = self._get_gaussian_kernels(sigmas, kernel_size)
        kernel = tf.expand_dims(kernel, 3)
        kernel = tf.repeat(kernel, channel_size, axis=3)

        return self._cnv2d_minibatchwise(images, kernel)

    def _get_gaussian_kernels(self, sigmas, kernel_size):

        sigma = tf.expand_dims(tf.convert_to_tensor(sigmas), 1)
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        x = tf.expand_dims(x, 0)
        x = tf.repeat(x, tf.shape(sigma)[0], axis=0)
        x = tf.cast(x ** 2, sigma.dtype)
        x = tf.exp(-x / (2.0 * (sigma ** 2)))
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        y = tf.expand_dims(x, 2)
        x = tf.expand_dims(x, 1)

        return tf.matmul(y, x)

    def _cnv2d_minibatchwise(self, imgs, kernels):

        # iter() is not supported in tf.function
        kb, kh, kw, kc = (tf.shape(kernels)[0],
                          tf.shape(kernels)[1],
                          tf.shape(kernels)[2],
                          tf.shape(kernels)[3])

        kernel_t = tf.transpose(kernels, [1, 2, 0, 3])
        kernel_t = tf.reshape(kernel_t, (kh, kw, kb*kc, 1))
        padded = tf.pad(imgs,
                        [[0, 0], [kh//2, kh//2], [kw//2, kw//2], [0, 0]],
                        "SYMMETRIC")

        img_t = tf.transpose(padded, [1, 2, 0, 3])
        img_t = tf.reshape(img_t, (1, tf.shape(padded)[1],
                                   tf.shape(padded)[2],
                                   tf.shape(padded)[0]*tf.shape(padded)[3]))
        img_t = tf.cast(img_t, dtype=kernels.dtype)

        cnved = tf.nn.depthwise_conv2d(img_t,
                                       filter=kernel_t,
                                       strides=[1, 1, 1, 1],
                                       padding='VALID')
        cnved = tf.reshape(cnved, [tf.shape(imgs)[1],
                                   tf.shape(imgs)[2],
                                   tf.shape(imgs)[0],
                                   tf.shape(imgs)[3]])
        return tf.transpose(cnved, [2, 0, 1, 3])
