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
    """The creator of tf.data.Dataset
    
    if create dataset from tfrecord files, use dataset_from_tfrecords()
    else if create dataset from image filepaths, use dataset_from_path()
    
    dataset_from_tfrecords() can adjust sampling ratios by specifying ratio_samples
    """

    def __init__(self,
                 shuffle_buffer: int,
                 batch_size: int,
                 label_type: str = None,
                 repeat: bool = False,
                 preproc: Callable = None,
                 augmentation: bool = True,
                 **kwargs: dict):
        """creator for tf.data.dataset
        

        Args:
            shuffle_buffer (int): shuffle buffer size. if you don't need shuffle, use shuffle_buffer=None.
            batch_size (int): batch size..
            label_type (str, optional): 'segmentation' or 'class'.
                automatically load from json file after ver.0.1.0.5.
                Defaults to None.
            repeat (bool, optional): repeat every datasets or use once. Defaults to False.
            preproc (Callable, optional): preprocess callback function before augment image. The default is None
                proprocess function must take (img, label) and return (img, label). Defaults to None.
            augmentation (bool, optional): use argmentation or not.
                Defaults to True.
            **kwargs (dict): augmentation parameters. 
                These parameters directly pass to the AugmentImg().
                Refer AugmentImg.__init__() to see the parameter descriptions

        Returns:
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
                          imgtype: str = 'png') -> tf.data.Dataset:
        """create dataset from filepaths
        

        Args:
            img_paths (List[str]): source image paths.
            labels (Union[List[str], List[int], np.ndarray], optional): filepaths or values of labels. Defaults to None.
            imgtype (str, optional): 'png' or 'jpg'. Defaults to 'png'.

        Raises:
            NotImplementedError: if image type is not in (png, jpg).

        Returns:
            dataset (tf.data.Dataset): dataset iterator.

        """

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
                               path_tfrecords: Union[List[str], List[List[str]]],
                               ratio_samples: List[float] = None) -> Tuple[tf.data.Dataset, int]: 
        """create dataset from tfrecords
        

        Args:
            path_tfrecords (Union[List[str], List[List[str]]]): paths to tfrecords 
                generated by TfrecordConverter.
            ratio_samples (List[float], optional): the sampling ratios from path_tfrecords.
                if use ratio_samples, path_tfrecord must be a 2-d list of tfrecord paths.    
                Defaults to None.

        Returns:
            dataset (tf.data.Dataset): dataset iterator.
            num_img (int): the number of images in all tfrecord files.

        """
        

        if ratio_samples is None:
            ds, num_img = self._get_ds_tfrecord(
                self._shuffle_buffer, path_tfrecords)
        else:
            assert len(
                path_tfrecords[0][0]) > 1, "if use ratio_samples, you must use a 2-d list"
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


def check_label_type(labels:Union[List[int],List[str],np.ndarray]) -> str:
    """check label type in labels[0]
    

    Args:
        labels (Union[List[int],List[str],np.ndarray]): label data source..

    Returns:
        label_type (str): 'segmentation' or 'class'.

    """

    label_type = None
    if labels is not None:
        #check label type
        label_type = 'segmentation' if isinstance(labels[0], str) or \
            (isinstance(labels, np.ndarray) and labels.ndim >= 3) or \
            (isinstance(labels, list) and isinstance(labels[0], np.ndarray)) else 'class'

    return label_type


class TfrecordConverter():
    """The converter of images to tfrecords
    
    If you would like to generate tfrecord from image filepaths, 
    use tfrecord_from_path_label()
    else if you would like to generate tfrecord from array or list, 
    use tfrecord_from_ary_label()
        
    If your images are not aligned width and height, you should resize or crop
    images to the same size for learning.
    split_to_patch() support that use case when split large images.    
    

    """  

    def __init__(self):
        """The converter of images to tfrecords
        
        Returns:
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

    def tfrecord_from_path_label(self, path_imgs : List[str],
                                 labels : Union[List[str], List[int], np.ndarray], 
                                 path_out : str,
                                 image_per_shard : int = None):
        """Convert from image paths
        

        Args:
            path_imgs (List[str]): paths to images.
            labels (Union[List[str], List[int], np.ndarray]): if segmentation, path to the label images 
                else if classification, int class labels.
            path_out (str): output path.
            image_per_shard (int, optional): the number of images when split 
                images to shards. Defaults to None.

        Returns:
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

    def tfrecord_from_ary_label(self, ary : Union[List[np.ndarray], np.ndarray],
                                labels : Union[List[int], List[np.ndarray], np.ndarray], 
                                path_out : str,
                                image_per_shard : int = None):
        """Convert from image arrays
        

        Args:
            ary (Union[List[np.ndarray], np.ndarray]): image array.
            labels (Union[List[np.ndarray], np.ndarray]): if segmentation, label image array
                else if classification, integer class labels.
            path_out (str): output path.
            image_per_shard (int, optional): the number of images when split 
                images to shards. Defaults to None.

        Returns:
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

    def split_to_patch(self, npimg : np.ndarray, 
                       patch_size : Union[int, Tuple[int, int]],
                       buffer_size : Union[int, Tuple[int, int]],
                       dtype : np.dtype = np.uint8 ) -> np.ndarray :
        """split images to patch
        

        Args:
            npimg (np.ndarray): input 3-d image.
            patch_size (Union[int, Tuple[int, int]]): patch size to split.
            buffer_size (Union[int, Tuple[int, int]]): overlap buffer size.
            dtype (np.dtype, optional): output dtype. Defaults to np.uint8.

        Returns:
            4-d np.ndarray: each splitted images are packed into first of the 4 dimension.
        """
                

        patch_x, patch_y = self._check_patch_axis(patch_size)

        xx, yy = self.get_patch_axis(
            npimg.shape[1], patch_x, npimg.shape[0], patch_y)

        return self.get_patch(npimg, patch_size, buffer_size, xx, yy, dtype=dtype)

    def get_patch(self, npimg : np.ndarray, 
                  patch_size : Union[int, Tuple[int, int]],
                  buffer_size : Union[int, Tuple[int, int]],
                  xx : List[int], yy : List[int],
                  dtype : np.dtype = np.uint8):
        """
        

        Args:
            npimg (np.ndarray): input 3-d image.
            patch_size (Union[int, Tuple[int, int]]): patch size to split.
            buffer_size (Union[int, Tuple[int, int]]): overlap buffer size.
            xx (List[int]): x-axis to split the point.
            yy (List[int]): y-axis to split the point.
            dtype (np.dtype, optional): dtype. Defaults to np.uint8.

        Returns:
            4-d np.ndarray : each splitted images are packed into first of the 4 dimension.

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
    """image augmentation class for tf.data
    
    set augmentation parameters in __int__() then call it directly or
    in tf.data.Dataset.map()

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
                 clslabel: bool = False,
                 dtype: type = None,
                 input_shape: Tuple[int, int, int, int] = None,
                 input_shape_label: Tuple[int, int, int, int] = None,
                 num_transforms: int = 10000,
                 training: bool = False) \
        -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:            
        """__init__() sets the parameters for augmantation.
        
        __call__() can take not only input image but also label image.
        Image and label image will be augmented with the same transformation at the same time.
        However, label image is not augmented by random_brightness, random_saturation, standardize

        This augmentation is executed on a batch of images. 
        Input image should be 4d Tensor(batch, x, y, channel)
        The image sizes are presumed to be the same on each image when 

        If training == False, this class will not augment images except standardize,
        resize, random_crop or central_crop.


        Args:
            standardize (bool, optional): image standardization. 
                if true, returned image dtype will be float automatically.
                Defaults to False.
            resize (Tuple[int, int], optional): specify resize image size [y_size, x_size]
                this resize operation is done before below augmentations. Defaults to None.
            random_rotation (float, optional): maximum rotation angle(degree). 
                Defaults to 0.
            random_flip_left_right (bool, optional): Defaults to False.
            random_flip_up_down (bool, optional): Defaults to False.
            random_shift (Tuple[float, float], optional): maximum shift ratios of images.
                vartical shift ratio: (-list[0], list[0])
                holizontal shift ratio: (-list[1], list[1])
                Defaults to None.
            random_zoom (Tuple[float, float], optional):random zoom ratios of an image. unit is width and height retios.
                random_zoom[0] is y-direction, random_zoom[1] is x-direction.
                Defaults to None.
            random_shear (Tuple[float, float], optional): randomly shear of image. unit is degree.
                random_shear[0] is y-direction(degrees), random_shear[1] is x-direction(degrees).
                Defaults to None.
            random_brightness (float, optional): maximum image delta brightness range [-random_brightness, random_brightness). 
                The value delta is added to all components of the tensor image.
                image is converted to float and scaled appropriately 
                if it is in fixed-point representation, and 
                delta is converted to the same data type. 
                For regular images, delta should be in the range (-1,1),
                as it is added to the image in floating point representation, 
                where pixel values are in the [0,1) range.
                Defaults to None.
            random_saturation (Tuple[float, float], optional): maximum image saturation factor range between [random_saturation[0],
                                                                random_saturation[1]).
                The value saturation factor is multiplying to the saturation channel of images.
                Defaults to None.
            random_hue (float, optional): maximum delta hue of RGB images between [-random_hue, random_hue).
                max_delta must be in the interval [0, 0.5].
                Defaults to None.
            random_contrast (Tuple[float, float], optional): randomly adjust contrast of RGB images by contrast factor 
                which lies between [random_contrast[0], random_contrast[1])
                result image is calculated by (x - mean) * contrast_factor + mean. 
                Defaults to None.
            random_crop (Tuple[int, int], optional): randomly crop image with size [height,width] = [random_crop[0], random_crop[1]].
                Defaults to None.
            central_crop (Tuple[int, int], optional): crop center of image with size [height,width] = [central_crop[0], central_crop[1]].
                Defaults to None.
            random_noise (float, optional): add random gaussian noise. 
                random_noise value means sigma param of gaussian.
                Defaults to None.            
            random_blur (float, optional): add random gaussian blur. the value means sigma param of gaussian.    
                random_blur generate sigma as uniform(0, random_blur) for every mini-batch
                random blur converts integer images to float images. Defaults to None.
            random_blur_kernel (float, optional): kernel size of gaussian random blur.
                Defaults to 3.
            interpolation (str, optional): interpolation method. nearest or bilinear
                Defaults to 'nearest'.
            clslabel (bool, optional): If false, labels are presumed to be the same dimensions as the image and 
                apply the same geometric transformations to labels. Defaults to False.
            dtype (type, optional): tfaug cast input images to this dtype after
                geometric transformation.
                Defaults to None.
            input_shape (Tuple[int, int, int, int], optional): input image (batch,y,x,channels) dimensions. 
                To reduce CPU load by generating all transform matrices at first, 
                use this. Defaults to None.
            input_shape_label (Tuple[int, int, int, int], optional): input label (batch,y,x,channels) dimensions. 
                To reduce CPU load by generating all transform matrices at first, 
                use this if label is segmentation. Defaults to None.
            num_transforms (int, optional): The number of transformation matrixes generated in advance. 
                when input_shape is used. Defaults to 10,000.
            training (bool, optional): If false, augment is not done except standardize.
                Defaults to False.

        Returns:
            callable object (Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]) : class instance

        """


        float_type = tf.keras.backend.floatx()
            
        self._standardize = standardize

        (self._resize, self._random_rotation, self._random_shift, self._random_zoom,
         self._random_shear, self._random_crop, self._central_crop) = (
             None, None, None, None, None, None, None)

        if resize:
            self._resize = tf.cast(resize, tf.int32)

        if random_rotation:
            self._random_rotation = tf.cast(random_rotation, float_type)
        self._random_flip_left_right = random_flip_left_right
        self._random_flip_up_down = random_flip_up_down
        if random_shift:
            self._random_shift = tf.cast(random_shift, float_type)
        if random_zoom:
            self._random_zoom = tf.cast(random_zoom, float_type)
        if random_shear:
            self._random_shear = tf.cast(random_shear, float_type)
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
        """call function of AugmentImg
        

        Args:
            image (tf.Tensor): 4d tf.Tensor (batch, x, y, channel).
            label (tf.Tensor, optional): 4d or 1d tf.Tensor (batch, x, y, channel), optional. 
                Defaults to None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: augmented images and labels.

        """
        

        return self._augmentation(image, label, self._training)

    def _get_transform(self, imgshape):

        if self._transform_active:

            in_size = imgshape[1:3]
            batch_size = imgshape[0]

            float_type = tf.keras.backend.floatx()
            size_fl = tf.cast(in_size, dtype=float_type)

            trans_matrix = tf.eye(3, 3, [batch_size], dtype=float_type)

            shift_y = tf.zeros([batch_size], dtype=float_type)
            shift_x = tf.zeros([batch_size], dtype=float_type)
            shear_y = tf.zeros([batch_size], dtype=float_type)
            shear_x = tf.zeros([batch_size], dtype=float_type)
            zoom_y = tf.zeros([batch_size], dtype=float_type)
            zoom_x = tf.zeros([batch_size], dtype=float_type)
            resize_factor_y = tf.ones([batch_size], dtype=float_type)
            resize_factor_x = tf.ones([batch_size], dtype=float_type)

            if isinstance(self._resize, tf.Tensor):
                resize_fl = tf.cast(self._resize, float_type)
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
                 [0, 0, 1]], float_type),
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
                         [0, 0, 1]], float_type),
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
