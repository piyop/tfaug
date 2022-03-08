# -*- coding: utf-8 -*-
"""
@license: MIT
@author: t.okuda

"""

import math
import io
import os
import json
import warnings
from collections import OrderedDict, namedtuple
from itertools import permutations
from tqdm import tqdm

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data.experimental import AUTOTUNE
from typing import Tuple, Callable, List, Union


class DatasetCreator():
    """The creator of tf.data.Dataset
    
    if create dataset from tfrecord files, use from_tfrecords()
    else if create dataset from image filepaths, use from_path()
    
    from_tfrecords() can adjust sampling ratios by specifying ratio_samples
    """

    # priority order definition of dtype cast
    cast_order = ('uint8', 'int8', 'float16', 'float32', 'float64')
    cast_order = OrderedDict(tuple((typ, i)
                                   for i, typ in enumerate(cast_order)))
    tfmax_type = {typ: eval('tf.'+typ) for typ in cast_order}

    # dict of label type (claslabel in AugmentImg, FixedLenFeature dtype)
    label_type_dict = {'segmentation': (False, tf.string),
                       'class': (True, tf.int64),
                       'None': (False, None),
                       None: (False, None)}

    def __init__(self,
                 shuffle_buffer: int,
                 batch_size: int,
                 repeat: bool = False,
                 preproc: Callable = None,
                 augmentation: bool = True,
                 **kwargs: dict):
        """creator for tf.data.dataset
        

        Args:
            shuffle_buffer (int): shuffle buffer size. if you don't need shuffle, use shuffle_buffer=None.
            batch_size (int): batch size..
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

    def _decode_raw(self, dtype):

        def decode_raw(x):
            return tf.io.decode_raw(x, out_type=dtype)

        return decode_raw

    def _check_types(self, label_types, imgs_dtypes, imgs_shapes, labels_shapes,
                     labels_dtypes):

        label_type, imgs_dtype, imgs_shape, labels_shape, labels_dtype\
            = None, None, None, None, None
        if len(label_types) > 0:
            #check label accordance
            assert all([label_type == label_types[0] for label_type in label_types]), (
                "label type is not accordance:")
            # set final label type
            label_type = label_types[0]

        if len(imgs_dtypes) > 0:
            #check dtype accordance
            assert all([imgs_dtype == imgs_dtypes[0] for imgs_dtype in imgs_dtypes]), (
                "imgs_dtype is not accordance:")
            imgs_dtype = imgs_dtypes[0]

        if len(imgs_shapes) > 0:
            #check dtype accordance
            assert all([imgs_shape == imgs_shapes[0] for imgs_shape in imgs_shapes]), (
                "imgs_shape is not accordance:")
            imgs_shape = imgs_shapes[0]

        if len(labels_shapes) > 0:
            #check dtype accordance
            assert all([labels_shape == labels_shapes[0] for labels_shape in labels_shapes]), (
                "label_shape is not accordance:")
            labels_shape = labels_shapes[0]

        if len(labels_dtypes) > 0:
            #check dtype accordance
            assert all([labels_dtype == labels_dtypes[0]
                        for labels_dtype in labels_dtypes]), (
                "imgs_dtype is not accordance:")
            labels_dtype = labels_dtypes[0]

        return label_type, imgs_dtype, imgs_shape, labels_shape, labels_dtype

    def _decoder_creator(self, label_type, labels_dtype, labels_shape,
                         imgs_dtype, imgs_shape=None):


        # TODO:label_typeをimageにも定義し、全く同じ処理でimageとlabelを処理するようにする
        def decoder(tfexamples):
            decoded = []
            for i, img_dtype in enumerate(imgs_dtype):
                if img_dtype == 'uint8':
                    decoded.append(
                        tf.map_fn(tf.image.decode_png, tfexamples[f"image_in{i}"], dtype=tf.uint8))
                else:
                    assert imgs_shape is not None, "raw format needs imgs_shape to decode"
                    decoded.append(tf.reshape(tf.map_fn(self._decode_raw(
                        tf.float32), tfexamples[f"image_in{i}"], dtype=tf.float32),
                        [-1, *imgs_shape[i]]))

            for i, label_dtype in enumerate(labels_dtype):
                if label_type == 'class':
                    decoded.append(tfexamples[f"label_in{i}"])
                elif label_type == 'segmentation':
                    if label_dtype == 'uint8':
                        decoded.append(
                            tf.map_fn(tf.image.decode_png, 
                                      tfexamples[f"label_in{i}"], dtype=tf.uint8))
                    else:
                        assert labels_shape is not None, "raw format needs labels_shape to decode"
                        decoded.append(tf.reshape(tf.map_fn(self._decode_raw(
                            tf.float32), tfexamples[f"label_in{i}"], dtype=tf.float32),
                            [-1, *labels_shape[i]]))
                        
            return decoded
        return decoder

    def _gen_example(self, label_type, labels_dtype,  imgs_dtype, imgs_shape):

        example_formats = {}
        for i, dtype in enumerate(imgs_dtype):
            example_formats[f"image_in{i}"] = tf.io.FixedLenFeature(
                [], dtype=tf.string)

        # set label type
        if label_type is not None and label_type != 'None':
            _, dtype = DatasetCreator.label_type_dict[label_type]
            for i, label_dtype in enumerate(labels_dtype):
                example_formats[f"label_in{i}"] = tf.io.FixedLenFeature([],
                                                                        dtype=dtype)

        return example_formats
    
    def _path_decorder(self, imgtype):

        def decode_jpeg(path_img):
            return tf.image.decode_jpeg(tf.io.read_file(path_img))
    
        def decode_png(path_img):
            return tf.image.decode_png(tf.io.read_file(path_img))
    
        if imgtype == '.png':
            decode_func = decode_png
        elif imgtype == '.jpeg':
            decode_func = decode_jpeg
        else:
            raise NotImplementedError('imgtype must specify jpeg or png')
            
        return decode_func
            

    def from_path(self, img_paths: List[str],
                  labels: Union[List[str],
                                List[int], np.ndarray] = None) -> tf.data.Dataset:
        """create dataset from filepaths
        Currently this method only supports .png and .jpg images.      
        

        Args:
            img_paths (Union[List[str], List[List[str]]]): source image paths.
                These images must be same size.
                if you have multiple input images, you should specifiy paths like
                [[image1, image2],[image1, image2], ...]
                
            labels (Union[List[str], List[int], np.ndarray], optional): filepaths 
                or values of labels. if use image files, '.png' format is required.
                Defaults to None.

        Raises:
            NotImplementedError: if image type is not in (png, jpg).

        Returns:
            dataset (tf.data.Dataset): dataset iterator.

        """

        # set label format
        label_type, num_in_labels = get_feature_type(labels)
        img_feature_type, num_in_imgs = get_feature_type(img_paths)

        
        # read img file          
        img_paths = list(zip(*img_paths)) if num_in_imgs > 1 else [img_paths]
        labels = list(zip(*labels)) if num_in_labels > 1 else [labels]
        features = img_paths + labels
        in_types = [img_feature_type] * num_in_imgs + [label_type] * num_in_labels
        
        datasets = []   
        for feature, in_type in zip(features, in_types): 
            ds = tf.data.Dataset.from_tensor_slices(list(feature))        
            if in_type == 'segmentation':                
                label_decoder = self._path_decorder(os.path.splitext(feature[0])[1])
                ds = ds.map(label_decoder, num_parallel_calls=AUTOTUNE)
            
            datasets.append(ds)
            
        zipped = tf.data.Dataset.zip(tuple(datasets))

        if self._shuffle_buffer:
            zipped = zipped.shuffle(self._shuffle_buffer)
        if self._repeat:
            zipped = zipped.repeat()

        batch = zipped.batch(self._batch_size)

        if self._preproc:
            batch = batch.map(self._preproc)

        batch = self.from_dataset(batch, label_type, num_in_imgs)

        return batch.prefetch(AUTOTUNE)

    def _ds_to_dict(self, dict_keys):
        """define callback function for converting dataset to dictionary

        Args:
            dict_keys (list): output dictionary keys.

        Returns:
            Callable : call back function for tf.Data.dataset.map().

        """
        dict_keys = list(dict_keys)

        def ds_to_dict(*args):
            return {key: args[i] for i, key
                    in enumerate(dict_keys)}
        return ds_to_dict

    def _apply_aug(self, aug_funs):
        def apply_aug(*args):
            return [aug_fun(arg) for (arg, aug_fun) in zip(args, aug_funs)] \
                if len(aug_funs) > 1 else aug_funs[0](args[0])
            # return [aug_fun(arg) for (arg, aug_fun) in zip(args, aug_funs)]

        return apply_aug

    def _get_inputs_shapes(self, ds, label_type, num_feature):
        """ get input shapes from ds(tf.Data.Dataset)

        Args:
            ds (tf.Data.Dataset): input dataset.
            label_type (str): input label type(segmentation or classification)
            num_feature (int): number of image inputs

        Returns:
            ret_in (List): list of input image shapes tensor.
            ret_label (List): list of label shapes tensor.

        """

        inputs = next(iter(ds))

        ret_in, ret_label = [], []
        
        # parse images
        if (len(inputs) > 1 and
                isnested(inputs)):  # multiple inputs
            for data_in in inputs[:num_feature]:
                ret_in.append(data_in.shape)
        else:
            ret_in.append(inputs[0].shape)
        
        # parse labels
        if (len(inputs) > 1) and label_type is not None:
            inputs = inputs[num_feature:]
            for data_in in inputs:
                ret_label.append(data_in.shape)

        return ret_in, ret_label

    def _get_decoded_ds(self, path_tfrecords, ratio_samples):

        if ratio_samples is None:
            (ds, num_img, label_type, imgs_dtype,
             imgs_shape, labels_shape,
             labels_dtype) = self._get_ds_tfrecord(self._shuffle_buffer,
                                                   path_tfrecords)
        else:
            assert len(
                path_tfrecords[0][0]) > 1, "if use ratio_samples, you must use a 2-d list"

            dss = [self._get_ds_tfrecord(self._shuffle_buffer, path_tfrecord)
                   for path_tfrecord in path_tfrecords]

            ds = tf.data.experimental.sample_from_datasets(
                list(zip(*dss))[0], ratio_samples)
            num_img = sum(list(zip(*dss))[1])

            (label_types, imgs_dtypes,
             imgs_shapes, labels_shapes,
             labels_dtypes) = list(zip(*dss))[2:]
            (label_type, imgs_dtype,
             imgs_shape, labels_shape,
             labels_dtype) = self._check_types(label_types,
                                               imgs_dtypes,
                                               imgs_shapes,
                                               labels_shapes,
                                               labels_dtypes)

        example_formats = self._gen_example(label_type, labels_dtype,
                                            imgs_dtype, imgs_shape)
        decoder = self._decoder_creator(label_type, labels_dtype, labels_shape, 
                                        imgs_dtype, imgs_shape)

        ds_aug = (ds.batch(self._batch_size)
                  .apply(tf.data.experimental.parse_example_dataset(example_formats))
                  .map(decoder, num_parallel_calls=AUTOTUNE))

        return ds_aug, num_img, label_type, imgs_dtype, example_formats

    def from_tfrecords(self,
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
            dataset (tf.data.Dataset): dataset generator.                
                If tfrecords have multiple input images, 
                this dataset generator generate
                a tuple of dictionary and label like 
                ({'image_in0':data2, 'image_in1':data2,...,}, labels).
            num_img (int): The number of images in all tfrecord files.

        """
        if not isnested(path_tfrecords):
            path_tfrecords = [path_tfrecords]

        (ds_aug, num_img, label_type, imgs_dtype, example_formats
         ) = self._get_decoded_ds(path_tfrecords, ratio_samples)
        if self._preproc:
            ds_aug = ds_aug.map(self._preproc)

        ds_aug = self.from_dataset(ds_aug, label_type, len(imgs_dtype),
                                  data_names=example_formats.keys())

        return ds_aug.prefetch(AUTOTUNE), num_img
            

    def from_dataset(self, ds_org, label_type, num_feature, data_names=None):
        """
        generate augmented dataset from tf.Data.Dataset

        Args:
            ds_org (tf.data.Dataset): Original dataset. If original dataset
                    has multiple inputs and labels, ds_org must generate 
                    date with tuple(Tensor(batch of input)) * num_features
            label_type (str): Segmentation or class.
            num_feature (int): Number of input features.
            data_names (List[str], optional): If num_features > 1, 
                        this option will used as key of output dict.
                        Defaults to None.

        Returns:
            ds_aug (tf.data.Dataset): augmented dataset.
        """

        inputs_shape, input_labels_shape = self._get_inputs_shapes(
            ds_org, label_type, num_feature)
        self._datagen_confs['input_shape'] = inputs_shape
        self._datagen_confs['input_label_shape'] = input_labels_shape

        self._datagen_confs['clslabel'], _ = DatasetCreator.label_type_dict[label_type]

        if self._augmentation:
            seeds = np.random.uniform(0, 2**32, (2**16))

            # aug for image
            aug_funs = []
            for shape in inputs_shape:
                # for training set
                self._datagen_confs['input_shape'] = shape
                aug_funs.append(AugmentImg(**self._datagen_confs,
                                           seeds=seeds))
                
            # aug for label
            label_aug = self._datagen_confs.copy()
            label_aug['random_brightness'] = None
            label_aug['random_contrast'] = None
            label_aug['random_saturation'] = None
            label_aug['random_hue'] = None
            label_aug['random_noise'] = None
            label_aug['random_blur'] = None
            for shape in input_labels_shape:
                label_aug['input_shape'] = shape
                if label_type == 'segmentation':
                    aug_funs.append(AugmentImg(**label_aug,
                                               seeds=seeds))
                elif label_type == 'class':
                    aug_funs.append(lambda x: x)

            aug_fun = self._apply_aug(aug_funs)

            ds_aug = ds_org.map(aug_fun, num_parallel_calls=AUTOTUNE)

        if len(inputs_shape) > 1 or len(input_labels_shape) > 1:
            if data_names is None:
                data_names = [f'image_in{i}' for i in range(len(inputs_shape))
                              ] + [f'label_in{i}' for i 
                                   in range(len(input_labels_shape))]
                                   
            # if multiple inputs or labels, output as dict
            ds_aug = ds_aug.map(self._ds_to_dict(data_names))

        return ds_aug

    def _get_ds_tfrecord(self, shuffle_buffer, path_tfrecords):

        (num_img, label_types, labels_dtypes, imgs_dtypes,
         imgs_shapes, labels_shapes) = 0, [], [], [], [], []
        for path_tfrecord in path_tfrecords:
            with open(os.path.splitext(path_tfrecord)[0]+'.json') as fp:
                fileds = json.load(fp)
                num_img += fileds['imgcnt']

                if 'label_type' in fileds.keys():
                    label_types.append(fileds['label_type'])
                if 'dtypes' in fileds.keys():
                    imgs_dtypes.append(fileds['dtypes'])
                if 'imgs_shapes' in fileds.keys():
                    imgs_shapes.append(fileds['imgs_shapes'])
                if 'labels_shape' in fileds.keys():
                    labels_shapes.append(fileds['labels_shape'])
                if 'labels_dtype' in fileds.keys():
                    labels_dtypes.append(fileds['labels_dtype'])

        (label_type, imgs_dtype,
         imgs_shape, labels_shape, labels_dtype) = self._check_types(label_types,
                                                                     imgs_dtypes,
                                                                     imgs_shapes,
                                                                     labels_shapes,
                                                                     labels_dtypes)

        # define dataset
        ds = tf.data.TFRecordDataset(
            path_tfrecords, num_parallel_reads=len(path_tfrecords))
        if self._repeat:
            ds = ds.repeat()

        if shuffle_buffer:
            ds = ds.shuffle(shuffle_buffer)

        return (ds, num_img, label_type, imgs_dtype,
                imgs_shape, labels_shape, labels_dtype)


def isnested(inputs_shape):
    return (isinstance(inputs_shape, list) or isinstance(inputs_shape, tuple))


def get_feature_type(labels: Union[List[int], List[str], np.ndarray]) -> str:
    """check label type in labels[0]
    

    Args:
        labels (Union[List[int],List[str],np.ndarray]): label data source.

    Returns:
        label_type (str): 'segmentation' or 'class'.
        num_feature (int): number of features

    """

    if labels is None:
        return None, 0

    #check label type
    assert (isinstance(labels, list) or isinstance(labels, tuple)) \
        or len(labels) > 0, \
        'labels is empty list or tuple. if do not use label, use None to labels'

    if isinstance(labels[0], int) or\
            (isinstance(labels, np.ndarray) and labels.ndim <= 1):
        return 'class', 1
    elif isnested(labels) and isnested(labels[0]) and isinstance(labels[0][0], int):
        return 'class', len(labels[0])
   

    elif ((isnested(labels) and isnested(labels[0]))
          and ((isinstance(labels[0][0], str) and
                len(labels[0]) > 1) or
               (isinstance(labels[0][0], np.ndarray) and
                labels[0][0].ndim >= 2))) or\
        (isnested(labels) and isinstance(labels[0], np.ndarray) and
         labels[0].ndim >= 4) or\
            (isinstance(labels, np.ndarray) and labels.ndim >= 5):
        return 'segmentation', len(labels[0])

    elif isinstance(labels[0], str) or \
        (isinstance(labels, np.ndarray) and labels.ndim >= 3) or \
            (isinstance(labels, list) and isinstance(labels[0], np.ndarray)):
        return 'segmentation', 1


class TfrecordConverter():
    """The converter of images to tfrecords
    
    If you would like to generate tfrecord from image filepaths, 
    use from_path_label()
    else if you would like to generate tfrecord from array or list, 
    use from_ary_label()
        
    If your images are not aligned width and height, you should resize or crop
    images to the same size for learning.
    split_to_patch() support that use case when split large images.    
    

    """

    def __init__(self):
        """The converter of images to tfrecords
        
        Returns:
            None.

        """

    def _np_to_pngstr(self, npary):
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

    def _image_reader(self, n_feature):
        def image_reader(path):
            if n_feature == 1:
                return [np.array(Image.open(path))]
            else:
                assert len(path) == n_feature, (
                    "n_inimg >= 2 must use 2Dlist including n_inimg images in each element")
                return [np.array(Image.open(impath)) for impath in path]

        return image_reader

    def from_path_label(self, path_imgs: List[str],
                        labels: Union[List[str], List[int], np.ndarray],
                        path_out: str,
                        image_per_shard: int = None):
        """Convert from image paths
        
        image type must be the same.
        only 1-channel is supported on tiff format

        Args:
            path_imgs (List[str]): paths to images. If you have multile input images, 
                use 2d List which every elements contain a group of input images.
            labels (Union[List[str], List[int], np.ndarray]): if segmentation,
                path to the label images 
                else if classification, list of int class labels.
                If you have multile labels, 
                use 2d List which every elements contain a group of input labels
            path_out (str): output path.
            image_per_shard (int, optional): the number of images when split 
                images to shards. Defaults to None.

        Returns:
            None.
        """

        img_type, n_imgin = get_feature_type(path_imgs)

        image_reader = self._image_reader(n_imgin)
        
        
        label_type, n_lblin = get_feature_type(labels)
        reader_label = None
        if label_type == 'segmentation':
            reader_label = self._image_reader(n_lblin)
        elif label_type == 'class':
            reader_label = self._ary_label_reader(n_lblin)
        
        # decode and save as integer png
        self.save_tfrecord(path_imgs, path_out,
                           lambda x: image_reader(x),
                           label_type,
                           labels,
                           reader_label,
                           image_per_shard)

    def _writejson(self, path_record, imgcnt, label_type, dtypes,
                   imgs_shapes, labels_shape, labels_dtype):
        with open(os.path.splitext(path_record)[0]+'.json', 'w') as file:
            json.dump({'imgcnt': imgcnt,
                       'label_type': label_type,
                       'dtypes': dtypes,
                       'imgs_shapes': imgs_shapes,
                       'labels_shape': labels_shape,
                       'labels_dtype': labels_dtype}, file)
            
    def _ary_label_reader(self, n_lblin):
        if n_lblin > 1:
            return lambda x :[np.array(i) for i in x]
        else:
            return lambda x :[np.array(x)]

    def from_ary_label(self, ary: Union[List[np.ndarray], np.ndarray],
                       labels: Union[List[int], List[np.ndarray], np.ndarray],
                       path_out: str,
                       image_per_shard: int = None):
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

        img_type, n_imgin = get_feature_type(ary)

        if n_imgin > 1:
            def read_func(x): return [np.array(i) for i in x]
        else:
            def read_func(x): return [np.array(x)]

        label_type, n_lblin = get_feature_type(labels)
        reader_label = self._ary_label_reader(n_lblin)
            
        self.save_tfrecord(ary,
                           path_out,
                           read_func,
                           label_type,
                           labels,
                           reader_label,
                           image_per_shard)

    def _segmentation_feature(self, img):
        if img.dtype == np.uint8:
            return self._bytes_feature(self._np_to_pngstr(img))
        elif img.dtype == np.float32 or img.dtype == np.float16:
            return self._bytes_feature(img.tobytes())
        else:
            assert True, 'undifined segmentation feature dtype' + img.dtype

    def _exformat_from_array(self, iimg, label_type, ilabel=None):

        feature = {}
        dtypes, imgs_shapes, labels_shape, labels_dtypes = [], [], [], []

        for i, img in enumerate(iimg):
            dtypes.append(img.dtype.name)
            # add channel axis if ndim < 3
            imgs_shapes.append([*img.shape, 1] if img.ndim < 3 else img.shape)

            feature[f'image_in{i}'] = self._segmentation_feature(img)

        if ilabel is not None:

            for i, label in enumerate(ilabel):
                labels_dtypes.append(label.dtype.name)
                
                # define label feature
                if label_type == 'segmentation':
                    feature[f'label_in{i}'] = self._segmentation_feature(label)
                    labels_shape.append([*label.shape,
                                          1] if label.ndim < 3 else label.shape)
                else:
                    feature[f'label_in{i}'] = self._int64_feature(label)
                    labels_shape.append([1])
               
                

        return feature, dtypes, imgs_shapes, labels_shape, labels_dtypes

    def save_tfrecord(self, imgs, path_out, reader_func, label_type,
                      labels=None, reader_label=None,
                      image_per_shard=None):

        label = None

        prefix, suffix = os.path.splitext(path_out)

        if image_per_shard:
            path_record = prefix+'_0'+suffix
            last_cnt = len(imgs) % image_per_shard
        else:
            path_record = path_out
            last_cnt = len(imgs)

        try:
            os.makedirs(os.path.dirname(path_record), exist_ok=True)
            writer = tf.io.TFRecordWriter(path_record)
            # save tfrecord
            dtype_old, imgs_shape_old, label_shape_old, label_dtype_old = 0, 0, 0, 0
            # to avoid syntax error
            dtype, imgs_shape, labels_shape, label_dtype = None, None, None, None
            for i, img in tqdm(enumerate(imgs),
                               total=len(imgs),
                               leave=False):

                if image_per_shard and i != 0 and i % image_per_shard == 0:
                    # use same image as msk
                    writer.close()
                    writer = None
                    self._writejson(path_record, image_per_shard, label_type,
                                    dtype, imgs_shape, labels_shape, label_dtype)
                    path_record = prefix+f'_{i//image_per_shard}'+suffix
                    writer = tf.io.TFRecordWriter(path_record)

                img = reader_func(img)

                if labels is not None:
                    label = reader_label(labels[i])

                feature, dtype, imgs_shape, labels_shape, label_dtype = \
                    self._exformat_from_array(img, label_type, label)
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))

                if i != 0:
                    # check dtype correspondance
                    assert dtype_old == dtype, \
                        'all input image dtype must be the same'
                    assert imgs_shape_old == imgs_shape, \
                        'all input image shape must be the same'
                    assert label_shape_old == labels_shape, \
                        'all input label shape must be the same'
                    assert label_dtype_old == label_dtype, \
                        'all input label dtype must be the same'

                dtype_old = dtype
                imgs_shape_old = imgs_shape
                label_shape_old = labels_shape
                label_dtype_old = label_dtype

                writer.write(example.SerializeToString())

            if writer:
                writer.close()
                writer = None
                # save datalen to json
                self._writejson(path_record, last_cnt, label_type, dtype,
                                imgs_shape, labels_shape, label_dtype)

        finally:
            if writer is not None:
                writer.close()

    def _check_patch_axis(self, patch_size):
        if (isinstance(patch_size, list) or
                isinstance(patch_size, tuple)) and len(patch_size) > 1:
            patch_x, patch_y = patch_size[1], patch_size[0]
        else:
            patch_x, patch_y = patch_size, patch_size
        return patch_x, patch_y

    def get_patch_axis(self, len_x, patch_x, len_y, patch_y):
        return ([x for x in range(0, len_x, patch_x)],
                [y for y in range(0, len_y, patch_y)])

    def concat_patch(self, nppatch: np.ndarray,
                     cy_size: int,
                     cx_size: int) -> np.ndarray:
        """Concat patches to single image.        
        nppatch must be reshape with 
        (cy_size x nppatch.shape[1], cx_size x nppatch.shape[2], nppatch[3])
        All patches have to be stacked in first dimension of nppatch.
        Order of patches must be 

        Args:
            nppatch (np.ndarray): input patch images. 
                shape is [#(patch), height, width, channel]
            cy_size (int): the num of concat patches along y(height).
            cx_size (int): the num of concat patches along x(width).

        Returns:
            concated single image

        """

        ph, pw, nc = nppatch.shape[1:]
        reshaped1 = nppatch.reshape(cy_size, cx_size, ph, pw, nc)
        swapped = reshaped1.swapaxes(1, 2)
        return swapped.reshape(cy_size*ph, cx_size*pw, nc)

    def split_to_patch(self, npimg: np.ndarray,
                       patch_size: Union[int, Tuple[int, int]],
                       buffer_size: Union[int, Tuple[int, int]],
                       dtype: np.dtype = None) -> np.ndarray:
        """split images to patch
        

        Args:
            npimg (np.ndarray): input 3-d image.
            patch_size (Union[int, Tuple[int, int]]): patch size to split.
            buffer_size (Union[int, Tuple[int, int]]): overlap buffer size.
            dtype (np.dtype, optional): output dtype. Defaults to np.uint8.

        Returns:
            4-d np.ndarray: each splitted images are packed into first of the 4 dimension.
        """

        if dtype is None:
            dtype = npimg.dtype

        patch_x, patch_y = self._check_patch_axis(patch_size)

        xx, yy = self.get_patch_axis(
            npimg.shape[1], patch_x, npimg.shape[0], patch_y)

        return self.get_patch(npimg, patch_size, buffer_size, xx, yy, dtype=dtype)

    def get_patch(self, npimg: np.ndarray,
                  patch_size: Union[int, Tuple[int, int]],
                  buffer_size: Union[int, Tuple[int, int]],
                  xx: List[int], yy: List[int],
                  dtype: np.dtype = np.uint8):
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

    fields = ['standardize',
              'resize',
              'random_rotation',
              'random_flip_left_right',
              'random_flip_up_down',
              'random_shift',
              'random_zoom',
              'random_shear',
              'random_brightness',
              'random_saturation',
              'random_hue',
              'random_contrast',
              'random_crop',
              'central_crop',
              'random_noise',
              'random_blur',
              'random_blur_kernel',
              'interpolation',
              'dtype',
              'input_shape',
              'input_label_shape',
              'num_transforms',
              'training']
    params = namedtuple('params', ','.join(fields))
    params.__new__.__defaults__ = (None,)*len(fields)

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
                 input_shape: Tuple[Tuple[int, int, int, int], ...] = None,
                 input_label_shape: Tuple[int, int, int, int] = None,
                 num_transforms: int = 10000,
                 seeds=None,
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
                if true, images were casted to floatx automatically.
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
            random_zoom (Tuple[float, float], optional):random zoom ratios of an image.
                unit is width and height retios.
                random_zoom[0] is y-direction, random_zoom[1] is x-direction.
                Defaults to None.
            random_shear (Tuple[float, float], optional): randomly shear of image. unit is degree.
                random_shear[0] is y-direction(degrees), random_shear[1] is x-direction(degrees).
                Defaults to None.
            random_brightness (float, optional): maximum image delta brightness 
                range [-random_brightness, random_brightness). 
                The value delta is added to all components of the tensor image.
                image is converted to float and scaled appropriately 
                if it is in fixed-point representation, and 
                delta is converted to the same data type. 
                For regular images, delta should be in the range (-1,1),
                as it is added to the image in floating point representation, 
                where pixel values are in the [0,1) range.
                Defaults to None.
            random_saturation (Tuple[float, float], optional): maximum image saturation 
                factor range between [random_saturation[0],random_saturation[1]).
                The value saturation factor is multiplying to the saturation channel of images.
                Defaults to None.
            random_hue (float, optional): maximum delta hue of RGB images between [-random_hue, random_hue).
                max_delta must be in the interval [0, 0.5].
                Defaults to None.
            random_contrast (Tuple[float, float], optional): randomly adjust contrast of 
                RGB images by contrast factor 
                which lies between [random_contrast[0], random_contrast[1])
                result image is calculated by (x - mean) * contrast_factor + mean. 
                Defaults to None.
            random_crop (Tuple[int, int], optional): randomly crop image with size
                [height,width] = [random_crop[0], random_crop[1]].
                Defaults to None.
            central_crop (Tuple[int, int], optional): crop center of image with 
                size [height,width] = [central_crop[0], central_crop[1]].
                Defaults to None.
            random_noise (float, optional): add random gaussian noise. 
                random_noise value means sigma param of gaussian.
                Defaults to None.            
            random_blur (float, optional): add random gaussian blur. 
                the value means sigma param of gaussian.    
                random_blur generate sigma as uniform(0, random_blur) for every mini-batch
                random blur converts integer images to float images. Defaults to None.
            random_blur_kernel (float, optional): kernel size of gaussian random blur.
                Defaults to 3.
            interpolation (str, optional): interpolation method. nearest or bilinear
                Defaults to 'nearest'.
            clslabel (bool, optional): If false, labels are presumed to be the same 
                dimensions as the image and 
                apply the same geometric transformations to labels. Defaults to False.
            dtype (type, optional): tfaug cast input images to this dtype after
                geometric transformation.
                Defaults to None.
            input_shape (Tuple[Tuple[int, int, int, int],...], optional): input image 
                ((batch,y,x,channels),...)=(img1_shape, img2_shape, ...) dimensions. 
                when use DatasetCreator, you dont need this.
                To reduce CPU load by generating all transform matrices at first, 
                use this. 
                if you have multiple inputs, use nested list.
                Defaults to None.
            input_label_shape (Tuple[int, int, int, int]): input label 
                (batch,y,x,channels) dimensions. 
                To reduce CPU load by generating all transform matrices at first, 
                use this if label is segmentation. Defaults to None.
            num_transforms (int, optional): The number of transformation matrixes generated in advance. 
                when input_shape is used. Defaults to 10,000.
            seeds (Tuple(float), optional): Multiple random seeds. Each time you call, 
                the next seed is used in the sequence. 
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
        self._input_label_shape = input_label_shape
        self._num_transforms = num_transforms
        self._training = training
        if seeds is None:
            self._seeds = np.random.uniform(0, 2**32, size=int(2**16))
        else:
            self._seeds = seeds
        self._num_seeds = len(self._seeds)
        self._seed_idx = 0
        tf.random.set_seed(self._seeds[self._seed_idx])

        self._transform_active = (isinstance(self._random_rotation, tf.Tensor) or
                                  isinstance(self._random_zoom, tf.Tensor) or
                                  isinstance(self._random_shift, tf.Tensor) or
                                  isinstance(self._random_shear, tf.Tensor) or
                                  self._random_flip_left_right or
                                  self._random_flip_up_down or
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
            self._set_seed()

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
            flip_x = tf.ones([batch_size], dtype=float_type)
            flip_y = tf.ones([batch_size], dtype=float_type)

            if isinstance(self._resize, tf.Tensor):
                resize_fl = tf.cast(self._resize, float_type)
                # update scalling of zoom
                resize_factor_y = (resize_fl[0] / size_fl[0])
                resize_factor_x = (resize_fl[1] / size_fl[1])

            if self._training:
                if isinstance(self._random_shift, tf.Tensor):
                    shift_y += tf.random.uniform([batch_size], -
                                                 self._random_shift[0], self._random_shift[0],
                                                 seed=self._seeds[self._seed_idx])
                    shift_x += tf.random.uniform([batch_size], -
                                                 self._random_shift[1], self._random_shift[1],
                                                 seed=self._seeds[self._seed_idx])

                if isinstance(self._random_shear, tf.Tensor):
                    shear_tan = tf.tan(self._random_shear / 180 * math.pi)
                    shear_y += tf.random.uniform([batch_size], -
                                                 shear_tan[0], shear_tan[0],
                                                 seed=self._seeds[self._seed_idx])\
                        * resize_factor_y
                    shear_x += tf.random.uniform([batch_size], -
                                                 shear_tan[1], shear_tan[1],
                                                 seed=self._seeds[self._seed_idx])\
                        * resize_factor_x

                    shift_y += -(size_fl[0] * shear_y) / 2
                    shift_x += -(size_fl[1] * shear_x) / 2

                if isinstance(self._random_zoom, tf.Tensor):
                    zoom_y = tf.random.uniform(
                        [batch_size], -self._random_zoom[0], self._random_zoom[0],
                        seed=self._seeds[self._seed_idx]) \
                        * resize_factor_y
                    zoom_x = tf.random.uniform(
                        [batch_size], -self._random_zoom[1], self._random_zoom[1],
                        seed=self._seeds[self._seed_idx])\
                        * resize_factor_x

                    shift_y += -(size_fl[0] * zoom_y) / 2
                    shift_x += -(size_fl[1] * zoom_x) / 2

            if isinstance(self._resize, tf.Tensor):
                zoom_y += resize_factor_y - 1
                zoom_x += resize_factor_x - 1

            trans_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                [[(x[5]+1), x[3], x[1]],
                 [x[2], (x[4]+1), x[0]],
                 [0, 0, 1]], float_type),
                tf.transpose([shift_y, shift_x, shear_y, shear_x, zoom_y, zoom_x]))

            if self._training:
                if isinstance(self._random_rotation, tf.Tensor):
                    rad_theta = self._random_rotation / 180 * math.pi

                    rot = tf.random.uniform(
                        [batch_size], -rad_theta, rad_theta,
                        seed=self._seeds[self._seed_idx])
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

                if self._random_flip_left_right or\
                        self._random_flip_up_down:
                    if self._random_flip_left_right:
                        flip_x = (tf.cast(tf.random.uniform([batch_size], 0, 2,
                                                            dtype=tf.int32,
                                                            seed=self._seeds[self._seed_idx]),
                                          float_type) - 0.5) * 2
                        shift_x = size_fl[1] * (flip_x - 1) * -1/2

                    if self._random_flip_up_down:
                        flip_y = (tf.cast(tf.random.uniform([batch_size], 0, 2,
                                                            dtype=tf.int32,
                                                            seed=self._seeds[self._seed_idx]),
                                          float_type) - 0.5) * 2
                        shift_y = size_fl[0] * (flip_y - 1) * -1/2

                    flip_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                        [[x[2], 0, x[3]],
                         [0, x[0], x[1]],
                         [0, 0, 1]], float_type),
                        tf.transpose([flip_y, shift_y, flip_x, shift_x]))

                    trans_matrix = tf.keras.backend.batch_dot(
                        trans_matrix, flip_matrix)

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
        self._set_seed()

        if self._input_shape:
            image = tf.ensure_shape(image, [None, *self._input_shape[1:]])

        shape = tf.shape(image)
        batch_size = shape[0]

        # keep image channel dims
        last_image_dim = shape[-1]
        # last_label_dim = tf.shape(label)[-1]

        if label is not None and not self._clslabel:
            if self._input_label_shape:
                label = tf.ensure_shape(
                    label, [None, *self._input_label_shape[1:]])
            image = tf.concat([image, label], axis=3)

        if self._transform_active:

            if hasattr(self, "_Ms"):
                M = tf.gather(self._Ms, tf.cast(tf.random.uniform(
                    [batch_size], seed=self._seeds[self._seed_idx]
                )*self._num_transforms, tf.int32), axis=0)
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
                                                         tf.expand_dims(depth, -1)), axis=0),
                                         seed=self._seeds[self._seed_idx])

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
                    image, self._random_brightness,
                    seed=self._seeds[self._seed_idx])
            if self._random_saturation:
                image = tf.image.random_saturation(
                    image, *self._random_saturation,
                    seed=self._seeds[self._seed_idx])
            if self._random_hue:
                image = tf.image.random_hue(image, self._random_hue,
                                            seed=self._seeds[self._seed_idx])
            if self._random_contrast:
                image = tf.image.random_contrast(image, *self._random_contrast,
                                                 seed=self._seeds[self._seed_idx])
            if self._random_noise:
                noise_source = tf.random.normal(tf.shape(image),
                                                mean=0,
                                                stddev=1,
                                                seed=self._seeds[self._seed_idx])
                noise_scale = tf.random.uniform([batch_size, 1, 1, 1],
                                                -self._random_noise,
                                                self._random_noise,
                                                seed=self._seeds[self._seed_idx])
                noise = tf.cast(noise_scale * noise_source, image.dtype)
                image += noise
            if self._random_blur:
                sigmas = tf.random.uniform([batch_size], 0, self._random_blur,
                                           seed=self._seeds[self._seed_idx])
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

    def _set_seed(self):
        self._seed_idx += 1
        self._seed_idx %= self._num_seeds
        tf.random.set_seed(self._seeds[self._seed_idx])
    

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
