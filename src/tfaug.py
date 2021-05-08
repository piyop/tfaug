# -*- coding: utf-8 -*-
"""
@license: MIT
@author: t.okuda
"""

import math
import io
import os
import json
import numpy as np
from tqdm import tqdm

from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.data.experimental import AUTOTUNE
from typing import Tuple, Callable


class DatasetCreator():

    def __init__(self,
                 shuffle_buffer: int,
                 batch_size: int,
                 label_type: str,
                 **kwargs: dict):
        self._shuffle_buffer = shuffle_buffer
        self._batch_size = batch_size
        self._datagen_confs = kwargs

        self._labeltype = label_type
        if self._labeltype == 'class':
            label_dtype = tf.int64
            self._clslabel = True
        elif self._labeltype == 'segmentation':
            label_dtype = tf.string
            self._clslabel = False
        else:
            assert False, 'undifined labeltype:'+label_type

        self._tfexample_format = {"image": tf.io.FixedLenFeature([], dtype=tf.string),
                                  "label": tf.io.FixedLenFeature([], dtype=label_dtype)}

    def dataset_from_path(self, img_paths, labels=None):

        def decode_png(path_img):
            return tf.image.decode_png(tf.io.read_file(path_img))

        aug = AugmentImg(**self._datagen_confs, clslabel=self._clslabel)

        ds_img = tf.data.Dataset.from_tensor_slices(
            img_paths).map(decode_png, num_parallel_calls=AUTOTUNE)

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

        batch = zipped.batch(self._batch_size)

        return batch.map(aug, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    def dataset_from_tfrecords(self, path_tfrecords):

        def decoder(tfexamples):
            return [tf.map_fn(tf.image.decode_png, tfexamples[key], dtype=tf.uint8)
                    if value.dtype == tf.string else tfexamples[key]
                    for key, value in self._tfexample_format.items()]

        # define augmentation
        aug_fun = AugmentImg(**self._datagen_confs, clslabel=self._clslabel)

        # define dataset
        ds = tf.data.TFRecordDataset(
            path_tfrecords, num_parallel_reads=len(path_tfrecords))
        if self._shuffle_buffer:
            ds = ds.shuffle(self._shuffle_buffer)

        ds_aug = (ds.batch(self._batch_size)
                  .apply(tf.data.experimental.parse_example_dataset(self._tfexample_format))
                  .map(decoder, num_parallel_calls=AUTOTUNE)
                  .map(aug_fun, num_parallel_calls=AUTOTUNE)
                  .prefetch(AUTOTUNE))

        # get dataset counts
        num_img = 0
        for path_tfrecord in path_tfrecords:
            with open(os.path.splitext(path_tfrecord)[0]+'.json') as fp:
                fileds = json.load(fp)
                num_img += fileds['imgcnt']

        return ds_aug, num_img


class TfrecordConverter():

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

    def tfrecord_from_path_label(self, path_imgs, labels, path_out):
        self.save_png_label_tfrecord(
            path_imgs, labels, path_out, lambda x: np.array(Image.open(x)))

    def tfrecord_from_ary_label(self, ary, labels, path_out):
        self.save_png_label_tfrecord(
            ary, labels, path_out, lambda x: np.array(x))

    def save_png_label_tfrecord(self, imgs, labels, path_out, reader_func):

        seg_label = True if isinstance(labels[0], str) else False
        if seg_label:
            def label_feature(ilabel): return self._bytes_feature(
                self.np_to_pngstr(ilabel))
        else:
            def label_feature(ilabel): return self._int64_feature(ilabel)

        def image_example(iimg, ilabel):
            feature = {'image': self._bytes_feature(self.np_to_pngstr(iimg)),
                       'label': label_feature(ilabel)}
            return tf.train.Example(features=tf.train.Features(feature=feature))

        # save tfrecord
        with tf.io.TFRecordWriter(path_out) as writer:
            for (img, label) in tqdm(zip(imgs, labels),
                                     total=len(imgs),
                                     leave=False):
                img = reader_func(img)
                if seg_label:
                    ext = os.path.splitext(label)[1]
                    assert ext in ['.jpg', '.JPG', '.png', '.PNG', '.bmp'],\
                        "file extention is imcompatible:"+label
                    label = reader_func(label)
                # use same image as msk
                writer.write(image_example(img, label).SerializeToString())

        # save datalen to json
        with open(os.path.splitext(path_out)[0]+'.json', 'w') as file:
            json.dump({'imgcnt': len(imgs)}, file)

    def get_paded_np(self, npimg, x1, y1, x2, y2):
        """
        get image which had cropped and paded out of input npimg size

        Parameters
        ----------
        npimg : 2d np.ndarray()
            input ary
        x1 : int
            left point of npimg. if left padding is needed, this value must be negative.
        y1 : int
            top point of npimg. if top padding is needed, this value must be negative.
        x2 : int
            right point of npimg. if right padding is needed, x2 > npimg.shape[1]
        y2 : int
            bottom point of npimg. if bottom padding is needed, y2 > npimg.shape[0]

        Returns
        -------
        TYPE
            paded image if needed.

        """

        if npimg.ndim == 2:
            len_y, len_x = npimg.shape
        elif npimg.ndim == 3:
            len_y, len_x = npimg.shape[:2]
        else:
            raise NotImplementedError('npimg.ndim must 2<=ndim<=3')

        if x1 < 0 or x2 > len_x or y1 < 0 or y2 > len_y:
            size_x = x2 - x1
            size_y = y2 - y1

            offset_left = - x1 if x1 < 0 else 0
            offset_right = x2 - len_x if x2 > len_x else 0
            offset_top = - y1 if y1 < 0 else 0
            offset_bottom = y2 - len_y if y2 > len_y else 0

            if npimg.ndim == 2:
                npary = np.zeros((size_y, size_x))
            elif npimg.ndim == 3:
                npary = np.zeros((size_y, size_x, npimg.shape[2]))

            npary[offset_top:size_y - offset_bottom, offset_left:size_x - offset_right, Ellipsis] = \
                npimg[y1 + offset_top:y2 - offset_bottom, x1 +
                      offset_left:x2 - offset_right, Ellipsis]

            return npary
        else:
            return npimg[y1:y2, x1:x2, Ellipsis]

    def get_patch(self, npimg, patch_size, buffer_size, xx, yy, dtype=np.uint8):
        """
        split npimg to patch
        """

        if isinstance(patch_size, list) and len(patch_size) > 1:
            patch_x, patch_y = patch_size[0], patch_size[1]
        else:
            patch_x, patch_y = patch_size, patch_size

        if isinstance(buffer_size, list) and len(buffer_size) > 1:
            buffer_x, buffer_y = buffer_size[0], buffer_size[1]
        else:
            buffer_x, buffer_y = buffer_size, buffer_size

        img_size_y = 2*buffer_x + patch_x
        img_size_x = 2*buffer_y + patch_y

        # get paded array
        all_image = [self.get_paded_np(npimg,
                                       x - buffer_x,
                                       y - buffer_y,
                                       x + buffer_x + img_size_x,
                                       y + buffer_y + img_size_y)
                     for y in yy for x in xx]

        allnp = np.array(all_image, dtype=dtype)
        del all_image
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
                 random_noise: float = None,
                 interpolation: str = 'nearest',
                 inshape: Tuple[int, int, int, int] = None,
                 clslabel: bool = False,
                 dtype: type = None,
                 training: bool = False) \
            -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        __init__() sets the parameters for augmantation.

        When __call__() can take not only input image but also label image.
        Image and label image will be augmanted with same transformation at the same time.
        However, label image is not augmanted by random_brightness, random_saturation, standardize

        This augmantation is executed on batch images. Input image should be 4d Tensor(batch, x, y, channel)
        Image x and y size is same as premise.

        If training == False, this class will not augment image except standardize.

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
            randomely adjust y and x directional shear degree
            random_shear[0] : y-direction, random_shear[1] : x-direction
            The default is None.
        random_brightness : float, optional
            randomely adjust image brightness range
            [-max_delta, max_delta).
             The default is None.
        random_saturation : Tuple[float, float], optional
            randomely adjust image brightness range between [lower, upper].
            The default is None.
        random_hue : float, optional
            randomely adjust hue of RGB images between [-random_hue, random_hue]
        random_contrast : Tuple[float, float], optional
            randomely adjust contrast of RGB images between [random_contrast[0], random_contrast[1]]
        random_crop : int, optional
            randomely crop image with size [x,y] = [random_crop_height, random_crop_width].
            The default is None.
        random_noise : float, optional
            add random gausian noise. random_noise value mean sigma param of gaussian.
        interpolation : str, The default is nearest.
            interpolation method. nearest or bilinear
        clslabel : bool, The default is False
            If false, labels are presumed to be the same dimension as the image
        dtype : tf.Dtype, The default is None
            tfaug cast input images to this dtype after geometric transformation.
        training : bool, optional
            If false, this class don't augment image except standardize.
            The default is False.

        Returns
        -------
        class instance : Callable[[tf.Tensor, tf.Tensor, bool], Tuple[tf.Tensor,tf.Tensor]]

        """

        self._standardize = standardize

        (self._resize, self._random_rotation, self._random_shift, self._random_zoom,
         self._random_shear, self._random_crop) = None, None, None, None, None, None

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
        self._random_noise = random_noise
        self._interpolation = interpolation
        self._clslabel = clslabel
        self._dtype = dtype
        self._training = training

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

        if (isinstance(self._random_rotation, tf.Tensor) or
            isinstance(self._random_zoom, tf.Tensor) or
            isinstance(self._random_shift, tf.Tensor) or
                isinstance(self._random_shear, tf.Tensor)):

            in_size = imgshape[1:3]
            batch_size = imgshape[0]

            size_fl = tf.cast(in_size, dtype=tf.float32)

            trans_matrix = tf.eye(3, 3, [imgshape[0]], dtype=tf.float32)

            shift_y = tf.zeros([batch_size], dtype=tf.float32)
            shift_x = tf.zeros([batch_size], dtype=tf.float32)
            shear_y = tf.zeros([batch_size], dtype=tf.float32)
            shear_x = tf.zeros([batch_size], dtype=tf.float32)
            zoom_y = tf.zeros([batch_size], dtype=tf.float32)
            zoom_x = tf.zeros([batch_size], dtype=tf.float32)

            if isinstance(self._random_shift, tf.Tensor):
                shift_y += tf.random.uniform([batch_size], -
                                             self._random_shift[0], self._random_shift[0])
                shift_x += tf.random.uniform([batch_size], -
                                             self._random_shift[1], self._random_shift[1])

            if isinstance(self._random_shear, tf.Tensor):
                shear_tan = tf.tan(self._random_shear / 180 * math.pi)
                shear_y += tf.random.uniform([batch_size], -
                                             shear_tan[0], shear_tan[0])
                shear_x += tf.random.uniform([batch_size], -
                                             shear_tan[1], shear_tan[1])

                shift_y += -(size_fl[0] * shear_y) / 2
                shift_x += -(size_fl[1] * shear_x) / 2

            if isinstance(self._random_zoom, tf.Tensor):
                zoom_y = tf.random.uniform(
                    [batch_size], -self._random_zoom[0], self._random_zoom[0])
                zoom_x = tf.random.uniform(
                    [batch_size], -self._random_zoom[1], self._random_zoom[1])

                shift_y += -(size_fl[0] * zoom_y) / 2
                shift_x += -(size_fl[1] * zoom_x) / 2

            trans_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                [[x[5]+1, x[3], x[1]],
                 [x[2], x[4]+1, x[0]],
                 [0, 0, 1]], tf.float32),
                tf.transpose([shift_y, shift_x, shear_y, shear_x, zoom_y, zoom_x]))

            if isinstance(self._random_rotation, tf.Tensor):
                rad_theta = self._random_rotation / 180 * math.pi
                rot = tf.random.uniform([batch_size], -rad_theta, rad_theta)
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

    def _augmentation(self, image, label=None, train=False):
        """

        Parameters
        --------------------
        image : 4d tensor
            input image
        label : 4d tensor, optional,
            image fort ture label
        train : bool, optional,
            training or not
        """

        batch_size = tf.shape(image)[0]

        # keep image channel dims
        last_image_dim = tf.shape(image)[-1]

        if label is not None and not self._clslabel:
            image = tf.concat([image, label], axis=3)

        if isinstance(self._resize, tf.Tensor):
            image = tf.image.resize(image, self._resize, self._interpolation)
            
        in_size = tf.shape(image)[1:3]

        if train:

            if self._random_flip_left_right:
                image = tf.image.random_flip_left_right(image)

            if self._random_flip_up_down:
                image = tf.image.random_flip_up_down(image)

            if (isinstance(self._random_rotation, tf.Tensor) or
                isinstance(self._random_zoom, tf.Tensor) or
                isinstance(self._random_shift, tf.Tensor) or
                    isinstance(self._random_shear, tf.Tensor)):

                M = self._get_transform(tf.shape(image))

                # geometric transform
                image = tfa.image.transform(
                    image, M, interpolation=self._interpolation)

        if isinstance(self._random_crop, tf.Tensor):
            if train:
                depth = tf.shape(image)[3]
                image = tf.image.random_crop(image,
                                             size=tf.concat((tf.expand_dims(batch_size, -1),
                                                             self._random_crop,
                                                             tf.expand_dims(depth, -1)), axis=0))
            else:
                # central crop
                offset_height = (in_size[0] - self._random_crop[0]) // 2
                target_height = self._random_crop[0]
                offset_width = (in_size[1] - self._random_crop[1]) // 2
                target_width = self._random_crop[1]
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

        if train:
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
                                                0, self._random_noise)
                noise = tf.cast(noise_scale * noise_source, image.dtype)
                image += noise

        if self._standardize:
            # tf.image.per_image_standardization have a bug #33892
            # image = tf.image.per_image_standardization(image)
            image = self.standardization(image)

        if label is not None:
            return image, label
        else:
            return image

    def standardization(self, image):
        # num_dims
        axs = tf.constant((1, 2, 3))
        ndim = tf.rank(image)
        max_axis = tf.math.reduce_max(image, axs[:ndim-1], keepdims=True)
        min_axis = tf.math.reduce_min(image, axs[:ndim-1], keepdims=True)
        return (image - min_axis) / (max_axis - min_axis) - 0.5
