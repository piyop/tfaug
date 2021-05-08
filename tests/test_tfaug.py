# -*- coding: utf-8 -*-
"""
@license: MIT
@author: t.okuda
"""

import test_tfaug_tool as tool
from tfaug import AugmentImg, DatasetCreator, TfrecordConverter
import tensorflow_addons as tfa
import tensorflow as tf
import unittest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
import os
import math

DATADIR = r'testdata\tfaug'+os.sep


class TestTfaug(unittest.TestCase):

    def test_standardization(self):

        im = np.arange(3*4*5*3, dtype=np.uint8).reshape(3, 4, 5, 3)
        tn = tf.Variable(im)

        ret = AugmentImg().standardization(tn).numpy()

        max_axis = np.max(ret, axis=(1, 2, 3))
        mean_axis = np.mean(ret, axis=(1, 2, 3))

        assert np.allclose(mean_axis, 0), 'standardization failed'
        assert np.allclose(max_axis, 0.5), 'standardization failed'

    def test_dataset_from_path(self):

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
                        'random_saturation': [0.5, 2]}

        BATCH_SIZE = 2
        flist = [DATADIR+'Lenna.png'] * 10 * BATCH_SIZE
        labels = [0] * 10 * BATCH_SIZE

        ds = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE, **DATAGEN_CONF,
                            labeltype='class', training=True).\
            dataset_from_path(flist, labels)

        tool.plot_dsresult(ds.take(10), BATCH_SIZE, 10,
                           DATADIR+'test_ds_creator.png')

    def test_dataset_from_tfrecord(self):
            
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
                        'random_saturation': [0.5, 2]}

        BATCH_SIZE = 2
        flist = [DATADIR+'Lenna.png'] * 10 * BATCH_SIZE
        labels = [0] * 10 * BATCH_SIZE

        # test for classification
        path_tfrecord = DATADIR+'ds_from_tfrecord.tfrecord'
        TfrecordConverter().tfrecord_from_path_label(flist,
                                                     labels,
                                                     path_tfrecord)

        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE, **
                            DATAGEN_CONF, labeltype='class', training=True)
        ds, cnt = dc.dataset_from_tfrecords([path_tfrecord])

        rep_cnt = 0
        for img, label in iter(ds):
            rep_cnt += 1

        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"

        tool.plot_dsresult(ds.take(10), BATCH_SIZE, 10,
                           DATADIR+'test_ds_from_tfrecord.png')
        
        # test for segmentation
        path_tfrecord = DATADIR+'ds_from_tfrecord.tfrecord'
        TfrecordConverter().tfrecord_from_path_label(flist,
                                                     flist,
                                                     path_tfrecord)

        dc = DatasetCreator(BATCH_SIZE*10, BATCH_SIZE, **
                            DATAGEN_CONF, labeltype='segmentation', training=True)
        ds, cnt = dc.dataset_from_tfrecords([path_tfrecord])

        rep_cnt = 0
        for img, label in iter(ds):
            rep_cnt += 1

        assert rep_cnt == 10, "repetition count is invalid"
        assert img.shape[1:3] == random_crop_size, "crop size is invalid"
        assert label.shape[1:3] == random_crop_size, "crop size is invalid"

        tool.plot_dsresult(ds.take(10), BATCH_SIZE, 10,
                           DATADIR+'test_ds_from_tfrecord.png')


    def test_tfrecord_from_path(self):

        flist_imgs = [DATADIR+'Lenna.png'] * 10
        flist_seglabels = flist_imgs.copy()
        img_org = np.array(Image.open(flist_imgs[0]))
        clslabels = list(range(10))

        path_tfrecord = DATADIR+'test_tfrecord_from_path.tfrecord'
        TfrecordConverter().tfrecord_from_path_label(flist_imgs,
                                                     flist_seglabels,
                                                     path_tfrecord)

        # check segmentation label
        dc = DatasetCreator(1, 1, labeltype='segmentation', training=True)
        ds, imgcnt = dc.dataset_from_tfrecords([path_tfrecord])

        for i, (img, label) in enumerate(ds):
            assert (img == img_org).numpy().all(), 'image is changed'
            assert (label == img_org).numpy().all(), 'labels is changed'

        TfrecordConverter().tfrecord_from_path_label(flist_imgs,
                                                     clslabels,
                                                     path_tfrecord)

        # check
        dc = DatasetCreator(False, 1, labeltype='class', training=True)
        ds, datacnt = dc.dataset_from_tfrecords([path_tfrecord])

        for i, (img, label) in enumerate(ds):
            assert (img == img_org).numpy().all(), 'image is changed'
            assert (label.numpy() == clslabels[i]), 'label is changed'


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
            training=training)

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

    def adjust_img_range(self, img):
        if img.dtype == np.float16:
            img = img.astype(np.float32)
        max_axis = np.max(img, axis=(1, 2, 3))[:, None, None, None]
        min_axis = np.min(img, axis=(1, 2, 3))[:, None, None, None]
        return (img - min_axis) / (max_axis - min_axis)

    def test_augmentation(self):
        """
        test class AugmentImg

        Returns
        -------
        None.

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
                  'random_noise',
                  'interpolation',
                  'dtype',
                  'training']
        params = namedtuple('params', ','.join(fields),
                            defaults=(None,)*len(fields))

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

        cases = [
            params(  # test dtype
                standardize=True,
                resize=(300, 400),
                random_brightness=0.5,
                random_hue=0.01,
                random_contrast=[.1, .5],
                random_noise=100,
                interpolation='nearest',
                dtype=tf.float16,
                training=True),
            params(  # test y_shift
                standardize=True,
                random_rotation=0,
                random_flip_left_right=True,
                random_flip_up_down=False,
                random_shift=[256, 0],
                random_brightness=False,
                interpolation='nearest',
                training=True),
            params(  # test x_shift and rotation
                random_rotation=45,
                random_flip_left_right=False,
                random_flip_up_down=True,
                random_shift=(0, 256),
                random_brightness=False,
                random_saturation=(0.5, 2),
                interpolation='nearest',
                training=True),
            params(  # test crop and zoom
                standardize=True,
                random_rotation=45,
                random_flip_left_right=False,
                random_flip_up_down=False,
                random_shift=None,
                random_zoom=(0.8, 0.1),
                random_crop=(256, 512),
                interpolation='bilinear',
                training=True),
            params(  # test shear and color
                standardize=True,
                random_flip_left_right=False,
                random_flip_up_down=False,
                random_zoom=[0.1,0.1],
                random_shear=(10, 10),
                random_brightness=0.5,
                random_saturation=[0.5, 2],
                random_hue=0.01,
                random_contrast=[.1, .5],
                interpolation='bilinear',
                training=True),
            params(  # test train = False
                standardize=True,
                random_rotation=45,
                random_flip_left_right=True,
                random_flip_up_down=True,
                random_brightness=0.5,
                random_contrast=[.1, .5],
                random_crop=(256, 256),
                interpolation='nearest',
                training=False),
            params(  # test resize
                standardize=True,
                resize=(300, 400),
                random_rotation=45,
                random_zoom=(0.1, 0.1),
                random_contrast=[.1, .5],
                interpolation='nearest',
                training=True),
            params(  # test random_noise
                standardize=True,
                resize=(300, 400),
                random_brightness=0.5,
                random_hue=0.01,
                random_contrast=[.1, .5],
                random_noise=50,
                interpolation='nearest',
                training=True),
        ]

        for no, case in enumerate(cases):
            with self.subTest(case=case):
                print(case)

                func = AugmentImg(**case._asdict())

                img, lbl = func(image, label)

                if case.resize and not case.random_crop:
                    assert img.shape == [BATCH_SIZE] + list(case.resize) + [3]
                    assert lbl.shape == [BATCH_SIZE] + list(case.resize) + [3]
                elif case.random_crop:
                    assert img.shape == [BATCH_SIZE] + \
                        list(case.random_crop) + [3]
                    assert lbl.shape == [BATCH_SIZE] + \
                        list(case.random_crop) + [3]
                else:
                    assert img.shape == image.shape
                    assert lbl.shape == label.shape

                # adjust value range to display images : canceling standardize effect.
                img = self.adjust_img_range(img.numpy())
                lbl = self.adjust_img_range(lbl.numpy())

                tool.plot_dsresult(zip(zip(img, lbl), (None,)*len(img)), 2,
                                   BATCH_SIZE, DATADIR+'test_augmentation_caseno'+str(no)+'.png')

    def test_valid(self):

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
                          random_flip_left_right=True,
                          random_flip_up_down=True,
                          random_shift=(0.1, 0.1),
                          random_zoom=(0.1, 0.1),
                          random_brightness=False,
                          random_saturation=False,
                          random_crop=[256, 128],
                          training=training)

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
    # unittest.main()
    # TestTfaug().test_augmentation()