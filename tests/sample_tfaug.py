# -*- coding: utf-8 -*-
"""
@license: MIT
@author: t.okuda
"""


import os
import requests
import random
import math
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

from tfaug import TfrecordConverter, DatasetCreator, AugmentImg

DATADIR = 'testdata/tfaug/'


def quick_toy_sample():

    # source image and labels
    imgpaths = ['testdata/tfaug/Lenna.png'] * 10
    labels = np.random.randint(0, 255, 10)

    # configure and create dataset
    dataset = DatasetCreator(shuffle_buffer=10,
                             batch_size=2,
                             repeat=True,
                             standardize=True,  # add augmentation params here
                             training=True
                             ).from_path(imgpaths, labels)

    # define and compile the model
    mbnet = tf.keras.applications.MobileNetV2(include_top=True, weights=None)
    mbnet.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # learn the model
    mbnet.fit(dataset, epochs=10, steps_per_epoch=10)


def toy_example():

    # prepare inputs and labels
    batch_size = 2
    shuffle_buffer = 10
    filepaths = [DATADIR+'Lenna.png'] * 10
    class_labels = np.random.randint(0, 10, 10)

    # define tfrecord path
    path_record = DATADIR + 'multi_input.tfrecord'

    # generate tfrecords in a one-line
    TfrecordConverter().from_path_label(filepaths,
                                        class_labels,
                                        path_record)

    # define augmentation parameters
    aug_parms = {'random_rotation': 5,
                 'random_flip_left_right': True,
                 'random_shear': [5, 5],
                 'random_brightness': 0.2,
                 'random_crop': None,
                 'random_blur': [0.5, 1.5]}

    # set augmentation and learning parameters to dataset
    dc = DatasetCreator(shuffle_buffer, batch_size, **
                        aug_parms, repeat=True, training=True)
    # define dataset and number of dataset
    ds, imgcnt = dc.from_tfrecords(path_record)

    # define the handling of multiple inputs => just resize and concat
    # multiple inputs were named {'image_in0', 'image_in1' , ...} in inputs dictionary
    def concat_inputs(inputs, label):
        resized = tf.image.resize(inputs['image_in1'], (512, 512))
        concated = tf.concat([inputs['image_in0'], resized], axis=-1)
        # resized = tf.image.resize(concated, (224, 224))
        return concated, label
    ds = ds.map(concat_inputs)

    # define the model
    mbnet = tf.keras.applications.MobileNetV2(input_shape=[512, 512, 6],
                                              include_top=True,
                                              weights=None)

    mbnet.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # learn the model
    mbnet.fit(ds,
              epochs=10,
              steps_per_epoch=imgcnt//batch_size,)


def learn_multi_seginout_fromtfds():

    import tensorflow_datasets as tfds
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    BATCH_SIZE = 2
    RESIZE = [214, 214]
    # create dataset from tensorflow_dataset
    augprm = AugmentImg.params(random_crop=[64, 64],
                               random_contrast=[.5, 1.5])

    train = dataset['train']
    valid = dataset['test']

    def prepare_ds(train):

        def resize(dataset):
            img = tf.image.resize(dataset['image'], RESIZE)
            msk = tf.image.resize(dataset['segmentation_mask'], RESIZE)
            return (img, img, msk, msk)

        extracted = train.map(resize).batch(BATCH_SIZE)
        auged = DatasetCreator(
            10, BATCH_SIZE, **augprm._asdict()
        ).from_dataset(extracted,
                      'segmentation', 2)

        def cat(data):
            return ({'in1': data['image_in0'], 'in2': data['image_in1']},
                    tf.concat([data['label_in0'], data['label_in1']], axis=-1))

        return auged.map(cat)

    ds_train = prepare_ds(train)
    ds_valid = prepare_ds(valid)

    # define the model
    model = def_branch_unet(tuple(augprm.random_crop+[3]),
                            tuple(augprm.random_crop+[3]),
                            2)  # 2 - input and concated mask

    model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      from_logits=True),
                  metrics=['categorical_accuracy'])

    model.fit(ds_train,
              epochs=10,
              validation_data=ds_valid,
              steps_per_epoch=info.splits['train'].num_examples//BATCH_SIZE,
              validation_steps=info.splits['test'].num_examples//BATCH_SIZE)

    # model.evaluate(ds_valid,
    #                steps=valid_cnt//batch_size,
    #                verbose=2)


def def_down_stack(input_size):

    # define downstack model
    mbnet2 = tf.keras.applications.MobileNetV2(input_size,
                                               include_top=False,
                                               weights='imagenet')

    # Use the activations of these layers
    layer_names = [
        'block_16_project',      # 8x8
        'block_13_expand_relu',  # 16x16
        'block_6_expand_relu',   # 32x32
        'block_3_expand_relu',   # 64x64
        'block_1_expand_relu',   # 128x128
    ]
    mbnet2_outputs = [mbnet2.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=mbnet2.input, outputs=mbnet2_outputs)

    down_stack.trainable = False

    return down_stack


def def_branch_unet(input_size1, input_size2, output_filters):

    down_stack1 = def_down_stack(input_size1)
    down_stack2 = def_down_stack(input_size2)

    # define upstack
    upstack = [upsample(2**i) for i in range(7, 3, -1)]

    # define input
    inputs1 = tf.keras.layers.Input(input_size1)
    inputs2 = tf.keras.layers.Input(input_size2)

    # calc down stack
    skips1 = down_stack1(inputs1)
    skips2 = down_stack2(inputs2)
    x, skips1 = skips1[0], skips1[1:]
    x2 = skips2[0]
    x = tf.keras.layers.Concatenate()([x, x2])

    # calc up stack
    for up, skip in zip(upstack, skips1):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # output dimension
    x = upsample(output_filters)(x)

    # define output of the model
    return tf.keras.Model(inputs={'in1': inputs1, 'in2': inputs2}, outputs=x)


def lean_mnist():
    """
    tfaug classification example

    Returns
    -------
    None.

    """

    os.makedirs(DATADIR+'mnist', exist_ok=True)
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # save as tfrecord
    TfrecordConverter().from_ary_label(
        x_train, y_train, DATADIR+'mnist/train.tfrecord')
    TfrecordConverter().from_ary_label(
        x_test, y_test, DATADIR+'mnist/test.tfrecord')

    batch_size, shuffle_buffer = 25, 25
    # create training and validation dataset using tfaug:
    ds_train, train_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                          batch_size=batch_size,
                                          repeat=True,
                                          random_zoom=[0.1, 0.1],
                                          random_rotation=20,
                                          random_shear=[10, 10],
                                          random_blur=10,
                                          training=True)
                           .from_tfrecords([DATADIR+'mnist/train.tfrecord']))
    ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                          batch_size=batch_size,
                                          repeat=True,
                                          training=False)
                           .from_tfrecords([DATADIR+'mnist/test.tfrecord']))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    # learn model
    model.fit(ds_train,
              epochs=10,
              validation_data=ds_valid,
              steps_per_epoch=train_cnt//batch_size,
              validation_steps=valid_cnt//batch_size)

    # evaluation result
    model.evaluate(ds_valid,
                   steps=valid_cnt//batch_size,
                   verbose=2)


def learn_ade20k():

    crop_size = [256, 256]  # cropped input image size
    # original input image size
    batch_size = 5

    # donwload
    overlap_buffer = 256 // 4
    download_and_convert_ADE20k(crop_size, overlap_buffer)

    # define training and validation dataset using tfaug:
    tfrecords_train = glob(
        DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/training_*.tfrecords')
    ds_train, train_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                          batch_size=batch_size,
                                          repeat=True,
                                          standardize=True,
                                          random_zoom=[0.1, 0.1],
                                          random_rotation=10,
                                          random_shear=[10, 10],
                                          random_crop=crop_size,
                                          dtype=tf.float16,
                                          training=True)
                           .from_tfrecords(tfrecords_train))

    tfrecords_valid = glob(
        DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/validation_*.tfrecords')
    ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                          batch_size=batch_size,
                                          repeat=True,
                                          standardize=True,
                                          random_crop=crop_size,
                                          dtype=tf.float16,
                                          training=False)
                           .from_tfrecords(tfrecords_valid))

    # define model
    model = def_unet(tuple(crop_size+[3]), 151)  # 150class + padding area

    model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    model.fit(ds_train,
              epochs=10,
              validation_data=ds_valid,
              steps_per_epoch=train_cnt//batch_size,
              validation_steps=valid_cnt//batch_size)

    model.evaluate(ds_valid,
                   steps=valid_cnt//batch_size,
                   verbose=2)


def test_parse_tfrecord():

    tfexample_format = {"image": tf.io.FixedLenFeature([], dtype=tf.string),
                        "label": tf.io.FixedLenFeature([], dtype=tf.string)}

    def decoder(tfexamples):
        return [tf.map_fn(tf.image.decode_png, tfexamples[key], dtype=tf.uint8)
                if value.dtype == tf.string else tfexamples[key]
                for key, value in tfexample_format.items()]

    tfrecords_train = glob(
        DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/training_*.tfrecords')

    for tfrecord in tfrecords_train:
        # define dataset
        ds = tf.data.TFRecordDataset(
            tfrecord, num_parallel_reads=len(tfrecords_train))

        ds_train = (ds.batch(4)
                    .apply(tf.data.experimental.parse_example_dataset(tfexample_format))
                    .map(decoder)
                    .prefetch(AUTOTUNE))

        for piyo in tqdm(ds_train, total=1000//4):
            img, lbl = piyo


def def_unet(input_size, output_filters):

    # define downstack model
    mbnet2 = tf.keras.applications.MobileNetV2(input_size,
                                               include_top=False,
                                               weights='imagenet')

    # Use the activations of these layers
    layer_names = [
        'block_16_project',      # 8x8
        'block_13_expand_relu',  # 16x16
        'block_6_expand_relu',   # 32x32
        'block_3_expand_relu',   # 64x64
        'block_1_expand_relu',   # 128x128
    ]
    mbnet2_outputs = [mbnet2.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=mbnet2.input, outputs=mbnet2_outputs)

    down_stack.trainable = False

    # define upstack
    upstack = [upsample(2**i) for i in range(7, 3, -1)]

    # define input
    inputs = tf.keras.layers.Input(input_size)

    # calc down stack
    skips = down_stack(inputs)
    x, skips = skips[0], skips[1:]

    # calc up stack
    for up, skip in zip(upstack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # output dimension
    x = upsample(output_filters)(x)

    # define output of the model
    return tf.keras.Model(inputs=inputs, outputs=x)


def upsample(filters):
    upsample = tf.keras.Sequential()
    upsample.add(tf.keras.layers.UpSampling2D())
    upsample.add(tf.keras.layers.Conv2D(filters, 3, padding='same'))
    upsample.add(tf.keras.layers.BatchNormalization())

    return upsample


def download_and_convert_ADE20k(input_size, overlap_buffer):
    """
    Donload and Converts the ADE20k dataset into tfrecord format.
    """

    link = r'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
    dstdir = DATADIR+'ADE20k/'
    os.makedirs(dstdir, exist_ok=True)

    if not os.path.isfile(dstdir+'ADEChallengeData2016.zip'):
        print('start donloading ADE20k...', flush=True)
        with requests.get(link, stream=True) as response:
            total_size_in_bytes = int(
                response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte
            progress_bar = tqdm(total=total_size_in_bytes,
                                unit='iB', unit_scale=True)
            with open(dstdir+'ADEChallengeData2016.zip', 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
        assert total_size_in_bytes != 0 and progress_bar.n == total_size_in_bytes,\
            "download ADE20k failed"

    if len(glob(dstdir+'ADEChallengeData2016/images/validation/ADE_*.jpg')) != 2000:
        print('unzipping ADE20k...')
        from zipfile import ZipFile
        with ZipFile(dstdir+'ADEChallengeData2016.zip', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(dstdir)

    dstdir += 'ADEChallengeData2016/'

    print('convert grayscale images to RGB:', 'test')
    for dirname in ['training', 'validation']:
        imgs = glob(f'{dstdir}images/{dirname}/ADE_*.jpg')
        gray_idxs = [i for i in range(len(imgs)) if len(
            Image.open(imgs[i]).getbands()) < 3]
        for rmidx in gray_idxs:
            im = Image.open(imgs[rmidx])
            im = im.convert('RGB')
            im.save(imgs[rmidx])
            print('converted L to RGB:', imgs[rmidx])

    # plot random label sample
    print('start check ADE20k_label', 'test')
    check_ADE20k_label()

    converter = TfrecordConverter()

    patchdir = dstdir+'patch/'
    if len(glob(patchdir+'images/*/ADE_*_no*.jpg')) < 6e4:
        print('splitting imgs to patch...', flush=True)

        # split images into patch
        overlap_buffer = [overlap_buffer, overlap_buffer]
        for dirname in ['training', 'validation']:
            print('convert', dirname, 'into patch')
            os.makedirs(f'{patchdir}images/{dirname}', exist_ok=True)
            os.makedirs(f'{patchdir}annotations/{dirname}', exist_ok=True)
            srcimgs = glob(f'{dstdir}/images/{dirname}/ADE_*.jpg')
            for path in tqdm(srcimgs):
                im = np.array(Image.open(path))
                lb = np.array(Image.open(os.sep.join(
                    Path(path).parts[:-3] + ('annotations', dirname, Path(path).stem+'.png'))))

                img_patches = converter.split_to_patch(
                    im, input_size, overlap_buffer, dtype=np.uint8)
                lbl_pathces = converter.split_to_patch(
                    lb, input_size, overlap_buffer, dtype=np.uint8)

                basename = Path(path).stem
                for no, (img_patch, lbl_patch) in enumerate(zip(img_patches, lbl_pathces)):
                    Image.fromarray(img_patch).save(
                        f'{patchdir}images/{dirname}/{basename}_no{no}.jpg')
                    Image.fromarray(lbl_patch).save(
                        f'{patchdir}annotations/{dirname}/{basename}_no{no}.png')

    image_per_shards = 1000
    if len(glob(dstdir+'tfrecord/*_*.tfrecords')) != 101:
        print('convert ADE20k to tfrecord', flush=True)
        os.makedirs(dstdir+'tfrecord', exist_ok=True)

        for dirname in ['training', 'validation']:
            imgs = glob(f'{patchdir}/images/{dirname}/ADE_*.jpg')
            # shuffle image order
            random.shuffle(imgs)

            path_labels = [os.sep.join(
                Path(path).parts[:-3] + ('annotations', dirname, Path(path).stem+'.png'))
                for path in imgs]

            converter.from_path_label(imgs,
                                      path_labels,
                                      dstdir +
                                      f'tfrecord/{dirname}.tfrecords',
                                      image_per_shards)

    path_tfrecord = DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/validation_1.tfrecords'
    # check converted tfrecord
    dc = DatasetCreator(
        False, 10, training=True)
    ds, datacnt = dc.from_tfrecords([path_tfrecord])
    piyo = next(iter(ds.take(1)))
    plt.imshow(piyo[0][5])


def check_ADE20k_label():
    path_label = DATADIR + \
        'ADE20k/ADEChallengeData2016/annotations/validation/ADE_val_00000001.png'
    paths = glob(
        DATADIR+'ADE20k/ADEChallengeData2016/annotations/validation/ADE_val_*.png')

    # create palette
    colnum = math.ceil(math.pow(150, 1/3))
    colvals = list(range(255//colnum, 255, 255//colnum))
    palette = [[r, g, b] for r in colvals for g in colvals for b in colvals]
    palette = sum(palette, [])

    fig, axs = plt.subplots(5, 1, figsize=(50, 10))
    for ax in axs:
        path_label = random.choice(paths)
        print(path_label)
        npimg = np.array(Image.open(path_label))
        pimg = Image.fromarray(npimg, 'P')
        pimg.putpalette(palette)
        ax.imshow(pimg)


def aug_multi_input():
    # toy example for multiple inputs

    # prepare inputs and labels
    batch_size = 2
    shuffle_buffer = 10
    filepaths0 = [DATADIR+'Lenna.png'] * 10
    filepaths1 = [DATADIR+'Lenna_crop.png'] * 10
    labels = np.random.randint(0, 10, 10)

    # define tfrecord path
    path_record = DATADIR + 'multi_input.tfrecord'

    # generate tfrecords in a one-line
    TfrecordConverter().from_path_label(list(zip(filepaths0, filepaths1)),
                                        labels,
                                        path_record,
                                        image_per_shard=2)

    # define augmentation parameters
    aug_parms = {'standardize': False,
                 'random_rotation': 5,
                 'random_flip_left_right': True,
                 'random_zoom': [0.2, 0.2],
                 'random_shear': [5, 5],
                 'random_brightness': 0.2,
                 'random_crop': None,
                 'random_blur': [0.5, 1.5],
                 'num_transforms': 10}

    # define dataset
    dc = DatasetCreator(shuffle_buffer, batch_size, **
                        aug_parms, repeat=True, training=True)
    ds, imgcnt = dc.from_tfrecords(path_record)

    # define the handling of multiple inputs => just resize and concat
    # multiple inputs were named {'image_in0', 'image_in1' , ...} in inputs dictionary
    def concat_inputs(inputs, label):
        resized = tf.image.resize(inputs['image_in1'], (512, 512))
        concated = tf.concat([inputs['image_in0'], resized], axis=-1)
        # resized = tf.image.resize(concated, (224, 224))
        return concated, label
    ds = ds.map(concat_inputs)

    # define the model
    mbnet = tf.keras.applications.MobileNetV2(input_shape=[512, 512, 6],
                                              include_top=True,
                                              weights=None)

    mbnet.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # learn the model
    mbnet.fit(ds,
              epochs=10,
              steps_per_epoch=imgcnt//batch_size,)


if __name__ == '__main__':
    pass
    # lean_mnist()
    # learn_ade20k()
    # check_ADE20k_label()
    learn_multi_seginout_fromtfds()
