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
import test_tfaug_tool as tool

DATADIR = 'testdata/tfaug/'


def lean_mnist():
    """
    tfaug application for classification

    Returns
    -------
    None.

    """

    os.makedirs(DATADIR+'mnist', exist_ok=True)
    # load mnist dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # save as tfrecord
    TfrecordConverter().tfrecord_from_ary_label(
        x_train, y_train, DATADIR+'mnist/train.tfrecord')
    TfrecordConverter().tfrecord_from_ary_label(
        x_test, y_test, DATADIR+'mnist/test.tfrecord')

    batch_size, shuffle_buffer = 25, 25
    # create training and validation dataset using tfaug:
    ds_train, train_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                          batch_size=batch_size,
                                          label_type='class',
                                          repeat=True,
                                          random_zoom=[0.1, 0.1],
                                          random_rotation=20,
                                          random_shear=[10, 10],
                                          training=True)
                           .dataset_from_tfrecords([DATADIR+'mnist/train.tfrecord']))
    ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                          batch_size=batch_size,
                                          label_type='class',
                                          repeat=True,
                                          training=False)
                           .dataset_from_tfrecords([DATADIR+'mnist/test.tfrecord']))


    # constant reguralization
    ds_train = ds_train.map(lambda x, y: (x/255, y))
    ds_train = ds_valid.map(lambda x, y: (x/255, y))

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
                   steps=x_test.shape[0]//batch_size,
                   verbose=2)


def learn_ade20k():

    input_size = [256, 256]  # cropped input image size
    batch_size = 5

    # donwload
    download_and_convert_ADE20k(input_size)

    # define training and validation dataset using tfaug:
    tfrecords_train = glob(
        DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/training_*.tfrecords')
    ds_train, train_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                          batch_size=batch_size,
                                          label_type='segmentation',
                                          repeat=True,
                                          standardize=True,
                                          random_zoom=[0.1, 0.1],
                                          random_rotation=10,
                                          random_shear=[10, 10],
                                          random_crop=input_size,
                                          dtype=tf.float16,
                                          input_shape=[batch_size]+input_size+[3],#batch, y, x, channel
                                          training=True)
                           .dataset_from_tfrecords(tfrecords_train))

    tfrecords_valid = glob(
        DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/validation_*.tfrecords')
    ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                          batch_size=batch_size,
                                          label_type='segmentation',
                                          repeat=True,
                                          standardize=True,
                                          random_crop=input_size,
                                          dtype=tf.float16,
                                          training=False)
                           .dataset_from_tfrecords(tfrecords_valid))


    # define model
    model = def_unet(tuple(input_size+[3]), 151)  # 150class + padding area

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

        # ds = ds.shuffle(4)

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


def download_and_convert_ADE20k(input_size):
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

    # plot random label sample
    check_ADE20k_label()

    dstdir += 'ADEChallengeData2016/'
    converter = TfrecordConverter()

    patchdir = dstdir+'patch/'
    if len(glob(patchdir+'images/validation/ADE_val_*_no*.jpg')) != 2760:
        print('splitting imgs to patch...', flush=True)

        # split images into patch
        overlap_buffer = [input_size[0]//4, input_size[1]//4]
        for dirname in ['training', 'validation']:
            print('convert', dirname, 'into patch')
            os.makedirs(f'{patchdir}images/{dirname}', exist_ok=True)
            os.makedirs(f'{patchdir}annotations/{dirname}', exist_ok=True)
            srcimgs = glob(f'{dstdir}/images/{dirname}/ADE_*.jpg')
            for path in tqdm(srcimgs):
                im = np.array(Image.open(path))
                lb = np.array(Image.open(os.sep.join(
                    Path(path).parts[:-3] + ('annotations', dirname, Path(path).stem+'.png'))))
                x_borders = [size
                             for size in range(0, im.shape[1]-input_size[0], input_size[0])]
                y_borders = [size
                             for size in range(0, im.shape[0]-input_size[1], input_size[1])]

                img_patches = converter.get_patch(im, input_size, overlap_buffer,
                                                  x_borders, y_borders, dtype=np.uint8)
                lbl_pathces = converter.get_patch(lb, input_size, overlap_buffer,
                                                  x_borders, y_borders, dtype=np.uint8)

                basename = Path(path).stem
                for no, (img_patch, lbl_patch) in enumerate(zip(img_patches, lbl_pathces)):
                    Image.fromarray(img_patch).save(
                        f'{patchdir}images/{dirname}/{basename}_no{no}.jpg')
                    Image.fromarray(lbl_patch).save(
                        f'{patchdir}annotations/{dirname}/{basename}_no{no}.png')

    image_per_shards = 1000
    if len(glob(dstdir+'tfrecord/*_*.tfrecords')) != 30:
        print('convert ADE20k to tfrecord', flush=True)
        os.makedirs(dstdir+'tfrecord', exist_ok=True)

        for dirname in ['training', 'validation']:
            imgs = glob(f'{patchdir}/images/{dirname}/ADE_*.jpg')
            # shuffle image order
            random.shuffle(imgs)

            # write tfrecords
            for sti in tqdm(range(math.ceil(len(imgs)/image_per_shards))):
                path_tfrecord = dstdir+f'tfrecord/{dirname}_{sti}.tfrecords'
                path_labels = [os.sep.join(
                    Path(path).parts[:-3] + ('annotations', dirname, Path(path).stem+'.png'))
                    for path
                    in imgs[sti:sti+image_per_shards]]
                converter.tfrecord_from_path_label(imgs[sti:sti+image_per_shards],
                                                   path_labels,
                                                   path_tfrecord)

    path_tfrecord = DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/validation_1.tfrecords'
    # check converted tfrecord
    dc = DatasetCreator(
        False, 10, label_type='segmentation', training=True)
    ds, datacnt = dc.dataset_from_tfrecords([path_tfrecord])
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


def aug_multi_input_and_pyfunc():

    batch_size = 1
    filepaths = [DATADIR+'kanoko.png'] * 10

    # define tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices(
        tf.range(10)).repeat().batch(batch_size)

    # define augmentation
    aug_fun = AugmentImg(
        standardize=False, random_rotation=90, clslabel=False,
        training=True)

    # construct preprocessing function
    dtype = tf.int32

    def tf_img_preproc(filepaths1, filepaths2, aug_func):
        def preproc(image_nos):

            # augment only image here
            return (aug_func(tf.convert_to_tensor(tool.read_imgs([filepaths1[no]
                                                                  for no
                                                                  in image_nos]),
                                                  dtype=dtype)),
                    tf.convert_to_tensor(tool.read_imgs([filepaths2[no]
                                                         for no
                                                         in image_nos]),
                                         dtype=dtype))
        return preproc

    # pass aug_fun to 3rd arg
    func = tf_img_preproc(filepaths, filepaths.copy(), aug_fun)
    # py_function for multiple output

    def func(x): return tool.new_py_function(
        func, [x], {'image1': dtype, 'image2': dtype})

    # map preprocess and augmentation
    ds_aug = ds.map(func, num_parallel_calls=AUTOTUNE)

    # check augmented image
    fig, axs = plt.subplots(
        batch_size*2, 10, figsize=(10, batch_size*2), dpi=300)
    for i, in_dict in enumerate(iter(ds_aug.take(10))):
        for row in range(batch_size):
            axs[row*2, i].axis("off")
            axs[row*2, i].imshow(in_dict['image1'][row])
            axs[row*2+1, i].axis("off")
            axs[row*2+1, i].imshow(in_dict['image2'][row])

    plt.savefig(DATADIR+'aug_multi_input_and_pyfunc.png')

    # to learn a model
    # model.fit(ds_aug)


if __name__ == '__main__':
    pass
    # lean_mnist()
    learn_ade20k()
    # check_ADE20k_label()
