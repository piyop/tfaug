# Table of Contents

* [tfaug package](#tfaug-package)
* [Features](#features)
* [Dependancies](#dependancies)
   * [For test script](#for-test-script)
* [Supported Augmentations](#supported-augmentations)
* [Install](#install)
* [Document](#document)
* [Quick-Samples](#quick-samples)
   * [Classification Problem](#classification-problem)
      * [Convert Images and Labels to Tfrecord Format by TfrecordConverter()](#convert-images-and-labels-to-tfrecord-format-by-tfrecordconverter)
      * [Create Dataset by DatasetCreator()](#create-dataset-by-datasetcreator)
      * [Define and Learn Model Using Defined Datasets](#define-and-learn-model-using-defined-datasets)
   * [Segmentation Problem](#segmentation-problem)
      * [Convert Images and Labels to Tfrecord Format by TfrecordConverter()](#convert-images-and-labels-to-tfrecord-format-by-tfrecordconverter-1)
      * [Create Dataset by DatasetCreator()](#create-dataset-by-datasetcreator-1)
      * [Define and Learn Model Using Defined Datasets](#define-and-learn-model-using-defined-datasets-1)
      * [Adjust sampling ratios from multiple tfrecord files](#adjust-sampling-ratios-from-multiple-tfrecord-files)
   * [Use AugmentImg Directly](#use-augmentimg-directly)
      * [1. Initialize](#1-initialize)
      * [2. use in tf.data.map() after batch()](#2-use-in-tfdatamap-after-batch)


# tfaug package
Tensorflow >= 2 recommends to be feeded data by tf.data.Dataset.
This package supports creation of tf.data.Dataset (generator) and augmentation for image.

This package includes below 3 classes:
 * DatasetCreator - creator of tf.data.Dataset from tfrecords or image paths
 * TfrecordConverter - pack images and labels to tfrecord format (recommended format for better peformance)
 * AugmentImg - image augmentation class. This is used inside DatasetCreator implicitly or you can use it directly.

# Features
 * Augment input image and label image with same transformations at the same time.
 * Reduce cpu load by generating all transformation matrix at first. (use `input_shape` parameter at `DatasetCreator()` or `AugmentImg()`)
 * It could adjust sampling ratios from multiple tfrecord files. (use `ratio_samples` parameter at `DatasetCreator().dataset_from_tfrecords`) This is effective for class imbalance problems.
 * Augment on batch. It is more efficient than augment each image.
 * Use only tensorflow operators and builtin functions while augmentation. Because any other operations or functions (e.g. numpy functions) may be bottleneck of learning. [mentined here](https://www.tensorflow.org/guide/function).

# Dependancies
 * Python >= 3.5
 * tensorflow >= 2.0
 * tensorflow-addons
## For test script
 * pillow
 * numpy
 * matplotlib

# Supported Augmentations
 * standardize
 * resize
 * random_rotation
 * random_flip_left_right
 * random_flip_up_down
 * random_shift
 * random_zoom
 * random_shear
 * random_brightness
 * random_saturation
 * random_hue
 * random_contrast
 * random_crop
 * random_noise
 * random_blur
 
# Install
python -m pip install git+https://github.com/piyop/tfaug

# Document
**The descriptions of each class and function can be found at [docs/tfaug.md](https://github.com/piyop/tfaug/tree/master/docs/tfaug.md)**

# Quick Samples

Simple Classification and Segmentation Usage is shown below. 
Whole ruunable codes is in sample_tfaug.py

## Classification Problem
Download, convert to tfrecord and learn MNIST dataset.
Below examples are part of `learn_mnist()` in sample_tfaug.py

Import tfaug and define directory to be store data.
```Python
from tfaug import TfrecordConverter, DatasetCreator, AugmentImg
DATADIR = 'testdata/tfaug/'
```

Load MNIST dataset using tensorflow.
```Python
os.makedirs(DATADIR+'mnist', exist_ok=True)
# load mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

### Convert Images and Labels to Tfrecord Format by TfrecordConverter()
Convert training and validation(test) images and labels into Tfrecord format by TfrecordConverter.
Therefore, Tensorflow could be load data from Tfrecord format with least overhead and parallel reading. 
```Python
# save as tfrecord
TfrecordConverter().tfrecord_from_ary_label(
    x_train, y_train, DATADIR+'mnist/train.tfrecord')
TfrecordConverter().tfrecord_from_ary_label(
    x_test, y_test, DATADIR+'mnist/test.tfrecord')
 ```

### Create Dataset by DatasetCreator()
Create and apply augmentation to training and validation Tfrecords by DatasetCreator.
Set image augmentation params to DatasetCreator constractor.
```Python
batch_size, shuffle_buffer = 25, 25
# create training and validation dataset using tfaug:
ds_train, train_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                      batch_size=batch_size,
                                      repeat=True,
                                      random_zoom=[0.1, 0.1],
                                      random_rotation=20,
                                      training=True)
                       .dataset_from_tfrecords([DATADIR+'mnist/train.tfrecord']))
ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                      batch_size=batch_size,
                                      repeat=True,
                                      training=False)
                       .dataset_from_tfrecords([DATADIR+'mnist/test.tfrecord']))

```

### Define and Learn Model Using Defined Datasets
Define Model
```Python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)])

model.compile(optimizer=tf.keras.optimizers.Adam(0.002),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['sparse_categorical_accuracy'])
```

Learn Model by `model.fit()`. This accepts training and validation `tf.data.Dataset` which is created by DatasetCreator.
`model.fit()` needs number of training and validation iterations per epoch.
```Python
# learn model
model.fit(ds_train,
          epochs=10,
          validation_data=ds_valid,
          steps_per_epoch=train_cnt//batch_size,
          validation_steps=valid_cnt//batch_size)
```

Evaluation
```Python
# evaluation result
model.evaluate(ds_valid,
               steps=valid_cnt//batch_size,
               verbose=2)
```

## Segmentation Problem
Download ADE20k dataset and convert to the tfrecord
Below examples are part of `learn_mnist()` in sample_tfaug.py

First, we set input image size and batch size for model
```Python
crop_size = [256, 256]  # cropped input image size
overlap_buffer = 256 // 4
batch_size = 5
```

Download and convert ADE20k dataset to tfrecord by defined function download_and_convertADE20k() in sample_tfaug.py
```Python
# donwload
download_and_convert_ADE20k(crop_size, overlap_buffer)
```

### Convert Images and Labels to Tfrecord Format by TfrecordConverter()
In download_and_convertADE20k(), split original images to patch image by `TfrecordConverter.get_patch()`
Though ADE20k images have not same image size, tensorflow model input should be the exactly same size.
```Python
converter = TfrecordConverter()

~~~~~~~~~~~~~~some codes~~~~~~~~~~~~~~~~~~~~~~

img_patches = converter.get_patch(im, input_size, overlap_buffer,
                                  x_borders, y_borders, dtype=np.uint8)
lbl_pathces = converter.get_patch(lb, input_size, overlap_buffer,
                                  x_borders, y_borders, dtype=np.uint8)
```

Save images and labels to separated tfrecord files by shards. <br/>
Shards separation is recommended. Tensorflow read images from each tfrecord files to buffer and shuffle it.
Therefore, multiple tfrecord files improve the randomness of input while learning.
 ```Python 
image_per_shards = 1000

~~~~~~~~~~~~~~some codes~~~~~~~~~~~~~~~~~~~~~~

converter.tfrecord_from_path_label(imgs,
                                   path_labels,
                                   dstdir +
                                   f'tfrecord/{dirname}.tfrecords',
                                   image_per_shards)

 ```

### Create Dataset by DatasetCreator()
After generate tfrecord files by `TfrecordConverter.tfrecord_from_path_label`, create training and validation dataset from these tfrecords by DatasetCreator.<br/>
If you use `input_shape` param in `DatasetCreator()` like below, `AugmentImge()` generate all transformation matrices when __init__() is called. It reduces CPU load while learning. 

```Python
# input batch, y, x, channel
input_shape = [batch_size, 
              crop_size[0]+2*overlap_buffer,
              crop_size[1]+2*overlap_buffer, 
              3]
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
                                      random_crop=input_size,
                                      dtype=tf.float16,
                                      input_shape=input_shape,
                                      training=True)
                       .dataset_from_tfrecords(tfrecords_train))

tfrecords_valid = glob(
    DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/validation_*.tfrecords')
ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                      batch_size=batch_size,
                                      repeat=True,
                                      standardize=True,
                                      random_crop=input_size,
                                      dtype=tf.float16,
                                      training=False)
                       .dataset_from_tfrecords(tfrecords_valid))

```

### Define and Learn Model Using Defined Datasets
Last step is define and fit and evaluate Model.
```Python
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
```

### Adjust sampling ratios from multiple tfrecord files
If number of images in each class are significantly imvalanced, you may want adjust sampling ratios from each class.
`DatasetCreator.dataset_from_tfrecords` could accepts sampling ratios. <br/>
In that case, you must use 2 dimensional nested list representing tfrecord files to `path_records` in `DatasetCreator.dataset_from_tfrecords()` and assign `sampling_ratios` parameter for every 1 dimensional lists in 2 dimensional `path_records`.
A simple example was written in test_tfaug.py like below:
```python
dc = DatasetCreator(5, 10,
                    label_type='class',
                    repeat=False,
                    **DATAGEN_CONF,  training=True)
ds, cnt = dc.dataset_from_tfrecords([[path_tfrecord_0, path_tfrecord_0],
                                     [path_tfrecord_1, path_tfrecord_1]],
                                    ratio_samples=np.array([1,10],dtype=np.float32))
```



## Use AugmentImg Directly 
Above examples ware create tf.data.Dataset by DatasetCreator. If you need to control your dataflow in other way, you could use AugmentImage Directly

### 1. Initialize
```python  
from tfaug import AugmentImg 
#set your augment parameters below:
arg_fun = AugmentImg(standardize=False,
                      random_rotation=5, 
                      random_flip_left_right=True,
                      random_flip_up_down=True, 
                      random_shift=(.1,.1), 
                      random_zoom=(.1,.1),
                      random_shear=(.1,.1),
                      random_brightness=.2,
                      random_saturation=None,
                      random_hue=.2,
                      random_contrast=(.2,.5),
                      random_crop=256,
                      interpolation='nearest'
                      clslabel=True,
                      training=True) 
 
```

### 2. use in tf.data.map() after batch()
```python 
ds=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(image),
                      tf.data.Dataset.from_tensor_slices(label))) \
                    .shuffle(BATCH_SIZE*10).batch(BATCH_SIZE).map(arg_fun)
model.fit(ds)
```

