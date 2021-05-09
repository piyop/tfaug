# tfaug package
Tensorflow create tf.data.Dataset and image augmentation support classes.

This package include below 3 classes:
 * DatasetCreator - creator of tf.data.Dataset from tfrecords or image paths
 * TfrecordConverter - pack images and labels to tfrecord format
 * AugmentImg - image augmentation class which is used inside DatasetCreator implicitly.

## Features
 * augment input image and label image with same transformations at the same time.
 * augment on batch which is more efficient than augment each image.
 * use only tensorflow operators and basic statments and functions. Because any other operations or functions (e.g. numpy functions) cause limitation on multiprocess augmentation while using @tf.function to get a better peformance [as mentined here](https://www.tensorflow.org/guide/function).

## Dependancies
 * Python >= 3.5
 * tensorflow >= 2.0
 * tensorflow-addons
### for test srcipt
 * pillow
 * numpy
 * matplotlib

## Supported Augmentations:
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
 
## Install
python -m pip install git+https://github.com/piyop/tfaug

## Samples

Some Classification and Segmentation Usage is shown here. 
Whole ruunable codes is in sample_tfaug.py

#### Classification Problem
Download, convert to tfrecord and learn MNIST dataset.
Below examples are part of learn_mnist() in sample_tfaug.py

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

Convert training and validation(test) images and labels into Tfrecord format by TfrecordConverter.
Therefore, Tensorflow could be load data from Tfrecord format with least overhead and parallel reading. 
```Python
# save as tfrecord
TfrecordConverter().tfrecord_from_ary_label(
    x_train, y_train, DATADIR+'mnist/train.tfrecord')
TfrecordConverter().tfrecord_from_ary_label(
    x_test, y_test, DATADIR+'mnist/test.tfrecord')
 ```

Create and apply augmentation to training and validation Tfrecords by DatasetCreator.
For the classification problem, use label_type = 'class' for DatasetCreator constractor.
Set image augmentation params to DatasetCreator constractor.
```Python
batch_size, shuffle_buffer = 25, 25
# create training and validation dataset using tfaug:
ds_train, train_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                      batch_size=batch_size,
                                      label_type='class',
                                      random_zoom=[0.1, 0.1],
                                      random_rotation=20,
                                      random_shear=[10, 10],
                                      training=True)
                       .dataset_from_tfrecords([DATADIR+'mnist/train.tfrecord']))
ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=shuffle_buffer,
                                      batch_size=batch_size,
                                      label_type='class',
                                      training=False)
                       .dataset_from_tfrecords([DATADIR+'mnist/test.tfrecord']))

# add repeat operation
ds_train, ds_valid = ds_train.repeat(), ds_valid.repeat()
```

Add constant reguralization to training and validation datasets.
```Python
# constant reguralization
ds_train = ds_train.map(lambda x, y: (x/255, y))
ds_train = ds_valid.map(lambda x, y: (x/255, y))
```

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

Learn Model by model.fit(). This accepts training and validation tf.data.Dataset which is created by DatasetCreator.
model.fit() needs number of training and validation iterations per epoch.
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
               steps=x_test.shape[0]//batch_size,
               verbose=2)
```

#### Segmentation Problem
Download ADE20k dataset and convert to the tfrecord
Below examples are part of learn_mnist() in sample_tfaug.py

First, we set input image size and batch size for model
```Python
input_size = [256, 256]  # cropped input image size
batch_size = 5
```

Download and convert ADE20k dataset to tfrecord by defined function download_and_convertADE20k() in sample_tfaug.py
```Python
# donwload
download_and_convert_ADE20k(input_size)
```

In download_and_convertADE20k(), split original images to patch image by TfrecordConverter.get_patch()
Though ADE20k images have not same image size, tensorflow model input should be exactly same size.
```Python
converter = TfrecordConverter()

~~~~~~~~~~~~~~some codes~~~~~~~~~~~~~~~~~~~~~~

img_patches = converter.get_patch(im, input_size, overlap_buffer,
                                  x_borders, y_borders, dtype=np.uint8)
lbl_pathces = converter.get_patch(lb, input_size, overlap_buffer,
                                  x_borders, y_borders, dtype=np.uint8)
```

Save images and labels as separated tfrecord format. 
 ```Python 
image_per_shards = 1000

~~~~~~~~~~~~~~some codes~~~~~~~~~~~~~~~~~~~~~~

converter.tfrecord_from_path_label(imgs[sti:sti+image_per_shards],
                                   path_labels,
                                   path_tfrecord)
 ```

After generate tfrecord files by TfrecordConverter.tfrecord_from_path_label, 
For classification problem, use label_type = 'segmentation' for constractor of the DatasetCreator.
create training and validation dataset from thease by DatasetCreator
```Python
# define training and validation dataset using tfaug:
tfrecords_train = glob(
    DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/training_*.tfrecords')
ds_train, train_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                      batch_size=batch_size,
                                      label_type='segmentation',
                                      standardize=True,
                                      random_zoom=[0.1, 0.1],
                                      random_rotation=10,
                                      random_shear=[10, 10],
                                      random_crop=input_size,
                                      dtype=tf.float16,
                                      training=True)
                       .dataset_from_tfrecords(tfrecords_train))

tfrecords_valid = glob(
    DATADIR+'ADE20k/ADEChallengeData2016/tfrecord/validation_*.tfrecords')
ds_valid, valid_cnt = (DatasetCreator(shuffle_buffer=batch_size,
                                      batch_size=batch_size,
                                      label_type='segmentation',
                                      standardize=True,
                                      random_crop=input_size,
                                      dtype=tf.float16,
                                      training=False)
                       .dataset_from_tfrecords(tfrecords_valid))

# add repeat operation
ds_train, ds_valid = ds_train.repeat(), ds_valid.repeat()
```

Add repeat() operation to learn multiple epochs.
```Python
# add repeat operation
ds_train, ds_valid = ds_train.repeat(), ds_valid.repeat()
```

Last step is define and fit and evaluate Model.
```Python
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
```

### Use AugmentImg Directory
Above examples ware create tf.data.Dataset by DatasetCreator. If you need to control your dataflow in other way, you could use AugmentImage Directory

#### 1. initialize
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

#### 2. use in tf.data.map() after batch()
```python 
ds=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(image),
                      tf.data.Dataset.from_tensor_slices(label))) \
                    .shuffle(BATCH_SIZE*10).batch(BATCH_SIZE).map(arg_fun)
model.fit(ds)
```

