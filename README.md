# tfaug
Tensorflow image augmentation class for tf.data

multiprocessing was officially deprecated Tensorflow >= 2.0. 
If we use multiprocessing, which cause unexpected interruption while learning.

Keras ImageDataGenerator class was widelly used before tf2, but it can no longer use multiprocessing and thus I made up this package.   
This package provides us easy way to augment image while learning using tf.data.   
tf.data is automatically map functions on multiprocesses except when using tf.py_funtion.  

any comment and pull request are welcomed.

## features
 * augment input image and label image with same transformations at the same time.
 * augment on batch which is more efficient than augment each image.
 * use only tensorflow operators and functions. Because any other operations or functions (e.g. numpy functions) cause limitation on multiprocess augmentation while using @tf.function to get a peformance [as mentined here](https://www.tensorflow.org/guide/function).

## dependancies
 * tensorflow >= 2.0
 * tensorflow-addons
### for test srcipt
 * pillow
 * numpy
 * matplotlib

## supported transformations:
 * standardize
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
 
## install
python -m pip install git+https://github.com/piyop/tfaug

## Usage
### 1. initialize
```python  
from tfaug import augment_img 
#set your augment parameters below:
arg_fun = augment_img(standardize=False,
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
                      training=True) 
 
```

### 2. use in tf.data.map() after batch()
```python 
ds=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(image),
                      tf.data.Dataset.from_tensor_slices(label))) \
                    .shuffle(BATCH_SIZE*10).batch(BATCH_SIZE).map(arg_fun)
model.fit(ds)
```

detailed usage is written in test code.


## future work
 * add other functions in tf.image
 * add function writing separated TFRecord files from image files
 * add save and load parameters method
