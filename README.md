# tfaug
Tensorflow >= 2.0 image augmentation class for tf.data

multiprocessing was officially deprecated Tensorflow >= 2.0. 
If we use multiprocessing, which cause unexpected interruption while learning.

Keras ImageDataGenerator class was widelly used before tf2, but it can no longer use multiprocessing and thus I made up this package. 
This provides us easy way to augment image while learning using tf.data. tf.data is automatically map functions on multiprocesses except when using tf.py_funtion.

any comment and pull request are welcomed.

## features
 * Augment input image and label image with same transformations at the same time.
 * Augment on batch which is more efficient than augment each image.

## dependancies
 * tensorflow >= 2.0
 * tensorflow-addons
### for test srcipt
 * pillow
 * numpy
 * matplotlib

## install
python -m pip install git+https://github.com/piyop/tfaug

## Usage
### 1. initialize
```python  
from tfaug import augment_img  
#set your augment parameters below:
arg_fun = argment_img(rotation=0, 
                      standardize=False,
                      random_flip_left_right=True,
                      random_flip_up_down=True, 
                      random_shift=(.1,.1), 
                      random_zoom=.1,
                      random_brightness=.2,
                      random_saturation=None,
                      training=True)  
```

### 2. use in tf.data.map() after batch()
```python 
ds=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(image),
                      tf.data.Dataset.from_tensor_slices(label))) \
                    .shuffle(BATCH_SIZE*10).batch(BATCH_SIZE).map(arg_fun)
model.fit(ds)
```

detailed useage is writtend in test code.


## future work
 * add parameter output image size
 * add other functions in tf.image
