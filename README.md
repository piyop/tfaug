# tfaug
Tensorflow >= 2.0 image augmentation class for tf.data

multiprocessing was officially deprecated Tensorflow >= 2.0. 
If we use multiprocessing, which cause unexpected interruption while learning.

I used keras ImageDataGenerator class before tf2, but it can no longer use multiprocessing.
Therefore I made this class. This provides us easy way to augment image while learning using tf.data. tf.data is
automatically map functions on multiprocesses except when using tf.py_funtion.

This class augment input image and label image with same transformations at the sametime.

any comment and pull request welcomed.

## dependancies
tensorflow >= 2.0

tensorflow-addons

 * for test srcipt

pillow

numpy

matplotlib


## install
python -m pip install git+https://github.com/piyop/tf2_augimg

## How to use

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

### 2. use
```python 
ds=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(image),
                      tf.data.Dataset.from_tensor_slices(label))) \
                    .shuffle(BATCH_SIZE*10).batch(BATCH_SIZE).map(arg_fun)
model.fit(ds)
```

detail is writtend in code and test code.
