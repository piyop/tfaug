# tfaug
Tensorflow >= 2.0 image augmentation class for tf.data

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
        """
        __init__() sets the parameters for augmantation.
        
        When __call__() can take not only input image but also label image.
        Image and label image will be augmanted with same transformation at the same time.
        However, label image is not augmanted by random_brightness, random_saturation, standardize
        
        This augmantation is executed on batch images. Input image should be 4d Tensor(batch, x, y, channel)
        Image x and y size is same as premise.
        
        If training == False, it will not augment image except standardize. 
        
        Parameters
        ----------
        standardize : bool, optional
            image standardization. The default is True.
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
            random_shear[0] : y-direction, random_shear[1] : x-direction
            value means degree of each axis
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
        interpolation : str, The default is nearest.
            interpolation method. nearest or bilinear
        training : bool, optional
            If false, this class don't augment image except standardize. 
            The default is False.

        Returns
        -------
        class instance : Callable[[tf.Tensor, tf.Tensor, bool], Tuple[tf.Tensor,tf.Tensor]]

        """        
 
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
 * add parameter output image size => random_crop can change output imase size
 * add other functions in tf.image
 * add function writing separated TFRecord files from image files
