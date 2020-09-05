# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:18:40 2020

@author: t.okuda
"""

import tensorflow as tf
import tensorflow_addons as tfa
from typing import Tuple, Callable


class augment_img():
    """
    
    image augmentation class for tf.data
        
    """
    
    def __init__(self,
                rotation : float = 0 , 
                standardize : bool = False,
                random_flip_left_right : bool = False,
                random_flip_up_down : bool = False, 
                random_shift :Tuple[float, float] = None ,
                random_zoom : float  = None,
                random_brightness : float = None,
                random_saturation :Tuple[float, float] = None,
                random_hue : float = None,
                random_crop : int = None,
                training : bool = False) \
                -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor,tf.Tensor]]:
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
        rotation : float, optional
            rotation angle(degree). The default is 0.
        standardize : bool, optional
            image standardization. The default is True.
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
        random_zoom : float, optional
            random zoom range -random_zoom to random_zoom.
            value of random_zoom is ratio of image size
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
        random_crop : int, optional
            randomely crop image with size [x,y] = [random_crop, random_crop]. 
            The default is None.
        training : bool, optional
            If false, this class don't augment image except standardize. 
            The default is False.

        Returns
        -------
        class instance : Callable[[tf.Tensor, tf.Tensor, bool], Tuple[tf.Tensor,tf.Tensor]]

        """        
        
        self._training = training
        self._rotation = rotation
        self._standardize = standardize 
        self._random_flip_left_right = random_flip_left_right
        self._random_flip_up_down = random_flip_up_down  
        self._random_shift = random_shift 
        self._random_zoom = random_zoom
        self._random_brightness = random_brightness 
        self._random_saturation = random_saturation 
        self._random_hue = random_hue
        self._random_crop = random_crop 
        
        
    @tf.function
    def __call__(self, image : tf.Tensor, label : tf.Tensor=None) -> Tuple[tf.Tensor,tf.Tensor]:
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
    
    
    def _augmentation(self, image, label = None, train = False):
        """
        
        Params
        --------------------
        image : 4d tensor
            input image
        label : 4d tensor, optional,  
            image fort ture label
        train : bool, optional, 
            training or not
        """
        in_size = tf.shape(image)[1:3]
        out_size = in_size
        
        #keep image channel dims
        last_image_dim = tf.shape(image)[-1]
        if label is not None:
            image = tf.concat([image, label], axis=3)
            
        if train:                    
            if self._random_flip_left_right:
                image = tf.image.random_flip_left_right(image)
            if self._random_flip_up_down:
                image = tf.image.random_flip_up_down(image)
                
            if self._random_zoom or self._random_shift or self._random_crop:
                zoom = tf.repeat(tf.constant(0, dtype=tf.float32),tf.shape(image)[0])
                if self._random_zoom:
                    zoom = tf.random.uniform([tf.shape(image)[0]], -self._random_zoom, self._random_zoom)
                    
                if self._random_crop:
                    zoom += tf.cast(self._random_crop / in_size[0] - 1, tf.float32)
                    out_size = tf.constant((self._random_crop, self._random_crop))
                
                shift_x, shift_y = tf.repeat(tf.constant(0, dtype=tf.float32),tf.shape(image)[0]),tf.repeat(tf.constant(0, dtype=tf.float32),tf.shape(image)[0])
                if self._random_shift:
                    shift_x = tf.random.uniform([tf.shape(image)[0]], -self._random_shift[1], self._random_shift[1])
                    shift_y = tf.random.uniform([tf.shape(image)[0]], -self._random_shift[0], self._random_shift[0])
                                        
                boxes = tf.map_fn(lambda x: tf.stack([-x[0]+x[2], -x[0]+x[1], 1+x[0]+x[2], 1+x[0]+x[1]]),
                                  tf.transpose([zoom, shift_x, shift_y]))
                num_boxes = tf.range((tf.shape(image)[0]))

                image = tf.image.crop_and_resize(image, 
                                                  boxes, 
                                                  num_boxes, 
                                                  out_size,
                                                  method='bilinear', 
                                                  extrapolation_value=0)
                
            if self._rotation > 0:
                angle_rad = self._rotation * 3.141592653589793 / 180.0
                angles = tf.random.uniform([tf.shape(image)[0]], -angle_rad, angle_rad)
                image = tfa.image.rotate(image, angles, interpolation='BILINEAR')
                
                
        elif self._random_crop:#not train and random_crop
            #assume width and height is same
            offset_height = (in_size[0] - self._random_crop) // 2
            target_height = self._random_crop
            image = tf.image.crop_to_bounding_box(image, 
                                                  offset_height, 
                                                  offset_height, 
                                                  target_height, 
                                                  target_height)
            
        #separete image and label
        if label is not None:
            image, label = (image[:, :, :, :last_image_dim],
                image[:, :, :, last_image_dim:])
        
        if train:
            if self._random_brightness:
                image = tf.image.random_brightness(image, self._random_brightness)
            if self._random_saturation:
                image = tf.image.random_saturation(image, *self._random_saturation)
            if self._random_hue:
                image = tf.image.random_hue(image, self._random_hue)
                
        if self._standardize:
            image = tf.image.per_image_standardization(image)
            
        if label is not None: return image, label
        else: return image
        

