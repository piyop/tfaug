# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:18:40 2020

@author: t.okuda
"""

import math
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Tuple, Callable

class augment_img():
    """
    
    image augmentation class for tf.data
        
    """
    
    def __init__(self,
                standardize : bool = False,
                random_rotation : float = 0 , 
                random_flip_left_right : bool = False,
                random_flip_up_down : bool = False, 
                random_shift : Tuple[float, float] = None ,
                random_zoom : Tuple[float, float]  = None,
                random_shear: Tuple[float, float] = None,
                random_brightness : float = None,
                random_saturation :Tuple[float, float] = None,
                random_hue : float = None,
                random_contrast : Tuple[float, float] = None,
                random_crop : Tuple[int, int] = None,
                interpolation : str = 'nearest',
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
            randomely adjust y and x directional shear degree
            random_shear[0] : y-direction, random_shear[1] : x-direction
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
        
        self._training = training
        self._standardize = standardize
        
        (self._random_rotation, self._random_shift, self._random_zoom, 
         self._random_shear, self._random_crop) = None, None, None, None, None
        
        if random_rotation:
            self._random_rotation = tf.cast(random_rotation, tf.float32)
        self._random_flip_left_right = random_flip_left_right
        self._random_flip_up_down = random_flip_up_down  
        if random_shift:
            self._random_shift = tf.cast(random_shift, tf.float32)
        if random_zoom:
            self._random_zoom = tf.cast(random_zoom, tf.float32)
        if random_shear:
            self._random_shear = tf.cast(random_shear, tf.float32)
        self._random_brightness = random_brightness 
        self._random_saturation = random_saturation 
        self._random_hue = random_hue
        self._random_contrast = random_contrast
        if random_crop:
            self._random_crop = tf.cast(random_crop, tf.int32)
        self._interpolation = interpolation
        
        
    # @tf.function
    def __call__(self, image : tf.Tensor, 
                 label : tf.Tensor=None) -> Tuple[tf.Tensor,tf.Tensor]:
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
        return self._augmentation(image, label, self._training )
    
    
    def _augmentation(self, image, label = None, train = False):
        """
        
        Parameters
        --------------------
        image : 4d tensor
            input image
        label : 4d tensor, optional,  
            image fort ture label
        train : bool, optional, 
            training or not
        """
        in_size = tf.shape(image)[1:3]
        batch_size = tf.shape(image)[0]
        
        #keep image channel dims
        last_image_dim = tf.shape(image)[-1]
        if label is not None:
            image = tf.concat([image, label], axis=3)
            
        if train:                    
            if self._random_flip_left_right:
                image = tf.image.random_flip_left_right(image)
            if self._random_flip_up_down:
                image = tf.image.random_flip_up_down(image)
                
            if (isinstance(self._random_rotation, tf.Tensor) or 
                isinstance(self._random_zoom, tf.Tensor) or 
                isinstance(self._random_shift, tf.Tensor) or 
                isinstance(self._random_shear, tf.Tensor)):
                    
                size_fl = tf.cast(in_size, dtype=tf.float32)                    
        
                trans_matrix = tf.eye(3, 3, [tf.shape(image)[0]], dtype=tf.float32)
                
                shift_y = tf.zeros([batch_size], dtype=tf.float32)
                shift_x = tf.zeros([batch_size], dtype=tf.float32)        
                shear_y = tf.zeros([batch_size], dtype=tf.float32)
                shear_x = tf.zeros([batch_size], dtype=tf.float32)
                zoom_y = tf.zeros([batch_size], dtype=tf.float32)
                zoom_x = tf.zeros([batch_size], dtype=tf.float32)
                
                if isinstance(self._random_shift, tf.Tensor):            
                    shift_y += tf.random.uniform([batch_size], -self._random_shift[0], self._random_shift[0])
                    shift_x += tf.random.uniform([batch_size], -self._random_shift[1], self._random_shift[1])
                    
                if isinstance(self._random_shear, tf.Tensor):
                    shear_tan = tf.tan(self._random_shear / 180 * math.pi)
                    shear_y += tf.random.uniform([batch_size], -shear_tan[0], shear_tan[0])
                    shear_x += tf.random.uniform([batch_size], -shear_tan[1], shear_tan[1])
                         
                    shift_y += -(size_fl[0] * shear_y) / 2
                    shift_x += -(size_fl[1] * shear_x) / 2                            
                
                if isinstance(self._random_zoom, tf.Tensor):
                    zoom_y = tf.random.uniform([batch_size], -self._random_zoom[0], self._random_zoom[0])
                    zoom_x = tf.random.uniform([batch_size], -self._random_zoom[1], self._random_zoom[1])
                    
                    shift_y += -(size_fl[0] * zoom_y) / 2
                    shift_x += -(size_fl[1] * zoom_x) / 2                               
                
                trans_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                                            [[x[5]+1, x[3], x[1]],
                                            [x[2], x[4]+1, x[0]],
                                            [0,0,1]], tf.float32),        
                                  tf.transpose([shift_y, shift_x, shear_y, shear_x, zoom_y, zoom_x]))
                    
                if isinstance(self._random_rotation, tf.Tensor):
                    rad_theta = self._random_rotation / 180 * math.pi
                    rot = tf.random.uniform([batch_size], -rad_theta, rad_theta)
                    h11 = tf.cos(rot)
                    h12 = -tf.sin(rot)
                    h21 = tf.sin(rot)
                    h22 = tf.cos(rot)
                    shift_rot_y = ( (size_fl[0] - size_fl[0] * tf.cos(rot)) - (size_fl[1] * tf.sin(rot)) ) / 2
                    shift_rot_x =( (size_fl[1] - size_fl[1] * tf.cos(rot)) + (size_fl[0] * tf.sin(rot)) ) / 2
                    
                    rot_matrix = tf.map_fn(lambda x: tf.convert_to_tensor(
                                            [[x[0], x[1], x[5]],
                                            [x[2], x[3], x[4]],
                                            [0,0,1]], tf.float32),        
                                  tf.transpose([h11, h12, h21, h22, shift_rot_y, shift_rot_x]))
                    
                    trans_matrix = tf.keras.backend.batch_dot(trans_matrix, rot_matrix)
                                        
                #get matrix
                M = tfa.image.transform_ops.matrices_to_flat_transforms(tf.linalg.inv(trans_matrix))
                
                #transform
                image = tfa.image.transform(image,M,interpolation=self._interpolation)
                
        if isinstance(self._random_crop, tf.Tensor):
            if train:
                depth = tf.shape(image)[3]
                image = tf.image.random_crop(image, 
                                              size=tf.concat((tf.expand_dims(batch_size,-1),
                                                              self._random_crop,
                                                              tf.expand_dims(depth,-1)),axis=0))       
            else:
                #central crop
                offset_height = (in_size[0] - self._random_crop[0]) // 2
                target_height = self._random_crop[0]
                offset_width = (in_size[1] - self._random_crop[1]) // 2
                target_width = self._random_crop[1]
                image = tf.image.crop_to_bounding_box(image, 
                                                      offset_height,
                                                      offset_width,
                                                      target_height,
                                                      target_width)
            
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
            if self._random_contrast:
                image = tf.image.random_contrast(image, *self._random_contrast)
                
        if self._standardize:
            image = tf.image.per_image_standardization(image)
            
        if label is not None: return image, label
        else: return image
        