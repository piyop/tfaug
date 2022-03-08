
import unittest

import numpy as np
import tensorflow as tf

from tfaug import AugmentImg
import test_tfaug_tool as tool


# constants
DATADIR = 'testdata/'
testimg = DATADIR+'Lenna.png'
testlbl = DATADIR+'Lenna.png'
BATCH_SIZE = 10
params = AugmentImg.params

class TestAugTypes(unittest.TestCase):

    def test_augmentation(self):
        """
        test class AugmentImg

        Returns
        -------
        None.

        """

        cases = [
            ('test resize',
             params(resize=(300, 500),
                    training=True)),
            ('test rotation',
             params(random_rotation=45,
                    training=True)),
            ('test random_flip_left_right',
             params(random_flip_left_right=True,
                    training=True)),
            ('test random_flip_up_down',
             params(random_flip_up_down=True,
                    training=True)),
            ('test y_shift',
             params(random_shift=[255, 0],
                    training=True)),
            ('test x_shift',
             params(random_shift=[0, 255],
                    training=True)),
            ('test zoom',
             params(random_zoom=(0.5, 0.1),
                    training=True)),
            ('test shear',
             params(random_shear=(20, 20),
                    training=True)),
            ('test random_brightness',
             params(random_brightness=0.5,
                    training=True)),
            ('test random_saturation',
             params(random_saturation=(0.5, 1.5),
                    training=True)),
            ('test random_hue1',
             params(random_hue=0.01,
                    training=True)),
            ('test random_hue2',
             params(random_hue=0.1,
                    training=True)),
            ('test random_contrast',
             params(random_contrast=[1.4, 2],
                    training=True)),
            ('test central_crop',
             params(central_crop=[255, 128],
                    training=True)),
            ('test random_noise',
             params(random_noise=50,
                    training=True)),
            ('test random_blur',
             params(resize=(50, 50),
                    random_blur=1,
                    random_blur_kernel=5,
                    training=True)),
            ('test dtype',
             params(standardize=True,
                    resize=(300, 400),
                    random_brightness=0.5,
                    random_hue=0.01,
                    random_contrast=[.1, .5],
                    random_noise=100,
                    interpolation='nearest',
                    dtype=tf.float16,
                    training=True)),
            ('test num_transforms',
             params(standardize=True,
                    resize=(300, 400),
                    random_brightness=0.5,
                    random_rotation=20,
                    interpolation='nearest',
                    input_shape=[BATCH_SIZE, 512, 512, 3],
                    num_transforms=50,
                    training=True)),
            ('test x_shift and rotation',
             params(random_rotation=45,
                    random_shift=(0, 256),
                    interpolation='nearest',
                    training=True)),
            ('test crop and zoom',
             params(random_rotation=45,
                    random_flip_left_right=False,
                    random_flip_up_down=False,
                    random_shift=None,
                    random_zoom=(0.8, 0.1),
                    random_crop=(256, 512),
                    interpolation='bilinear',
                    training=True)),
            ('test shear and color',
             params(standardize=True,
                    random_flip_left_right=False,
                    random_flip_up_down=False,
                    random_zoom=[0.1, 0.1],
                    random_shear=(10, 10),
                    random_brightness=0.5,
                    random_saturation=[0.5, 1.5],
                    random_hue=0.001,
                    random_contrast=[.1, .5],
                    interpolation='bilinear',
                    training=True)),
            ('test train = False',
             params(standardize=True,
                    random_rotation=45,
                    random_flip_left_right=True,
                    random_flip_up_down=True,
                    random_brightness=0.5,
                    random_contrast=[.5, 1.5],
                    random_crop=(256, 256),
                    interpolation='nearest',
                    training=False)),
            ('test resize and zoom',
             params(resize=(256, 512),
                    random_zoom=(0, 0.5),
                    interpolation='nearest',
                    training=True)),
            ('test resize and rotation',
             params(resize=(900, 400),
                    random_rotation=45,
                    interpolation='nearest',
                    training=True)),
        ]

        for no, case in enumerate(cases):
            with self.subTest(case=case):
                print(case)
                tool.test_aug_prm(case[1], case[0], testimg, testlbl, DATADIR)


if __name__ == '__main__':
    pass
    unittest.main()    
    # testimg = 'lenna_l.png'    
    # prm = params(resize=(50, 50),
    #        random_blur=1,
    #        random_blur_kernel=5,
    #        training=True)
    
    # tool.test_aug_prm(prm, 'test', testimg, testlbl, DATADIR)
