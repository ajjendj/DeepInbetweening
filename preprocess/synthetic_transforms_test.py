#!/usr/bin/env python

import cv2
import numpy as np
import scipy.misc
import unittest

import preprocess.synthetic_transforms as st
import util.basics as basics
import util.img_util as img_util


class TestTransformer(unittest.TestCase):

    def test_sane_transform(self):
        # create some transforms
        img = scipy.misc.imread(basics.testdata_dir('preprocess/frame.png'))
        shape = np.shape(img)
        rect = img_util.ImageRect(60, 250, 50, 50, shape)
        transformer = st.Transformer(rect)

        trans, frac_trans = transformer.random_affine_transform(homogeneous=False)
        print str(trans)

        img_trans = cv2.warpAffine(img, trans, (shape[1], shape[0]), borderValue=(0,0,0))
        img_frac_trans = cv2.warpAffine(img, frac_trans, (shape[1], shape[0]), borderValue=(0,0,0))


        rect.draw(img, (255, 0, 0))
        rect.draw(img_trans, (255, 0, 0))
        rect.draw(img_frac_trans, (255, 0, 0))

        # Apply and crop out the region from the transformed image.
        # Does this look sane?
        scipy.misc.imsave('/tmp/a.png', img)
        scipy.misc.imsave('/tmp/b.png', img_frac_trans)
        scipy.misc.imsave('/tmp/c.png', img_trans)

        #Apply inverse transform.
        trans_inv = np.identity(3)
        frac_trans_inv = np.identity(3)
        trans_inv[:2,:] = trans
        trans_inv = np.linalg.inv(trans_inv)
        frac_trans_inv[:2,:] = frac_trans
        frac_trans_inv = np.linalg.inv(frac_trans_inv)

        img_trans_inv = cv2.warpAffine(img, trans_inv[:2,:], (shape[1], shape[0]), borderValue=(0,0,0))
        img_frac_trans_inv = cv2.warpAffine(img, frac_trans_inv[:2,:], (shape[1], shape[0]), borderValue=(0,0,0))

        scipy.misc.imsave('/tmp/b_inv.png', img_frac_trans_inv)
        scipy.misc.imsave('/tmp/c_inv.png', img_trans_inv)




        # Also write the crops
