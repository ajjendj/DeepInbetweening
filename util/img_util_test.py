#!/usr/bin/env python

import numpy as np
import scipy.misc
import unittest

import util.img_util as img_util
import util.basics as basics

class TestImageRect(unittest.TestCase):

    def test_resize_methods(self):
        image_path = basics.testdata_dir('util/color_image0.png')
        img = scipy.misc.imread(image_path)

        # Test resizing
        rect = img_util.ImageRect(10, 20, 100, 50, np.shape(img))
        self.assertEqual(200, rect.image_width)
        self.assertEqual(160, rect.image_height)

        rect.resize_for_image((320, 400, 3))  # double the size
        self.assertEqual(20, rect.x)
        self.assertEqual(40, rect.y)
        self.assertEqual(200, rect.w)
        self.assertEqual(100, rect.h)

        # Test make square --------
        # Middle
        rect = img_util.ImageRect(10, 30, 80, 60, np.shape(img))
        self.assertTrue(rect.make_square())
        self.assertEqual(10, rect.x)
        self.assertEqual(20, rect.y)
        self.assertEqual(80, rect.w)
        self.assertEqual(80, rect.h)

        # Left side
        rect = img_util.ImageRect(2, 20, 10, 40, np.shape(img))
        self.assertFalse(rect.make_square())
        self.assertEqual(0, rect.x)
        self.assertEqual(20, rect.y)
        self.assertEqual(40, rect.w)
        self.assertEqual(40, rect.h)

        # Bottom side
        rect = img_util.ImageRect(100, 140, 40, 10, np.shape(img))
        self.assertFalse(rect.make_square())
        self.assertEqual(100, rect.x)
        self.assertEqual(120, rect.y)
        self.assertEqual(40, rect.w)
        self.assertEqual(40, rect.h)

        # Right side
        rect = img_util.ImageRect(180, 100, 20, 50, np.shape(img))
        self.assertFalse(rect.make_square())
        self.assertEqual(150, rect.x)
        self.assertEqual(100, rect.y)
        self.assertEqual(50, rect.w)
        self.assertEqual(50, rect.h)


    def test_indexing(self):
        image_path = basics.testdata_dir('util/color_image0.png')
        img = scipy.misc.imread(image_path)
        print np.shape(img)

        # Test creating degenerate rectangle
        with self.assertRaises(RuntimeError):
            rect = img_util.ImageRect(150, 160, 60, 50, np.shape(img))

        rect = img_util.ImageRect(150, 160, 60, 50, np.shape(img),
                                  accept_degenerate=True)
        self.assertEqual(150, rect.x)
        self.assertEqual(159, rect.y)
        self.assertEqual(50, rect.w)
        self.assertEqual(0, rect.h)

        # Test mutable sub-array and drawing are consistent with each other
        rect = img_util.ImageRect(50, 70, 20, 10, np.shape(img))
        rect.draw(img, (255, 255, 255), thickness=-1)
        s = rect.subarray(img)
        self.assertEqual(255, s.max())
        self.assertEqual(255, s.min())

        # Test we can actually assign
        s = rect.subarray(img)
        s = 0
