#!/usr/bin/env python

import cv2
import numpy as np
import scipy.misc
import unittest

import util.frame_grabbers as fg
import util.basics as basics

class TestFrameGrabbers(unittest.TestCase):

    def test_basics(self):
        grabber = fg.FileFrameGrabber(
            [ basics.testdata_dir('util/color_image0.png'),
              basics.testdata_dir('util/color_image1.png') ])
        self.assertEqual(2, grabber.total_frames())
        grabber.skip_frame()
        f = grabber.next_frame()
        second_frame = f
        self.assertFalse(f is None)
        f = grabber.next_frame()
        self.assertTrue(f is None)

        grabber = fg.VideoFrameGrabber(basics.testdata_dir('util/tiny_video.mov'))
        self.assertEqual(3, grabber.total_frames())
        grabber.skip_frame()
        f = grabber.next_frame()
        second_video_frame = f
        self.assertFalse(f is None)
        f = grabber.next_frame()
        self.assertFalse(f is None)
        f = grabber.next_frame()
        self.assertTrue(f is None)

        d = cv2.absdiff(second_frame, second_video_frame)
        diff = cv2.absdiff(second_frame, second_video_frame).sum()
        shape = np.shape(second_frame)
        diff /= (1.0 * shape[0] * shape[1] * shape[2])
        # Per pixel difference < 4 (due to encodings)
        self.assertTrue(diff < 4)
