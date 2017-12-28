#!/usr/bin/env python

import cv2
import numpy as np
import scipy.misc
import unittest

import preprocess.motion_analysis as ma
import util.basics as basics

class TestMotionAnalysis(unittest.TestCase):
    # Note: this is just a unit test; it does not evaluate
    # precision of these detectors on large varied sets of videos

    def test_frame_cuts(self):
        im1 = scipy.misc.imread(basics.testdata_dir('preprocess/before_cut.png'))
        im2 = scipy.misc.imread(basics.testdata_dir('preprocess/after_cut.png'))

        im1n = ma.normalize_for_analysis(im1)
        im2n = ma.normalize_for_analysis(im2)

        analysis = ma.MotionAnalysis(im1n, im2n)
        self.assertTrue(analysis.likely_frame_cut())


    def test_same_frame(self):
        im1 = scipy.misc.imread(basics.testdata_dir('preprocess/frame.png'))
        im2 = scipy.misc.imread(basics.testdata_dir('preprocess/frame_repeat.png'))
        im3 = scipy.misc.imread(basics.testdata_dir('preprocess/frame_subtle_change.png'))

        im1n = ma.normalize_for_analysis(im1)
        im2n = ma.normalize_for_analysis(im2)
        im3n = ma.normalize_for_analysis(im3)

        analysis1 = ma.MotionAnalysis(im1n, im2n)
        self.assertFalse(analysis1.likely_frame_cut())
        self.assertTrue(analysis1.likely_same_frame())

        analysis2 = ma.MotionAnalysis(im2n, im3n)
        self.assertFalse(analysis2.likely_frame_cut())
        self.assertFalse(analysis2.likely_same_frame())

        print ('Same frame metric vs different frame (%0.3f vs %0.3f)' %
               (analysis1.motion_metric(), analysis2.motion_metric()))
        self.assertTrue(analysis1.motion_metric() < analysis2.motion_metric())


class TestMotionLocalizer(unittest.TestCase):
    # Note: this is just a unit test; it does not evaluate
    # precision of these detectors on large varied sets of videos

    def test_localization(self):
        im1 = scipy.misc.imread(basics.testdata_dir('preprocess/frame_repeat.png'))
        im2 = scipy.misc.imread(basics.testdata_dir('preprocess/frame_subtle_change.png'))
        im3 = scipy.misc.imread(basics.testdata_dir('preprocess/frame_subtle_change2.png'))

        im1n = ma.normalize_for_analysis(im1)
        im2n = ma.normalize_for_analysis(im2)
        im3n = ma.normalize_for_analysis(im3)
        proc_shape = np.shape(im1n)

        analysis1 = ma.MotionAnalysis(im1n, im2n)
        self.assertFalse(analysis1.likely_frame_cut())
        self.assertFalse(analysis1.likely_same_frame())

        analysis2 = ma.MotionAnalysis(im2n, im3n)
        self.assertFalse(analysis2.likely_frame_cut())
        self.assertFalse(analysis2.likely_same_frame())

        diff = cv2.cvtColor(analysis2.diff, cv2.COLOR_GRAY2BGR)

        localizer = ma.MotionLocalizer([im1n, im2n, im3n], analysis1, analysis2)
        localizer.stochastic_localization()
        rects = localizer.motion_rectangles()
        for r in rects:
            print str(r)
            r.resize_for_image(proc_shape)
            r.draw(diff, (255, 0, 0))
        scipy.misc.imsave('/tmp/r2.png', diff)
        print 'Saved output to /tmp/r2.png'
        self.assertTrue(len(rect) >= 3)
