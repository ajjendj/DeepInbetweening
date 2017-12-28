#!/usr/bin/env python

import math
import random

import numpy as np


class TransformerOptions(object):
    def __init__(self, rect):
        self.x_translate_range = (-rect.w / 3, rect.w / 3)
        self.y_translate_range = (-rect.h / 3, rect.h / 3)
        self.shear_range = (-rect.w / 50.0, rect.w / 50.0)
        self.angle_range = (-math.pi / 6, math.pi / 6)
        self.scale_range = (-0.4, 0.5)  # addition to 1


class Transformer(object):
    '''
    Helper transform for creating affine transformation matrices for
    augmenting the data for a particular region of the frame.
    The entire frame can be transformed with this transformation,
    without changing the rectangle of interest.
    '''
    def __init__(self,
                 rect,
                 options=None):
        ''' Initialize for a particular rectangle. '''
        if options is None:
            options = TransformerOptions(rect)
        self.rect = rect
        self.opts = options


    def random_affine_transform(self, fraction=0.5, homogeneous=True):
        '''
        Based on parameters passed to the constructor creates a random affine
        transform with scale,rotation,shear,translation all applied applied
        with some probability in the system centered on the rectangle.

        Returns: two 3x3 matrices in homogeneous coordinates
                 1. Full affine transform matrix
                 2. Transform matrix with *fraction* of all the transformations
                    (e.g. rotation by half of the angle in 1)
        '''
        transforms = ['scale', 'shear', 'rotate', 'translate']
        random.shuffle(transforms)
        transforms = transforms[0:random.randint(1,4)]

        cx, cy = self.rect.center()
        Tcenter = self.translation_matrix(np.array([cx, cy]))
        Tback = self.translation_matrix(np.array([-cx, -cy]))

        # Apply scale, rotation, shear with rectangle center
        # as the center of the coordinate system.
        # TODO: maybe should be randomized as well?
        full_transf = Tback
        frac_transf = Tback
        if 'scale' in transforms:
            scale_params_addition = self.random_scale()
            scale_params = 1 + scale_params_addition
            scale_params_frac = 1 + fraction * scale_params_addition
            scale = self.scale_matrix(scale_params)
            frac_scale = self.scale_matrix(scale_params_frac)
            full_transf = np.matmul(scale, full_transf)
            frac_transf = np.matmul(frac_scale, frac_transf)

        if 'shear' in transforms:
            shear_params = self.random_shear()
            print str(shear_params)
            shear = self.shear_matrix(shear_params)
            frac_shear = self.shear_matrix(shear_params * fraction)
            full_transf = np.matmul(shear, full_transf)
            frac_transf = np.matmul(frac_shear, frac_transf)

        if 'rotate' in transforms:
            angle = self.random_rotation_angle()
            rotate = self.rotation_matrix(angle)
            frac_rotate = self.rotation_matrix(angle * fraction)
            full_transf = np.matmul(rotate, full_transf)
            frac_transf = np.matmul(frac_rotate, frac_transf)

        if 'translate' in transforms:
            translation_params = self.random_translation()
            translation = self.translation_matrix(translation_params)
            frac_transflation = self.translation_matrix(translation_params * fraction)
            full_transf = np.matmul(translation, full_transf)
            frac_transf = np.matmul(frac_transflation, frac_transf)

        # Transform back to the original coordinate system
        full_transf = np.matmul(Tcenter, full_transf)
        frac_transf = np.matmul(Tcenter, frac_transf)
        print str(full_transf)

        if homogeneous:
            return full_transf, frac_transf
        else:
            return full_transf[0:2, :], frac_transf[0:2, :]


    # TRANSLATION ------------------------------------------
    def random_translation(self):
        return np.array([self._rand(self.opts.x_translate_range),
                         self._rand(self.opts.y_translate_range)])

    def translation_matrix(self, params, homogeneous=True):
        row0 = [1.0, 0, params[0]]
        row1 = [0, 1.0, params[1]]
        if homogeneous:
            return np.array([row0, row1, [0, 0, 1.0]])
        else:
            return np.array([row0, row1])

    # ROTATION ---------------------------------------------
    def random_rotation_angle(self):
        return self._rand(self.opts.angle_range)

    def rotation_matrix(self, radians):
        c = math.cos(radians)
        s = math.sin(radians)
        return np.array([[c, s, 0],
                         [-s, c, 0],
                         [0, 0, 1.0]])

    # SHEAR ------------------------------------------------
    def random_shear(self):
        return np.array([self._rand(self.opts.shear_range),
                         random.randint(0, 1) * 2.0 - 1])

    def shear_matrix(self, params):
        return np.array([[1.0, (params[0] if params[1] > 0 else 0), 0],
                         [(params[0] if params[1] < 0 else 0), 1.0, 0],
                         [0, 0, 1.0]])

    # SCALE ------------------------------------------------
    def random_scale(self):
        return np.array([self._rand(self.opts.scale_range),
                         self._rand(self.opts.scale_range)])

    def scale_matrix(self, params):
        return np.array([[params[0], 0, 0],
                         [0, params[1], 0],
                         [0, 0, 1.0]])

    # UTILS ------------------------------------------------

    def _rand(self, rng):
        return random.uniform(rng[0], rng[1])
