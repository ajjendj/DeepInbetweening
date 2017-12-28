#!/usr/bin/env python

import cv2
import numpy as np

class ImageRect(object):
    '''
    Convenience class to represent a rectangle inside an image of particular shape.
    Many operations on rectangles rely on relationship between image boundaries,
    and ImageRect allows performing these conveniently.
    '''

    def __init__(self, x, y, w, h, image_shape, accept_degenerate=False):
        '''
        Standard constructor from x,y of upper left corner, and width, height.
        ----------------------
        Parameters:
        image_shape - tuple (height,width) such as the result of np.shape(image)
                      for the image to which this rectangle is relative
        accept_degenerate - if true, will automatically clamp rectangle to image
                            boundaries, otherwise will complain
        '''
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if not accept_degenerate:
            if x < 0 or y < 0:
                raise RuntimeError('Creating image rectangle with negative origin')
            if x + w > self.image_width or y + h > self.image_height:
                raise RuntimeError('Creating image rectangle overflowing image border')
            if w < 0 or h < 0:
                raise RuntimeError('Creating image rectangle with degenerate dimension')
        self.clamp_to_valid()


    @staticmethod
    def create_from_center(center, width, height, image_shape, accept_degenerate=False):
        '''
        Convenience constructor to create from rectangle center point and width, height.
        '''
        x = center[0] - width / 2
        y = center[1] - height / 2
        return ImageRect(x, y, width, height, image_shape, accept_degenerate)


    def center(self):
        cx = self.x + self.w / 2
        cy = self.y + self.h / 2
        return cx, cy


    def subarray(self, image):
        '''
        Returns a subarray of the image corresponding to the region.
        Image must be same shape as the one rect was created with.
        '''
        self.__check_shape(np.shape(image))
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


    def set_subarray(self, image, val):
        self.__check_shape(np.shape(image))
        image[self.y : self.y + self.h, self.x : self.x + self.w] = val


    def draw(self, image, color, thickness=1):
        '''
        Draw rectangle in the image. Image must be same shape as the one rect was created with.
        ----------------------
        Parameters:
        color: a uint8 value for grayscale, and a (r, g, b) tuple for color images.
        thickness: negative thickness fills the rectangle.
        '''
        self.__check_shape(np.shape(image))
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      color, thickness=thickness)


    def make_square(self):
        '''
        Tries to pad itself to become square, returns True if successful.
        If expanding crosses image boundary, may shrink one dimension, returning
        False.
        '''
        cropped = False

        cx, cy = self.center()
        dim = max(self.h, self.w)
        x2 = cx - dim / 2
        y2 = cy - dim / 2
        self.x = max(x2, 0)
        self.y = max(y2, 0)
        cropped = cropped or self.x != x2 or self.y != y2
        self.w = min(dim, self.image_width - self.x)
        self.h = min(dim, self.image_height - self.y)
        cropped = cropped or self.w != dim or self.h != dim
        if self.w < dim:
            # Try to move x left
            diff = dim - self.w
            right = self.w + self.x
            self.x = max(self.x - diff, 0)
            self.w = right - self.x
        if self.h < dim:
            # Try to move y up
            diff = dim - self.h
            bottom = self.h + self.y
            self.y = max(self.y - diff, 0)
            self.h = bottom - self.y
        self.w = min(self.w, self.h)
        self.h = self.w

        return not cropped


    def resize_for_image(self, image_shape):
        '''
        Resizes the rectangle so that it corresponds to the same rectangle
        in a different image shape.
        '''
        width_factor = image_shape[1] / self.image_width
        height_factor = image_shape[0] / self.image_height
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]
        self.x *= width_factor
        self.w *= width_factor
        self.y *= height_factor
        self.h *= height_factor


    def clamp_to_valid(self):
        if self.x >= self.image_width:
            self.w = 0
        if self.y >= self.image_height:
            self.h = 0
        self.x = max(0, min(self.x, self.image_width - 1))
        self.y = max(0, min(self.y, self.image_height - 1))
        self.w = max(0, self.w)
        self.h = max(0, self.h)
        self.w = min(self.image_width - self.x, self.w)
        self.h = min(self.image_height - self.y, self.h)


    def __check_shape(self, shape):
        img_shape = (shape[0], shape[1])  # disregard channel
        my_shape = (self.image_height, self.image_width)
        if img_shape != my_shape:
            raise RuntimeError(
                'Extracting rectangle relative to image size %s from images of size %s' %
                (str(my_shape), str(img_shape)))


    def __str__(self):
        return ('(x: %s, y: %s), (w: %s, h: %s)' %
                (str(self.x), str(self.y), str(self.w), str(self.h)))
