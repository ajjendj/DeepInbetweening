#!usr/bin/env python

''' Utility functions for network visualization. '''

import tensorflow as tf
import numpy as np
import cv2

def kernels_to_image(kernels, N, M, padding):
    '''
    Convert NxM kernels (e.g. in a CNN) into an N x M image
    visualizing all the kernels.

    Parameters:
    ----------
    kernels: tensor with shape
        [ kernel_width, kernel_height, channels, kernel_num = (NxM) ]
    N : number of kernels to lay out in a row
    M : number of rows
    padding: number of pixels between rows

    Returns:
    -------
    Tensor of shape [ im_width, im_height, 3, 1].
    '''
    assert kernels.get_shape()[3] == N * M

    #only handles num of channels to be 1 (greyscale), 3(RGB) or 4(RGBA)
    #if kernels.shape[2] == 2: i.e. input images have two channels, copy and stack second channel to make RGB image
    if kernels.get_shape()[2] == 2:
        kernel_chan2 = kernels[:,:,1,:]
        kernel_chan2 = tf.reshape(kernel_chan2, [int(kernels.get_shape()[0]), int(kernels.get_shape()[1]), 1, int(kernels.get_shape()[3])])
        kernels = tf.concat(2,[kernels, kernel_chan2])

    #pad kernels
    padded_kernels = tf.pad(kernels, tf.constant([[padding,padding],[padding,padding],[0,0],[0,0]]))

    #split the kernels into kernel_num separate kernels
    split_kernels = tf.split(3, padded_kernels.get_shape()[3], padded_kernels)

    #lay the kernels out in a grid
    row_kernels_list = [tf.concat(0, split_kernels[row*N:(row+1)*N]) for row in range(N)]
    grid_kernels = tf.concat(1, row_kernels_list)

    if kernels.get_shape()[2] == 1:
        grid_kernels = tf.reshape(grid_kernels, [1, int(padded_kernels.get_shape()[0])*N, int(padded_kernels.get_shape()[1])*M, 1])
    else:
        grid_kernels = tf.reshape(grid_kernels, [1, int(padded_kernels.get_shape()[0])*N, int(padded_kernels.get_shape()[1])*M, 3])

    scaled_grid_kernels = (grid_kernels - tf.reduce_min(grid_kernels))/(tf.reduce_max(grid_kernels) - tf.reduce_min(grid_kernels))

    return scaled_grid_kernels


def generate_output_imagegrid(a,b,output,output_t,mask,mask_t):
    '''
    This function can used to visualize the output of a network. The batch of images
    will be arranged in a row, where the first two rows corresponds with the input (a),
    the third row represents the ground truth middle frame (b), the fourth row represents
    the prediction of the network (output) and the final row contains the difference image
    between the ground truth and the network output.

    Parameters:
    -----------
    a: [num_samples, image_height, image_width, num_channels] numpy array containing the
            input image pairs
    b: [num_samples, image_height, image_width, num_channels] numpy array containing the
            ground truth images
    output: [num_samples, image_height, image_width, num_channels] numpy array containing
            the output images

    Returns:
    -----------
    image_grid: A grid of 5 x num_samples images

    '''

    [num_samples, height, width, num_channels] = a.shape
    num_rows = ((height + 1) * 9) + 1
    num_cols = ((width + 1) * num_samples) + 1
    image_grid = np.ones((num_rows, num_cols, 3)) * 255

    for i in range(num_samples):

        #input A
        image_grid[((width + 1) * 0) + 1:(width + 1) * 1,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = a[i,:,:,:3] * 255
        #ground truth
        image_grid[((width + 1) * 1) + 1:(width + 1) * 2,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = b[i,:,:,:] * 255
        #input C
        image_grid[((width + 1) * 2) + 1:(width + 1) * 3,
                     ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = a[i,:,:,3:] * 255

        #soft mask
        image_grid[((width + 1) * 3) + 1:(width + 1) * 4,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),0] = np.reshape(mask[i,:,:],(height,width)) * 255
        image_grid[((width + 1) * 3) + 1:(width + 1) * 4,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),1] = np.reshape(mask[i,:,:],(height,width)) * 255
        image_grid[((width + 1) * 3) + 1:(width + 1) * 4,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),2] = np.reshape(mask[i,:,:],(height,width)) * 255
        #prediction
        image_grid[((width + 1) * 4) + 1:(width + 1) * 5,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = output[i,:,:,:] * 255
        #difference image
        diff_image = np.absolute(np.uint8(output[i,:,:,:]*255) - np.uint8(b[i,:,:,:]*255))
        image_grid[((width + 1) * 5) + 1:(width + 1) * 6,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = diff_image

        #hard mask
        image_grid[((width + 1) * 6) + 1:(width + 1) * 7,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),0] = np.reshape(mask_t[i,:,:],(height,width)) * 255
        image_grid[((width + 1) * 6) + 1:(width + 1) * 7,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),1] = np.reshape(mask_t[i,:,:],(height,width)) * 255
        image_grid[((width + 1) * 6) + 1:(width + 1) * 7,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),2] = np.reshape(mask_t[i,:,:],(height,width)) * 255
        #prediction
        image_grid[((width + 1) * 7) + 1:(width + 1) * 8,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = output_t[i,:,:,:] * 255
        #difference image
        diff_image = np.absolute(np.uint8(output_t[i,:,:,:]*255) - np.uint8(b[i,:,:,:]*255))
        image_grid[((width + 1) * 8) + 1:(width + 1) * 9,
                    ((height + 1) * i) + 1:(height + 1) * (i + 1),:] = diff_image



    return image_grid
