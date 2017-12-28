#!usr/bin/env python

import collections
import math
import tensorflow as tf
import numpy as np
from viz_util import kernels_to_image
from spatial_transformer import transformer

LayerOpts = collections.namedtuple('LayerOpts', 'filter_size n_kernels')

class EvaluationGraph(object):
    '''
    Responsible for setting up our graph.
    '''

    def __init__(self):
        self.input_tensor = None
        self.truth = None
        self.prediction = None
        self.keep_prob = None #Variable that specifies probability of applying drop out to a neuron
        self.inputA = None #Frame A of the input tensor
        self.inputB = None #Frame B of the input tensor
        self.outputA = None #Predicted Output tensor A
        self.outputB = None #Predicted Output tensor B
        self.alpha_mask = None #Predicted Alpha Mask
        self.output = None #Predicted final output

    def weight_variable(self, shape, name='weights',initializer=tf.truncated_normal_initializer(0.0, stddev=0.01)):
                            #initializer=tf.contrib.layers.xavier_initializer()):
                        #initializer=tf.constant_initializer(0.0)):
                        #initializer=tf.truncated_normal_initializer(0.0, stddev=0.1))
        return tf.get_variable(name, shape, initializer=initializer)


    def bias_variable(self, shape, name='bias',
                      initializer=tf.constant_initializer(0.01),
                      initial_value = None):
        if (initial_value == None):
            return tf.get_variable(name, shape, initializer=initializer)
        else:
            return tf.Variable(name, initial_value=initial_value)


    def get_layer_var(self, layer_name, var_name):
        with tf.variable_scope(layer_name, reuse=True):
            return tf.get_variable(var_name)


    def attach_image_summary_op(self, var):
        with tf.name_scope('images'):
            num_rows = int(math.sqrt(var.get_shape()[3].value))
            image = kernels_to_image(var, num_rows, num_rows, 1)
            self.image = image
            return tf.image_summary(var.name, image, max_images=1)


    def attach_summary_ops(self, var,
                           stats=['mean', 'stddev', 'max', 'min', 'hist']):
        '''Attach summaries to a Tensor. '''

        with tf.name_scope('summary'):
            for stat in stats:
                mean = tf.reduce_mean(var)
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                summary_name = stat + '/' + var.name
                if stat == 'mean':
                    tf.scalar_summary(summary_name, mean)
                elif stat == 'stddev':
                    tf.scalar_summary(summary_name, stddev)
                elif stat == 'max':
                    tf.scalar_summary(summary_name, tf.reduce_max(var))
                elif stat == 'min':
                    tf.scalar_summary(summary_name, tf.reduce_min(var))
                elif stat == 'hist':
                    tf.histogram_summary(summary_name, var)
                else:
                    raise RuntimeError('Unknow stat: ' + str(stat))


    def build_conv_layer(self, input_tensor,
                         filter_tensor_shape,
                         layer_name,
                         strides = [1,2,2,1],
                         act=tf.nn.relu):
        '''Reusable code for making a convolutional neural net layer.

        It does a series of convolutions and uses relu to nonlinearize. It also sets up
        name scoping so that the resultant graph is easy to read, and adds a number of
        summary ops.'''

        with tf.variable_scope(layer_name):
            weights = self.weight_variable(filter_tensor_shape)
            self.attach_summary_ops(weights)

            biases = self.bias_variable([filter_tensor_shape[-1]])
            self.attach_summary_ops(biases)

            preactivate = tf.nn.conv2d(input_tensor, weights,
                                       strides = strides, padding='SAME') + biases
            # preactivate = tf.nn.local_response_normalization(preactivate)
            batch_mean, batch_var = tf.nn.moments(preactivate,[0, 1, 2])
            num_kernels = filter_tensor_shape[3]
            scale = tf.Variable(tf.ones([num_kernels]))
            beta = tf.Variable(tf.zeros([num_kernels]))
            epsilon = 1e-3
            preactivate = tf.nn.batch_normalization(preactivate,batch_mean,batch_var,beta,scale,epsilon)

            self.attach_summary_ops(preactivate, stats=['hist'])

            if act != None:
                activations = act(preactivate, 'activation')
                self.attach_summary_ops(activations, stats=['hist'])
            else:
                activations = preactivate
            return activations


    def build_fc_layer(self, input_tensor, input_dim, output_dim, layer_name, act=None, initial_value=None):
        '''Reusable code for making a fully connected neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        '''
        with tf.variable_scope(layer_name):
            weights = self.weight_variable([input_dim, output_dim])
            self.attach_summary_ops(weights)

            biases = self.bias_variable([output_dim], initial_value=initial_value)
            self.attach_summary_ops(biases)

            preactivate = tf.matmul(input_tensor, weights) + biases
            # preactivate = tf.nn.l2_normalize(preactivate, 0) #Note: Use batch norm instead
            batch_mean, batch_var = tf.nn.moments(preactivate,[0])
            scale = tf.Variable(tf.ones([output_dim]))
            beta = tf.Variable(tf.zeros([output_dim]))
            epsilon = 1e-3
            preactivate = tf.nn.batch_normalization(preactivate,batch_mean,batch_var,beta,scale,epsilon)
            self.attach_summary_ops(preactivate, stats=['hist'])

            if act != None:
                activations = act(preactivate, 'activation')
                self.attach_summary_ops(activations, stats=['hist'])
            else:
                activations = preactivate

            return activations


class TranslateGraph(EvaluationGraph):
    '''
    Graph for the translation framework.
    '''
    def __init__(self):
        pass

    def build(self, image_size, conv_layer_opts=[
        LayerOpts(filter_size = 5, n_kernels = 16),
        LayerOpts(filter_size = 5, n_kernels = 32)],
              fc_layer_sizes=[400, 2]):

        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 2])
        self.truth = tf.placeholder(tf.float32, [None, 1, 2])
        self.prediction = None

        input_tensor = self.input_tensor
        for i in range(0, len(conv_layer_opts)):
            filter_size = conv_layer_opts[i].filter_size
            n_kernels = conv_layer_opts[i].n_kernels
            n_channels = input_tensor.get_shape()[3].value
            layer_name = 'conv_layer%02d' % i
            activations = self.build_conv_layer(
                input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                layer_name)
            input_tensor = activations
        self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights'))

        n_channels = input_tensor.get_shape()[3].value
        activation_dim = input_tensor.get_shape()[2].value
        activations_r = tf.reshape(
            input_tensor, [-1, activation_dim * activation_dim * n_channels])

        input_tensor = activations_r

        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = None
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,
                                              layer_name, act=nonlin)
            input_tensor = activations

        self.prediction = input_tensor

class ToyGraph(EvaluationGraph):
    '''
    Graph for the toy framework.
    '''
    def __init__(self):
        pass

    def build(self, image_size, conv_layer_opts=[
        LayerOpts(filter_size = 5, n_kernels = 16),
        LayerOpts(filter_size = 5, n_kernels = 32)],
        fc_layer_sizes=[400,2]):

        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 2])
        self.truth = tf.placeholder(tf.float32, [None, 1, 2])
        self.prediction = None

        input_tensor = self.input_tensor
        for i in range(0, len(conv_layer_opts)):
            filter_size = conv_layer_opts[i].filter_size
            n_kernels = conv_layer_opts[i].n_kernels
            n_channels = input_tensor.get_shape()[3].value
            layer_name = 'conv_layer%02d' % i
            activations = self.build_conv_layer(
                input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                layer_name)
            input_tensor = activations
        self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights'))

        n_channels = input_tensor.get_shape()[3].value
        activation_dim = input_tensor.get_shape()[2].value
        activations_r = tf.reshape(
            input_tensor, [-1, activation_dim * activation_dim * n_channels])


        input_tensor = activations_r
        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = None
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,
                                              layer_name, act=nonlin)
            input_tensor = activations

        self.prediction = input_tensor

class MnistGraph(EvaluationGraph):
    '''
    Graph for the MNIST framework.
    '''
    def __init__(self):
        pass

    def build(self, image_size, fc_layer_sizes=[10]):

        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
        self.truth = tf.placeholder(tf.float32, [None, 10])
        self.prediction = None

        input_tensor = self.input_tensor


        n_channels = input_tensor.get_shape()[3].value
        activation_dim = input_tensor.get_shape()[2].value
        input_tensor = tf.reshape(
            input_tensor, [-1, activation_dim * activation_dim * n_channels])

        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = None
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,
                                              layer_name, act=nonlin)
            input_tensor = activations

        self.prediction = tf.nn.softmax(input_tensor)

class STNGraph(EvaluationGraph):
    '''
    Graph for a grayscale dataset with a Spatial Transformation Network,
    so that the loss can be computed on the pixels.

    The input to the graph is a 2-frame image of size:
        [num_samples, image_width, image_height, 2]
    This graph consists of a series of convolutional and fully connected
    layers whose final output is six numbers representing the parameters
    of the affine transformation matrix. The affine transformation is
    then applied to the input image, and the transformed image is the o
    output of the graph, which is used to compute the loss. The output
    of the graph is a 2-frame image of size:
        [num_samples, image_width, image_height, 1]
    '''

    def __init__(self):
        pass

    def build(self, image_size, conv_layer_opts=[
        LayerOpts(filter_size = 5, n_kernels = 16),
        LayerOpts(filter_size = 5, n_kernels = 32)],
        fc_layer_sizes=[1000,400]):

        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 2])
        self.truth = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
        input_tensor = self.input_tensor

        if (conv_layer_opts == []):
            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            input_tensor = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

        else:
            for i in range(0, len(conv_layer_opts)):
                filter_size = conv_layer_opts[i].filter_size
                n_kernels = conv_layer_opts[i].n_kernels
                n_channels = input_tensor.get_shape()[3].value
                layer_name = 'conv_layer%02d' % i
                activations = self.build_conv_layer(
                    input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                    layer_name)
                input_tensor = activations
            self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights'))

            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            activations_r = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

            input_tensor = activations_r

        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = tf.nn.tanh
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,layer_name, act=nonlin)
            input_tensor = activations

        #Final layer to regress on the transformation parameters
        W_final = self.weight_variable([fc_layer_sizes[-1], 6], name='W_final')
        initialA = np.array([[1., 0, 0], [0, 1., 0]])
        initialA = initialA.astype('float32')
        initialA = initialA.flatten()
        b_final = tf.Variable(initial_value=initialA, name='b_final')
        activations_final = tf.nn.tanh(tf.matmul(input_tensor, W_final) + b_final)

        self.inputA = tf.reshape(self.input_tensor[:,:,:,0], (-1, image_size, image_size, 1))
        self.outputA = transformer(self.inputA, activations_final, (image_size,image_size))

class STNColorGraph(EvaluationGraph):

    '''
    Graph for a grayscale dataset with a Spatial Transformation Network,
    so that the loss can be computed on the pixels.

    The input to the graph is a 6-frame image of size (Stack of 2 color images):
        [num_samples, image_width, image_height, 6]
    This graph consists of a series of convolutional and fully connected
    layers whose final output is six numbers representing the parameters
    of the affine transformation matrix. The affine transformation is
    then applied to the input image, and the transformed image is the o
    output of the graph, which is used to compute the loss. The output
    of the graph is a 2-frame image of size:
        [num_samples, image_width, image_height, 3]

    '''

    def __init__(self):
        pass

    def build(self, image_size, conv_layer_opts=[
        LayerOpts(filter_size = 5, n_kernels = 16),
        LayerOpts(filter_size = 5, n_kernels = 32)],
        fc_layer_sizes=[1000,400]):

        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 6])
        self.truth = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        input_tensor = self.input_tensor

        if (conv_layer_opts == []):
            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            input_tensor = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

        else:
            for i in range(0, len(conv_layer_opts)):
                filter_size = conv_layer_opts[i].filter_size
                n_kernels = conv_layer_opts[i].n_kernels
                n_channels = input_tensor.get_shape()[3].value
                layer_name = 'conv_layer%02d' % i
                activations = self.build_conv_layer(
                    input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                    layer_name)
                input_tensor = activations
            #self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights'))

            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            activations_r = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

            input_tensor = activations_r

        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = tf.nn.tanh
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,layer_name, act=nonlin)
            input_tensor = activations

        W_final = self.weight_variable([fc_layer_sizes[-1], 6], name='W_final')
        initialA = np.array([[1., 0, 0], [0, 1., 0]])
        initialA = initialA.astype('float32')
        initialA = initialA.flatten()
        b_final = tf.Variable(initial_value=initialA, name='b_final')

        # Add dropout
        self.keep_prob = tf.placeholder(tf.float32)
        input_tensor_drop = tf.nn.dropout(input_tensor, self.keep_prob)
        activations_final = tf.nn.tanh(tf.matmul(input_tensor, W_final) + b_final)

        self.inputA = tf.reshape(self.input_tensor[:,:,:,:3], (-1, image_size, image_size, 3))
        self.outputA = transformer(self.inputA, activations_final, (image_size,image_size))


class STNHybridColorGraph(EvaluationGraph):

    '''
    Graph for a Color dataset with a Spatial Transformation Network,
    so that the loss can be computed on the pixels.

    The input to the graph is a 6-frame image of size (Stack of 2 color images):
        [num_samples, image_width, image_height, 6]
    This graph consists of a series of convolutional and fully connected
    layers whose final output is six numbers representing the parameters
    of the affine transformation matrix, as well as a mask the size of
    the input image. The affine transformation is then applied to the first
    input image, the inverse affine transformation is applied to the second
    image and the combination (defined by the mask) of the transformed image is
    the output of the graph, which is used to compute the loss. The output
    of the graph is a 2-frame image of size:
        [num_samples, image_width, image_height, 3]

    '''

    def __init__(self):
        pass

    def build(self, image_size, conv_layer_opts=[
        LayerOpts(filter_size = 5, n_kernels = 32),
        LayerOpts(filter_size = 5, n_kernels = 64)
        ],
        fc_layer_sizes=[1000,400]):
        self.input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 6])
        self.truth = tf.placeholder(tf.float32, [None, image_size, image_size, 3])

        #transform tower 1
        input_tensor = self.input_tensor
        if (conv_layer_opts == []):
            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            input_tensor = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

        else:
            for i in range(0, len(conv_layer_opts)):
                filter_size = conv_layer_opts[i].filter_size
                n_kernels = conv_layer_opts[i].n_kernels
                n_channels = input_tensor.get_shape()[3].value
                layer_name = 'conv_layer_t1%02d' % i
                activations = self.build_conv_layer(
                    input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                    layer_name)
                input_tensor = activations
            #self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights'))

            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            activations_r = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

            input_tensor = activations_r

        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer_t1%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = tf.nn.tanh
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,layer_name, act=nonlin)
            input_tensor = activations

        W_final_t1 = self.weight_variable([fc_layer_sizes[-1], 6], name='W_final_t1')
        initialA_t1 = np.array([[1., 0, 0], [0, 1., 0]])
        initialA_t1 = initialA_t1.astype('float32')
        initialA_t1 = initialA_t1.flatten()
        b_final_t1 = tf.Variable(initial_value=initialA_t1, name='b_final_t1')

        # Add dropout
        self.keep_prob = tf.placeholder(tf.float32)
        input_tensor_drop_t1 = tf.nn.dropout(input_tensor, self.keep_prob)
        activations_final_t1 = tf.nn.tanh(tf.matmul(input_tensor, W_final_t1) + b_final_t1)

        self.inputA = tf.reshape(self.input_tensor[:,:,:,:3], (-1, image_size, image_size, 3))
        self.outputA = transformer(self.inputA, activations_final_t1, (image_size,image_size))

        #transform tower 2
        input_tensor = self.input_tensor

        if (conv_layer_opts == []):
            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            input_tensor = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

        else:
            for i in range(0, len(conv_layer_opts)):
                filter_size = conv_layer_opts[i].filter_size
                n_kernels = conv_layer_opts[i].n_kernels
                n_channels = input_tensor.get_shape()[3].value
                layer_name = 'conv_layer_t2%02d' % i
                activations = self.build_conv_layer(
                    input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                    layer_name)
                input_tensor = activations
            #self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights')

            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            activations_r = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

            input_tensor = activations_r

        for j in range(0, len(fc_layer_sizes)):
            input_dim = input_tensor.get_shape()[1].value
            output_dim  = fc_layer_sizes[j]
            layer_name = 'fc_layer_t2%02d' % j
            if (j < len(fc_layer_sizes) - 1):
                nonlin =  tf.nn.relu
            else:
                nonlin = tf.nn.tanh
            activations = self.build_fc_layer(input_tensor, input_dim, output_dim,layer_name, act=nonlin)
            input_tensor = activations

        W_final_t2 = self.weight_variable([fc_layer_sizes[-1], 6], name='W_final_t2')
        initialA_t2 = np.array([[1., 0, 0], [0, 1., 0]])
        initialA_t2 = initialA_t2.astype('float32')
        initialA_t2 = initialA_t2.flatten()
        b_final_t2 = tf.Variable(initial_value=initialA_t2, name='b_final_t2')

        # Add dropout
        input_tensor_drop_t2 = tf.nn.dropout(input_tensor, self.keep_prob)
        activations_final_t2 = tf.nn.tanh(tf.matmul(input_tensor, W_final_t2) + b_final_t2)

        self.inputB = tf.reshape(self.input_tensor[:,:,:,3:], (-1, image_size, image_size, 3))
        self.outputB = transformer(self.inputB, activations_final_t2, (image_size,image_size))

        #alpha tower
        input_tensor = tf.concat(3, [self.outputA, self.outputB])
        input_tensor = tf.reshape(input_tensor, (-1, image_size, image_size, 6))
        if (conv_layer_opts == []):
            n_channels = input_tensor.get_shape()[3].value
            activation_dim = input_tensor.get_shape()[2].value
            input_tensor = tf.reshape(
                input_tensor, [-1, activation_dim * activation_dim * n_channels])

        else:
            for i in range(0, len(conv_layer_opts)):
                filter_size = conv_layer_opts[i].filter_size
                n_kernels = conv_layer_opts[i].n_kernels
                n_channels = input_tensor.get_shape()[3].value
                layer_name = 'conv_layer_alpha%02d' % i
                activations = self.build_conv_layer(
                    input_tensor, [filter_size, filter_size, n_channels, n_kernels],
                    layer_name, strides = [1,1,1,1])
                input_tensor = activations
            #self.attach_image_summary_op(self.get_layer_var('conv_layer00', 'weights'))

        #Final convolutional layer to output 2-channel output mask
        filter_size = 7
        n_kernels = 2
        n_channels = input_tensor.get_shape()[3].value
        layer_name = 'conv_layer_t2_final'
        activations = self.build_conv_layer(
            input_tensor, [filter_size, filter_size, n_channels, n_kernels],
            layer_name, strides = [1,1,1,1], act=tf.nn.tanh)

        #compute softmax
        beta = 4.0 #has an effect on the slope of the softmax
        max_axis = tf.reduce_max(activations, 3, keep_dims=True)
        act_exp = tf.exp(beta * (activations - max_axis))
        normalize = tf.reduce_sum(act_exp, 3, keep_dims=True)
        softmax = act_exp / normalize

        self.alpha_mask = softmax
        self.output = tf.mul(tf.reshape(self.alpha_mask[:,:,:,0],[-1,image_size,image_size,1]), self.outputA) + \
                        tf.mul(tf.reshape(self.alpha_mask[:,:,:,1],[-1,image_size,image_size,1]), self.outputB)

        self.alpha_mask_t = tf.select(self.alpha_mask>0.5, tf.ones_like(self.alpha_mask), tf.zeros_like(self.alpha_mask))
        self.output_t = tf.mul(tf.reshape(self.alpha_mask_t[:,:,:,0],[-1,image_size,image_size,1]), self.outputA) + \
                            tf.mul(tf.reshape(self.alpha_mask_t[:,:,:,1],[-1,image_size,image_size,1]), self.outputB)

        #Compute histogram of gradient magnitudes for the ground truth and the prediction
        #   as a penalty against blurred images
        #
        gx_output = self.output[:,1:,1:,:] - self.output[:,1:,:image_size-1,:]
        gy_output = self.output[:,1:,1:,:] - self.output[:,:image_size-1,1:,:]
        gmag_output = tf.sqrt(tf.square(gx_output) + tf.square(gy_output))

        gx_truth = self.truth[:,1:,1:,:] - self.truth[:,1:,:image_size-1,:]
        gy_truth = self.truth[:,1:,1:,:] - self.truth[:,:image_size-1,1:,:]
        gmag_truth = tf.sqrt(tf.square(gx_truth) + tf.square(gy_truth))

        #compute histograms
        truth_grad_hist = tf.histogram_fixed_width(gmag_truth, [-1.0,1.0], nbins = 100)
        self.truth_grad_hist = tf.cast(truth_grad_hist, tf.float32)/tf.cast(tf.reduce_sum(truth_grad_hist), tf.float32)
        output_grad_hist = tf.histogram_fixed_width(gmag_output, [-1.0,1.0], nbins = 100)
        self.output_grad_hist = tf.cast(output_grad_hist, tf.float32)/tf.cast(tf.reduce_sum(output_grad_hist), tf.float32)













