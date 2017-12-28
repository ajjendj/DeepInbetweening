#!/usr/bin/env python

from abc import ABCMeta, abstractmethod

import scipy.io
import scipy.misc
import numpy
import os
from glob import glob


class Loader(object):
    '''
    Base class for all classes that load training data from disk.
    '''
    __metaclass__ = ABCMeta


    def __init__(self,
                 data_dir,
                 image_width=None,
                 image_extension='.png'):
        self.data_dir = data_dir
        self.image_extension = image_extension
        self.is_initialized = False
        self.input_to_tensor = None
        self.labels_to_tensor = None

        if image_width is not None:
            self.image_size = (image_width, image_width)
        else:
            self.image_size = None


    def set_tensor_transforms(self,
                              image_pairs_to_tensor,
                              labels_to_tensor):
        '''
        Sets functions to turn read data into tensors acceptable by the CNN
        framework.
        '''
        self.input_to_tensor = image_pairs_to_tensor
        self.labels_to_tensor = labels_to_tensor


    def next_batch_tensors(self, batch_size):
        '''
        Returns (X, Y), where X and Y are tensors corresponding to all the inputs
        in the batch and all outputs in the batch.
        Must run set_tensor_transforms to run this.
        '''
        if self.input_to_tensor is None or self.labels_to_tensor is None:
            raise RuntimeError('No functions to turn output to tensor specified')

        X,Y = self.next_batch(batch_size)
        return (self.input_to_tensor(X), self.labels_to_tensor(Y))


    def next_batch(self, batch_size):
        '''
        Returns (X, Y), where all images are resized to image_size.

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
        Y: [ label_1, ... label_n ] where label is case-dependent
        '''

        if not self.is_initialized:
            self._init_internal()
            self.is_initialized = True
        return self._next_batch_internal(batch_size)

    def train_sample_tensors(self, sample_size):
        '''
        Returns (X, Y), where X and Y are tensors corresponding to all the inputs
        in the random sample of the training set and all outputs in the batch.
        Must run set_tensor_transforms to run this.
        '''
        if self.input_to_tensor is None or self.labels_to_tensor is None:
            raise RuntimeError('No functions to turn output to tensor specified')

        X,Y = self.train_sample(sample_size)
        return (self.input_to_tensor(X), self.labels_to_tensor(Y))


    def train_sample(self, sample_size):
        '''
        Returns (X, Y), where all images are resized to image_size.

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
        Y: [ label_1, ... label_n ] where label is case-dependent
        '''

        if not self.is_initialized:
            self._init_internal()
            self.is_initialized = True
        return self._train_sample_internal(sample_size)


    def _format_image(self, im):
        '''
        Formats image according to loader settings.
        Returns formatted image.
        '''
        if self.image_size is not None:
            return scipy.misc.imresize(im, self.image_size)
        return im


    @abstractmethod
    def _init_internal(self):
        '''
        Walk input directory and do any initalization necessary before
        calling next_batch.
        '''
        pass


    @abstractmethod
    def _next_batch_internal(self, batch_size):
        '''
        Same as next_batch, but assumes that _init_internal has been called.
        '''
        pass


    @staticmethod
    def bw_image_pairs_to_2_channel_tensor(pairs):
        if len(pairs) == 0:
            return []
        bsize = len(pairs)
        b = numpy.asarray([numpy.swapaxes(numpy.swapaxes(numpy.asarray([pairs[i][0], pairs[i][1]]), 0, 2), 0, 1) for i in range(bsize)])
        return b


    @staticmethod
    def float_pairs_to_2_channel_tensor(pairs):
        if len(pairs) == 0:
            return []
        bsize = len(pairs)
        b = numpy.asarray(pairs)
        return numpy.reshape(b, (bsize, 1, 2))

    @staticmethod
    def bw_image_to_1_channel_tensor(images):
        if len(images) == 0:
            return []
        bsize = len(images)
        b = numpy.asarray([images[i] for i in range(bsize)])
        return b

    @staticmethod
    def col_image_pairs_to_6_channel_tensor(pairs):
        if len(pairs) == 0:
            return []
        bsize = len(pairs)
        b_temp = numpy.asarray([numpy.concatenate([pairs[i][0], pairs[i][1]], axis=2) for i in range(bsize)])
        b = numpy.stack(b_temp, axis=0)
        return b

    @staticmethod
    def col_image_to_3_channel_tensor(images):
        if len(images) == 0:
            return []
        bsize = len(images)
        b = numpy.stack(images, axis=0)
        return b

    # TODO(ajjen): add static utility methods for turning other common input/output
    # object types into tensors. Test these on toy examples in loaders_test.py

# CONCRETE SUBCLASSES ----------------------------------------------------------
class TranslationLoader(Loader):
    '''
    The TranslationLoader class loads data/labels used to train a network
    to learn translations between two images. The data is encoded as .png
    images where the first channel corresponds to the first image and the
    second (and third) channel corresponds to the translated image. The label
    is stored as .csv files, containing the translation vector.
    '''

    def __init__(self,
                 data_dir,
                 image_width=None,
                 image_extension='.png'):
        Loader.__init__(self, data_dir, image_width, image_extension)
        self.set_tensor_transforms(
            Loader.bw_image_pairs_to_2_channel_tensor,
            Loader.float_pairs_to_2_channel_tensor)

    def _init_internal(self):
        '''
        Walks the input directory and populates a list of filenames,
        one corresponding to Data and one corresponding to labels.
        Initializes the variable batch_index, which defines which batch of
        Data/labels is loaded using the _next_batch_internal() method.
        '''

        self.index = 0
        self.fnames_X = []
        self.fnames_Y = []
        for dir,_,_ in os.walk(self.data_dir):
            self.fnames_X.extend(glob(os.path.join(dir, '*' + self.image_extension)))
            self.fnames_Y.extend(glob(os.path.join(dir, '*.csv')))
        return


    def _next_batch_internal(self, batch_size):
        '''
        Reads and returns a batch of Data and its corresponding labels. The batch
        depends on the attribute batch_index and the argument batch_size.

        Args:
        batch_size -- Number of data files in a batch

                Returns -- (X, Y)

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
                where (frameA_i, frameC_i) is a tuple of numpy arrays corresponding
                to the input images
        Y: [ label_1, ... label_n ]
                where label_i is a list of numpy arrays encoding the translation
                between frameA_i and frameC_i

        '''

        batch_filesX = self.fnames_X[self.index:self.index+batch_size]
        batch_filesY = self.fnames_Y[self.index:self.index+batch_size]
        self.index += batch_size
        self.index = self.index % (len(self.fnames_X) - batch_size + 1)

        batchX = numpy.asarray([self._format_image(scipy.misc.imread(batch_file))
                                for batch_file in batch_filesX])
        frameA = batchX[:,:,:,0]/255.0
        frameC = batchX[:,:,:,2]/255.0

        X = [(frameA[i,:,:],frameC[i,:,:]) for i in range(batch_size)]
        Y = numpy.asarray([numpy.loadtxt(batch_file) for batch_file in batch_filesY])/10.0

        return X, Y


class ToyLoader(Loader):
    '''
    The ToyLoader class loads data/labels used to train a network
    to learn translations between two images. The data is encoded as .png
    images where the first channel corresponds to the first image and the
    second (and third) channel corresponds to the translated image. The label
    is stored as .csv files, containing the translation vector.
    '''

    def __init__(self,
                 data_dir,
                 image_width=None,
                 image_extension='.png'):
        Loader.__init__(self, data_dir, image_width, image_extension)
        self.set_tensor_transforms(
            Loader.bw_image_pairs_to_2_channel_tensor,
            Loader.float_pairs_to_2_channel_tensor)

    def _init_internal(self):
        '''
        Walks the input directory and populates a list of filenames,
        one corresponding to Data and one corresponding to labels.
        Initializes the variable batch_index, which defines which batch of
        Data/labels is loaded using the _next_batch_internal() method.
        '''

        self.index = 0
        self.fnames_X = []
        self.fnames_Y = []
        self.fnames_X.extend(glob(os.path.join(self.data_dir, '*' + self.image_extension)))
        self.fnames_Y.extend(glob(os.path.join(self.data_dir, '*.csv')))
        return


    def _next_batch_internal(self, batch_size):
        '''
        Reads and returns a batch of Data and its corresponding labels. The batch
        depends on the attribute batch_index and the argument batch_size.

        Args:
        batch_size -- Number of data files in a batch

                Returns -- (X, Y)

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
                where (frameA_i, frameC_i) is a tuple of numpy arrays corresponding
                to the input images
        Y: [ label_1, ... label_n ]
                where label_i is a list of numpy arrays encoding the translation
                between frameA_i and frameC_i

        '''

        batch_filesX = self.fnames_X[self.index:self.index+batch_size]
        batch_filesY = self.fnames_Y[self.index:self.index+batch_size]
        self.index += batch_size
        self.index = self.index % (len(self.fnames_X) - batch_size + 1)

        batchX = numpy.asarray([self._format_image(scipy.misc.imread(batch_file))
                                for batch_file in batch_filesX])
        frameA = (batchX[:,:,:,0] - numpy.mean(batchX[:,:,:,0]))/255.0
        frameC = (batchX[:,:,:,1] - numpy.mean(batchX[:,:,:,1]))/255.0

        X = [(frameA[i,:,:],frameC[i,:,:]) for i in range(batch_size)]
        Y = numpy.asarray([numpy.loadtxt(batch_file) for batch_file in batch_filesY])/10.0


        #classification experiment
        #Yc = numpy.asarray([[1,0] if numpy.sum(numpy.loadtxt(batch_file)) > 0 else [0,1] for batch_file in batch_filesY])

        return X, Y

class ToySTNLoader(Loader):
    '''
    The ToySTNLoader class loads data/labels used to train a network
    to learn translations between two images. The data is encoded as .png
    images where the first channel corresponds to the first image and the
    second (and third) channel corresponds to the translated image. The label
    is stored as .csv files, containing the translation vector.
    '''

    def __init__(self,
                 data_dir,
                 image_width=None,
                 image_extension='.png'):
        Loader.__init__(self, data_dir, image_width, image_extension)
        self.set_tensor_transforms(
            Loader.bw_image_pairs_to_2_channel_tensor,
            Loader.bw_image_to_1_channel_tensor)

    def _init_internal(self):
        '''
        Walks the input directory and populates a list of filenames,
        one corresponding to Data and one corresponding to labels.
        Initializes the variable batch_index, which defines which batch of
        Data/labels is loaded using the _next_batch_internal() method.
        '''

        self.index = 0
        self.fnames_X = []
        self.fnames_Y = []
        self.fnames_X.extend(glob(os.path.join(self.data_dir, '*' + self.image_extension)))
        self.fnames_Y.extend(glob(os.path.join(self.data_dir, '*.csv')))
        return


    def _next_batch_internal(self, batch_size):
        '''
        Reads and returns a batch of Data and its corresponding labels. The batch
        depends on the attribute batch_index and the argument batch_size.

        Args:
        batch_size -- Number of data files in a batch

                Returns -- (X, Y)

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
                where (frameA_i, frameC_i) is a tuple of numpy arrays corresponding
                to the input images
        Y: [ label_1, ... label_n ]
                where label_i is a list of numpy arrays encoding the translation
                between frameA_i and frameC_i

        '''

        batch_filesX = self.fnames_X[self.index:self.index+batch_size]
        batch_filesY = self.fnames_Y[self.index:self.index+batch_size]
        self.index += batch_size
        self.index = self.index % (len(self.fnames_X) - batch_size + 1)

        batchX = numpy.asarray([self._format_image(scipy.misc.imread(batch_file))
                                for batch_file in batch_filesX])
        frameA = (batchX[:,:,:,0])/255.0
        frameB = (batchX[:,:,:,1])/255.0
        frameC = (batchX[:,:,:,2])/255.0
        X = [(frameA[i,:,:],frameC[i,:,:]) for i in range(batch_size)]
        Y = [frameB[i,:,:] for i in range(batch_size)]

        return X, Y


class BWImageTripletsLoader(Loader):
    '''
    The BWImageTripletsLoader class loads data/labels used to train a network
    to learn the middle frame between two images. The data is encoded as .png
    images where the first and third channel corresponds to the first and third
    images respectively, and the second channel corresponds to the middle image.
    '''

    def __init__(self,
                 data_dir,
                 image_width=None,
                 image_extension='.png'):
        Loader.__init__(self, data_dir, image_width, image_extension)

    def _init_internal(self):
        '''
        Walks the input directory and populates a list of filenames,
        one corresponding to Data and one corresponding to labels.
        Initializes the variable batch_index, which defines which batch of
        Data/labels is loaded using the _next_batch_internal() method.
        '''

        self.index = 0
        self.fnames = []
        for dir,_,_ in os.walk(self.data_dir):
                self.fnames.extend(glob(os.path.join(dir, '*' + self.image_extension)))
        return

    def _next_batch_internal(self, batch_size):
        '''
        Reads and returns a batch of Data and its corresponding labels. The batch
        depends on the attribute batch_index and the argument batch_size.

        Args:
        batch_size -- Number of data files in a batch

                Returns -- (X, Y)

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
                where (frameA_i, frameC_i) is a tuple of numpy arrays corresponding
                to the input images
        Y: [ label_1, ... label_n ]
                where label_i is a numpy array corresponding to the middle image

        '''

        batch_files = self.fnames[self.index:self.index+batch_size]
        self.index += batch_size
        self.index = self.index % (len(self.fnames) - 1)

        batchX = numpy.asarray([self._format_image(scipy.misc.imread(batch_file))
                                for batch_file in batch_files])
        frameA = (batchX[:,:,:,0] - numpy.mean(batchX[:,:,:,0]))/255.0
        frameB = (batchX[:,:,:,1] - numpy.mean(batchX[:,:,:,1]))/255.0
        frameC = (batchX[:,:,:,2] - numpy.mean(batchX[:,:,:,2]))/255.0
        X = [(frameA[i,:,:],frameC[i,:,:]) for i in range(batch_size)]
        Y = [frameB[i,:,:] for i in range(batch_size)]

        return X, Y


class ColorImageLoader(Loader):
    '''
    The ColorImageLoader class loads data/labels used to train a network
    to learn the middle frame between two images. All three frames are
    separate .png images.
    '''

    def __init__(self,
                 data_dir,
                 image_width=None,
                 image_extension='.png'):
        Loader.__init__(self, data_dir, image_width, image_extension)
        self.set_tensor_transforms(
            Loader.col_image_pairs_to_6_channel_tensor,
            Loader.col_image_to_3_channel_tensor)

    def _init_internal(self):
        '''
        Walks the input directory and populates a list of filenames,
        one corresponding to Data and one corresponding to labels.
        Initializes the variable batch_index, which defines which batch of
        Data/labels is loaded using the _next_batch_internal() method.
        '''

        self.index = 0
        self.fnames = []
        self.fnames.extend(glob(os.path.join(self.data_dir, '*A' + self.image_extension)))

        #shuffle file names
        numpy.random.seed(0)
        numpy.random.shuffle(self.fnames)

        #random list of indexes (to be used for _train_sample_internal())
        self.fnames_randind = range(len(self.fnames))
        numpy.random.seed(1)
        numpy.random.shuffle(self.fnames_randind)

        print "loader init called"

        return

    def _next_batch_internal(self, batch_size):
        '''
        Reads and returns a batch of Data and its corresponding labels. The batch
        depends on the attribute batch_index and the argument batch_size.

        Args:
        batch_size -- Number of data files in a batch

                Returns -- (X, Y)

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
                where (frameA_i, frameC_i) is a tuple of numpy arrays corresponding
                to the input images
        Y: [ label_1, ... label_n ]
                where label_i is a numpy array corresponding to the middle image

        Note:
        For grayscale images, we were encoding all three frames of a triplet into 1 .png file.
        For color images however, we are encoding the frames as 3 separate images, the first frame
        ends with 'A.png', the second frame ends with 'B.png' and the last frame ends with
        'C.png' with the rest of the filename the same.

        '''

        self.index = self.index % (len(self.fnames) - batch_size + 1)
        batch_files = self.fnames[self.index:self.index+batch_size]
        self.index += batch_size

        batchX_A = numpy.asarray([self._format_image(scipy.misc.imread(batch_file))
                                for batch_file in batch_files])
        batchX_B = numpy.asarray([self._format_image(scipy.misc.imread(batch_file[:-5]+'B'+self.image_extension))
                                for batch_file in batch_files])
        batchX_C = numpy.asarray([self._format_image(scipy.misc.imread(batch_file[:-5]+'C'+self.image_extension))
                                for batch_file in batch_files])
        frame_A = batchX_A/255.0
        frame_B = batchX_B/255.0
        frame_C = batchX_C/255.0

        X = [(frame_A[i], frame_C[i]) for i in range(batch_size)]
        Y = [frame_B[i] for i in range(batch_size)]

        return X, Y

    def _train_sample_internal(self, sample_size):
        '''
        Reads and returns a sample of training Data and its corresponding labels,
        on which to compute an approximation of the train loss.

        Args:
        sample_size -- Number of data files in the sampling of the training set

                Returns -- (X, Y)

        X: [ (frameA_1, frameC_1), ... (frameA_n, frameC_n) ]
                where (frameA_i, frameC_i) is a tuple of numpy arrays corresponding
                to the input images
        Y: [ label_1, ... label_n ]
                where label_i is a numpy array corresponding to the middle image

        Note:
        For grayscale images, we were encoding all three frames of a triplet into 1 .png file.
        For color images however, we are encoding the frames as 3 separate images, the first frame
        ends with 'A.png', the second frame ends with 'B.png' and the last frame ends with
        'C.png' with the rest of the filename the same.

        '''

        batch_files = [self.fnames[i] for i in self.fnames_randind[:sample_size]]

        batchX_A = numpy.asarray([self._format_image(scipy.misc.imread(batch_file))
                                for batch_file in batch_files])
        batchX_B = numpy.asarray([self._format_image(scipy.misc.imread(batch_file[:-5]+'B'+self.image_extension))
                                for batch_file in batch_files])
        batchX_C = numpy.asarray([self._format_image(scipy.misc.imread(batch_file[:-5]+'C'+self.image_extension))
                                for batch_file in batch_files])
        frame_A = batchX_A/255.0
        frame_B = batchX_B/255.0
        frame_C = batchX_C/255.0

        X = [(frame_A[i], frame_C[i]) for i in range(sample_size)]
        Y = [frame_B[i] for i in range(sample_size)]

        return X, Y
