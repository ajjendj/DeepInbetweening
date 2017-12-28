#!usr/bin/env python

import unittest

import train.loaders as loaders

# To run test: python -m unittest -v train.loaders_test

class TestLoader(unittest.TestCase):

    def test_cannot_instantiate(self):
        try:
            loader = loaders.Loader()
            self.fail("Should fail to instantiate base class Loader")
        except TypeError:
            pass

    #Tests for the TranslationLoader Concrete Classs
    def test_next_batch_TranslationLoader(self):
    	loader = loaders.TranslationLoader('./testdata/train/loaders/translation/test/')
        batch_size = 1
        X, Y = loader.next_batch(batch_size)
    	self.assertTrue(len(X) == batch_size)
    	self.assertTrue(len(X[0]) == 2)
    	self.assertTrue(X[0][0].shape == (64,64))
    	self.assertTrue(X[0][1].shape == (64,64))
    	self.assertTrue(len(Y) == batch_size )
    	self.assertTrue(Y[0].shape == (2,))

    def test_next_batch_tensors_TranslationLoader(self):
        loader = loaders.TranslationLoader('./testdata/train/loaders/translation/test/')
        batch_size = 1
        X, Y = loader.next_batch_tensors(batch_size)
        self.assertTrue(X.shape == (batch_size,64,64,2))
        self.assertTrue(Y.shape == (batch_size,1,2))

    def test_batch_index_update_TranslationLoader(self):
    	loader = loaders.TranslationLoader('./testdata/train/loaders/translation/')
    	loader._init_internal()
    	batch_size = 2
    	X, Y = loader._next_batch_internal(batch_size)
    	self.assertTrue(loader.index == 2)

    def test_image_resize_TranslationLoader(self):
    	loader = loaders.TranslationLoader('./testdata/train/loaders/translation/', image_width=32)
    	loader._init_internal()
    	batch_size = 2
    	X, Y = loader._next_batch_internal(batch_size)
    	self.assertTrue(loader.image_size==(32,32))
    	self.assertTrue(X[0][0].shape==(32,32))

    def test_pixel_values(self):
        loader = loaders.TranslationLoader('./testdata/train/loaders/translation/')
        loader._init_internal()
        batch_size = 1
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(int(X[0][0][2,1]*255) == 67)
        self.assertTrue(X[0][1][2,1]*255 == 255)
        self.assertTrue(Y[0][0]*10 == -5)
        self.assertTrue(Y[0][1]*10 == 2)
        self.assertTrue(X[0][0][3,2] == X[0][1][1,7])

    #Tests for the ToyLoader Concrete Class
    def test_instantiate_ToyLoader(self):
        loader = loaders.TranslationLoader('./testdata/train/loaders/toy/train')
        loader._init_internal()
        self.assertTrue(len(loader.fnames_X) > 0)

    def test_next_batch_ToyLoader(self):
        loader = loaders.TranslationLoader('./testdata/train/loaders/toy/test/')
        batch_size = 1
        X, Y = loader.next_batch(batch_size)
        self.assertTrue(len(X) == batch_size)
        self.assertTrue(len(X[0]) == 2)
        self.assertTrue(X[0][0].shape == (32,32))
        self.assertTrue(X[0][1].shape == (32,32))
        self.assertTrue(len(Y) == batch_size )
        self.assertTrue(Y[0].shape == (2,))

    def test_next_batch_tensors_ToyLoader(self):
        loader = loaders.TranslationLoader('./testdata/train/loaders/toy/train/')
        batch_size = 1
        X, Y = loader.next_batch_tensors(batch_size)
        self.assertTrue(X.shape == (batch_size,32,32,2))
        self.assertTrue(Y.shape == (batch_size,1,2))

    def test_batch_index_update_ToyLoader(self):
        loader = loaders.TranslationLoader('./testdata/train/loaders/toy/train')
        loader._init_internal()
        batch_size = 2
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(loader.index == 2)
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(loader.index == 4)
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(loader.index == 6)
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(loader.index == 8)
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(loader.index == 0)


    #Tests for the BWImageTripletsLoader Concrete Class
    def test_instantiate_BWImageTripletsLoader(self):
    	loader = loaders.BWImageTripletsLoader('./testdata/train/loaders/bwimagetriplets/')
    	loader._init_internal()
    	self.assertTrue(len(loader.fnames) > 0)


    def test_next_batch_BWImageLoader(self):
    	loader = loaders.BWImageTripletsLoader('./testdata/train/loaders/translation/')
    	loader._init_internal()
    	batch_size = 2
    	X, Y = loader._next_batch_internal(batch_size)
    	self.assertTrue(len(X) == batch_size)
    	self.assertTrue(len(X[0]) == 2)
    	self.assertTrue(X[0][0].shape == (64,64))
    	self.assertTrue(X[0][1].shape == (64,64))
    	self.assertTrue(len(Y) == batch_size)
    	self.assertTrue(Y[0].shape == (64,64))

    def test_batch_index_update_BWImageLoader(self):
    	loader = loaders.BWImageTripletsLoader('./testdata/train/loaders/bwimagetriplets/')
    	loader._init_internal()
    	batch_size = 2
    	X, Y = loader._next_batch_internal(batch_size)
    	self.assertTrue(loader.index == 2)

    def test_image_resize_BWImageLoader(self):
    	loader = loaders.BWImageTripletsLoader('./testdata/train/loaders/translation/', image_width=32)
    	loader._init_internal()
    	batch_size = 2
    	X, Y = loader._next_batch_internal(batch_size)
    	self.assertTrue(loader.image_size==(32,32))
    	self.assertTrue(X[0][0].shape == (32,32))

    #Tests for the ColorImageTripletsLoader Concrete Class
    def test_instantiate_ColorImageLoader(self):
        loader = loaders.ColorImageLoader('./testdata/train/loaders/toySTN_color/train/')
        loader._init_internal()
        self.assertTrue(len(loader.fnames) > 0)


    def test_next_batch_internal_ColorImageLoader(self):
        loader = loaders.ColorImageLoader('./testdata/train/loaders/toySTN_color/train/')
        loader._init_internal()
        batch_size = 4
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(len(X) == batch_size)
        self.assertTrue(len(X[0]) == 2)
        self.assertTrue(X[0][0].shape == (32,32,3))
        self.assertTrue(X[0][1].shape == (32,32,3))
        self.assertTrue(len(Y) == batch_size)
        self.assertTrue(Y[0].shape == (32,32,3))

    def test_batch_index_update_ColorImageLoader(self):
        loader = loaders.ColorImageLoader('./testdata/train/loaders/bwimagetriplets/')
        loader._init_internal()
        batch_size = 2
        X, Y = loader._next_batch_internal(batch_size)
        self.assertTrue(loader.index == 2)

    def test_next_batch_ColorImageLoader(self):
        loader = loaders.ColorImageLoader('./testdata/train/loaders/toySTN_color/train/')
        loader._init_internal()
        batch_size = 4
        X, Y = loader.next_batch_tensors(batch_size)
        self.assertTrue(X.shape) == (batch_size,32,32,6)
        self.assertTrue(Y.shape) == (batch_size,32,32,3)


    def test_pixel_value(self):
        pass


if __name__ == '__main__':
   unittest.main()
