#!usr/bin/env python

import unittest

import train.graphs as graphs

# To run test: python -m unittest -v train.graphs_test

class TestGraph(unittest.TestCase):

    def test_basics(self):
        graph = graphs.TranslateGraph()
        graph.build(image_size=32)
