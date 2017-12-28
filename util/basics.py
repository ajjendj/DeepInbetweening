#!/usr/bin/env python

import os
import sys

def print_error(arg):
    print >> sys.stderr, arg


def repo_root_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, '..')

def testdata_dir(subdir=None):
    return os.path.join(repo_root_dir(), 'testdata', subdir)
