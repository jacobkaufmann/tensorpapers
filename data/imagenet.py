from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import six
import tensorflow as tf

tf.app.flags.DEFINE_string("train_directory", "/tmp/",
                           "Training data directory")
tf.app.flags.DEFINE_string("validation_directory", "/tmp/",
                           "Validation data directory")
tf.app.flags.DEFINE_string("output_directory", "/tmp/",
                           "Output data directory")

tf.app.flags.DEFINE_integer("train_shards", 1024,
                            "Number of shards in training TFRecord files.")
tf.app.flags.DEFINE_integer("validation_shards", 128,
                            "Number of shards in validation TFRecord files.")
tf.app.flags.DEFINE_integer("num_threads", 8,
                            "Number of threads to preprocess the images.")

tf.app.flags.DEFINE_string('labels_file',
                           'imagenet_lsvrc_2015_synsets.txt',
                           'Labels file')
tf.app.flags.DEFINE_string('imagenet_metadata_file',
                           'imagenet_metadata.txt',
                           'ImageNet metadata file')
tf.app.flags.DEFINE_string('bounding_box_file',
                           './imagenet_2012_bounding_boxes.csv',
                           'Bounding box file')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
