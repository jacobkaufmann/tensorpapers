from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Alexnet(object):
    def __init__(self, num_classes, data_format="NHWC", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    def __call__(self, inputs, training):
        with tf.variable_scope("alexnet_model"):
            network = None
            with tf.variable_scope("conv1"):
                kernel = tf.get_variable("weights", [11, 11, 3, 96], 
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [96],
                    initializer=tf.zeros_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="VALID",
                    strides=[4, 4],
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(tf.nn.bias_add(conv, biases))

            with tf.variable_scope("pool2"):
                max_pool = tf.nn.max_pool(
                    input=network,
                    ksize=[1, 3, 3, 3],
                    strides=[1, 2, 2, 3],
                    name="max_pool",
                    data_format=self.data_format
                )
                network = tf.nn.local_response_normalization(
                    input=max_pool,
                    depth_radius=5,
                    alpha=0.0001,
                    beta=0.75
                )

            with tf.variable_scope("conv3"):
                kernel = tf.get_variable("weights", [5, 5, 96, 256],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [256],
                    initializer=tf.ones_initializer())
                conv = tf.nn.convolution(
                    input=network,
                    filter=kernel,
                    padding="VALID",
                    strides=[1, ]
                )
                