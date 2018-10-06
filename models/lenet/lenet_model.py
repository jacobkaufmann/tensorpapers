from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from math import ceil

class Lenet(object):
    def __init__(self, num_classes, data_format="NHWC", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    def __call__(self, inputs, training):
        with tf.variable_scope("lenet_model"):
            if self.data_format == "NCHW":
                # Convert inputs from NHWC (channels_last) to NCHW
                # Performance gains on GPU
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            network = None
            with tf.variable_scope("c1"):
                kernel = tf.get_variable("weights", [5, 5, 1, 6],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [6],
                    initializer=tf.zeros_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="VALID",
                    strides=[1, 1],
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.sigmoid(tf.nn.bias_add(conv, biases))

            with tf.variable_scope("s2"):
                network = tf.nn.avg_pool(
                    value=network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    name="subsample",
                    padding="VALID"
                )

            with tf.variable_scope("c3"):
                kernel = tf.get_variable("weights", [5, 5, 6, 16],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [16],
                    initializer=tf.zeros_initializer())
                conv = tf.nn.convolution(
                    input=network,
                    filter=kernel,
                    padding="VALID",
                    strides=[1, 1],
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.sigmoid(tf.nn.bias_add(conv, biases))

            with tf.variable_scope("s4"):
                network = tf.nn.avg_pool(
                    value=network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    name="subsample",
                    padding="VALID"
                )

            with tf.variable_scope("c5"):
                kernel = tf.get_variable("weights", [5, 5, 16, 120],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [120],
                    initializer=tf.zeros_initializer())
                conv = tf.nn.convolution(
                    input=network,
                    filter=kernel,
                    padding="VALID",
                    strides=[1, 1],
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.sigmoid(tf.nn.bias_add(conv, biases))

            with tf.variable_scope("f6"):
                network = tf.squeeze(network)
                weights = tf.get_variable("weights", [120, 84],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [84],
                    initializer=tf.zeros_initializer())
                fc = tf.matmul(network, weights)
                network = tf.nn.tanh(tf.nn.bias_add(fc, biases))
                            
            with tf.variable_scope("logits"):
                weights = tf.get_variable("weights", [84, self.num_classes],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [self.num_classes],
                    initializer=tf.zeros_initializer())
                network = tf.nn.bias_add(tf.matmul(network, weights), biases)

            return network
