from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CHANNELS = 3
NUM_CLASSES = 1000

class GoogLeNet(object):
    def __init__(self, num_classes, data_format="NHWC", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    class InceptionModule(tf.keras.layers.Layer):
        """Inception module as a subclass of keras.layers.Layer
        """
        pass

    def inception_module(self, inputs, name):
        """Inception module as a function
        """
        with tf.variable_scope(name):
            pass

    def local_reponse_normalization_layer(self, inputs, k=2, n=5, alpha=0.0001, beta=0.75):
        return tf.nn.local_response_normalization(
            input=inputs,
            depth_radius=n,
            alpha=alpha,
            beta=beta,
            name="local_response_norm"
        )

    def reduction_layer(self, inputs, filters):
        with tf.variable_scope("reduction"):
            kernel = tf.get_variable(
                "weights",
                [1, 1, tf.shape(inputs)[3], filters],
                initializer=tf.glorot_uniform_initializer())
            biases = tf.get_variable("biases", [filters],
                initializer=tf.ones_initializer())
            conv = tf.nn.convolution(
                input=inputs,
                filter=kernel,
                padding="VALID",
                strides=[1, 1],
                name="conv",
                data_format=self.data_format
            )
            return tf.nn.relu(
                tf.nn.bias_add(conv, biases), name="activations")

    def __call__(self, inputs, training):
        with tf.variable_scope("googlenet_model"):
            if self.data_format == "NCHW":
                # Convert inputs from NHWC (channels_last) to NCHW
                # Performance gains on GPU
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            network = None
            with tf.variable_scope("conv1"):
                kernel = tf.get_variable("weights", [7, 7, 3, 64],
                    initializer=tf.glorot_uniform_initializer())
                biases = tf.get_variable("biases", [64],
                    initializer=tf.ones_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="VALID",
                    strides=[2, 2],
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(
                    tf.nn.bias_add(conv, biases), name="activations")
            
            with tf.variable_scope("pool1"):
                network = tf.nn.max_pool(
                    input=network,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="maxpool",
                    data_format=self.data_format
                )

            with tf.variable_scope("norm1"):
                network = self.local_reponse_normalization_layer(network)

            with tf.variable_scope("conv2"):
                network = self.reduction_layer(network, 64)
                kernel = tf.get_variable("weights", [3, 3, 64, 192],
                    initializer=tf.glorot_uniform_initializer())
                biases = tf.get_variable("biases", [192],
                    initializer=tf.ones_initializer())
                conv = tf.nn.convolution(
                    input=network,
                    filter=kernel,
                    padding="VALID",
                    strides=[1, 1],
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(
                    tf.nn.bias_add(conv, biases), name="activations")

            with tf.variable_scope("norm2"):
                network = self.local_reponse_normalization_layer(network)

            with tf.variable_scope("pool2"):
                network = tf.nn.max_pool(
                    input=network,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding="VALID",
                    name="maxpool",
                    data_format=self.data_format
                )