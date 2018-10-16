from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CHANNELS = 3
NUM_CLASSES = 1000

class GoogLeNet(object):
    def __init__(self, num_classes, data_format="channels_last", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    def inception_module(self, inputs, data_format, filters1x1,
        filters3x3_reduce, filters3x3, filters5x5_reduce, filters5x5,
        pool_proj, name):
        """Inception module as a function
        """
        with tf.variable_scope(name):
            with tf.variable_scope("branch_1x1"):
                branch1x1 = tf.layers.Conv2D(
                    inputs=inputs,
                    filters=filters1x1,
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    activation=tf.nn.relu,
                    name="conv"
                )

            with tf.variable_scope("branch_3x3"):
                branch3x3 = tf.layers.Conv2D(
                    inputs=inputs,
                    filters=filters3x3_reduce,
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    activation=tf.nn.relu,
                    name="reduce"
                )
                branch3x3 = tf.layers.Conv2D(
                    inputs=branch3x3,
                    filters=filters3x3,
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    activation=tf.nn.relu,
                    name="conv"
                )
            
            with tf.variable_scope("branch_5x5"):
                branch5x5 = tf.layers.Conv2D(
                    inputs=inputs,
                    filters=filters5x5_reduce,
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    activation=tf.nn.relu,
                    name="reduce"
                )
                branch5x5 = tf.layers.Conv2D(
                    inputs=branch5x5,
                    filters=filters5x5,
                    kernel_size=(5,5),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    activation=tf.nn.relu,
                    name="conv"
                )

            with tf.variable_scope("branch_pool"):
                branch_pool = tf.layers.MaxPooling2D(
                    inputs=inputs,
                    pool_size=(3,3),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    name="pool"
                )
                branch_pool = tf.layers.Conv2D(
                    inputs=branch_pool,
                    filters=pool_proj,
                    kernel_size=(1,1),
                    strides=(1,1),
                    padding="same",
                    data_format=data_format,
                    activation=tf.nn.relu,
                    name="conv"
                )

            branches = [branch1x1, branch3x3, branch5x5, branch_pool]
            if data_format == "channels_last":
                return tf.concat(concat_dim=3, values=branches)
            else:
                return tf.concat(concat_dim=1, values=branches)

    def auxiliary_network(self, inputs, data_format, name):
        """Auxiliary classifier
        """
        with tf.variable_scope(name):
            aux_network = tf.layers.AveragePooling2D(
                inputs=inputs,
                pool_size=(5,5),
                strides=(3,3),
                padding="valid",
                data_format=data_format,
                name="avg_pool"
            )
            aux_network = tf.layers.Conv2D(
                inputs=aux_network,
                filters=128,
                kernel_size=(1,1),
                strides=(1,1),
                padding="same",
                data_format=data_format,
                activation=tf.nn.relu,
                name="conv"
            )
            aux_network = tf.layers.Dense(
                inputs=aux_network,
                units=1024,
                activation=tf.nn.relu,
                name="fc"
            )
            aux_network = tf.layers.Dropout(
                inputs=aux_network,
                rate=0.7
            )
            aux_network = tf.layers.Dense(
                inputs=aux_network,
                units=self.num_classes,
                activation=None,
                name="logits"
            )
            return aux_network

    def local_reponse_normalization_layer(self, inputs, name, k=2, n=5, 
        alpha=0.0001, beta=0.75):
        """Reusable local response normalization layer
        """
        return tf.nn.local_response_normalization(
            input=inputs,
            depth_radius=n,
            alpha=alpha,
            beta=beta,
            name="local_response_norm"
        )

    def __call__(self, inputs, training):
        with tf.variable_scope("googlenet_model"):
            if self.data_format == "channels_first":
                # Convert inputs from channels_last to channels_first
                # Performance gains on GPU
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            network = tf.layers.Conv2D(
                inputs=inputs,
                filters=64,
                kernel_size=(7,7),
                strides=(2,2),
                padding="valid",
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv1"
            )
            network = tf.layers.MaxPooling2D(
                inputs=network,
                pool_size=(3,3),
                strides=(2,2),
                padding="valid",
                data_format="channels_last",
                name="pool1"
            )
            network = self.local_reponse_normalization_layer(
                inputs=network,
                name="norm1"
            )

            network = tf.layers.Conv2D(
                inputs=network,
                filters=64,
                kernel_size=(1,1),
                strides=(1,1),
                padding="valid",
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv2_reduce"
            )
            network = tf.layers.Conv2D(
                inputs=network,
                filters=192,
                kernel_size=(7,7),
                strides=(1, 1),
                padding="same",
                data_format="channels_last",
                activation=tf.nn.relu,
                name="conv2"
            )
            network = self.local_reponse_normalization_layer(
                inputs=network,
                name="norm2"
            )
            network = tf.layers.MaxPooling2D(
                inputs=network,
                pool_size=(3,3),
                strides=(2,2),
                padding="valid",
                data_format="channels_last",
                name="pool2"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=64,
                filters3x3_reduce=96,
                filters3x3=128,
                filters5x5_reduce=16,
                filters5x5=32,
                pool_proj=32,
                name="module_3a"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=128,
                filters3x3_reduce=128,
                filters3x3=192,
                filters5x5_reduce=32,
                filters5x5=96,
                pool_proj=64,
                name="module_3b"
            )

            network = tf.layers.MaxPooling2D(
                inputs=network,
                pool_size=(3,3),
                strides=(2,2),
                padding="valid",
                data_format="channels_last",
                name="pool3"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=192,
                filters3x3_reduce=96,
                filters3x3=208,
                filters5x5_reduce=16,
                filters5x5=48,
                pool_proj=64,
                name="module_4a"
            )

            aux_logits1 = self.auxiliary_network(
                inputs=network,
                data_format="channels_last",
                name="auxiliary_classifier1"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=160,
                filters3x3_reduce=112,
                filters3x3=224,
                filters5x5_reduce=24,
                filters5x5=64,
                pool_proj=64,
                name="module_4b"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=128,
                filters3x3_reduce=128,
                filters3x3=256,
                filters5x5_reduce=24,
                filters5x5=64,
                pool_proj=64,
                name="module_4c"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=112,
                filters3x3_reduce=144,
                filters3x3=288,
                filters5x5_reduce=32,
                filters5x5=64,
                pool_proj=64,
                name="module_4d"
            )

            aux_logits2 = self.auxiliary_network(
                inputs=network,
                data_format="channels_last",
                name="auxiliary_classifier2"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=256,
                filters3x3_reduce=160,
                filters3x3=320,
                filters5x5_reduce=32,
                filters5x5=128,
                pool_proj=128,
                name="module_4e"
            )

            network = tf.layers.MaxPooling2D(
                inputs=network,
                pool_size=(3,3),
                strides=(2,2),
                padding="valid",
                data_format="channels_last",
                name="pool4"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=256,
                filters3x3_reduce=160,
                filters3x3=320,
                filters5x5_reduce=32,
                filters5x5=128,
                pool_proj=128,
                name="module_5a"
            )

            network = self.inception_module(
                inputs=network,
                data_format="channels_last",
                filters1x1=384,
                filters3x3_reduce=192,
                filters3x3=384,
                filters5x5_reduce=48,
                filters5x5=128,
                pool_proj=128,
                name="module_5b"
            )

            network = tf.layers.AveragePooling2D(
                inputs=network,
                pool_size=(7,7),
                strides=(1,1),
                padding="valid",
                data_format="channels_last",
                name="pool5"
            )
            network = tf.layers.Dropout(
                inputs=network,
                rate=0.4
            )

            network = tf.layers.Dense(
                inputs=network,
                units=1000,
                activation=tf.nn.relu,
                name="fc"
            )

            network = tf.layers.Dense(
                inputs=network,
                units=self.num_classes,
                activation=None,
                name="logits"
            )

            return network
