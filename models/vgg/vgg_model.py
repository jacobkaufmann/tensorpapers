from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CHANNELS = 3
NUM_CLASSES = 1000

class Vgg(object):
    def __init__(self, num_classes, version=16, data_format="NCHW", dtype=tf.float32):
        self.num_classes = num_classes
        self.version = version
        self.data_format = data_format
        self.dtype = dtype
        assert self.version in {16, 19}, "Invalid VGG version"

    def vgg_conv_layer(self, inputs, filters, name):
        if self.data_format == "NCHW":
            in_channels = tf.shape(inputs)[1]
        else:
            in_channels = tf.shape(inputs)[3]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channels, filters],
                initializer=tf.glorot_uniform_initializer())
            biases = tf.get_variable("biases", [filters],
                initializer=tf.zeros_initializer())
            conv = tf.nn.convolution(
                input=inputs,
                filter=kernel,
                padding="SAME",
                strides=[1, 1],
                name="conv",
                data_format=self.data_format
            )
        return tf.nn.relu(tf.nn.bias_add(conv, biases), name="activations")

    def vgg_pool_layer(self, inputs, name):
        return tf.nn.max_pool(
            inputs=inputs,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            name=name,
            data_format=self.data_format
        )

    def __call__(self, inputs, training):
        with tf.variable_scope("vgg_model"):
            if self.data_format == "NCHW":
                # Convert inputs from NHWC (channels_last) to NCHW
                # Performance gains on GPU
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            network = None
            with tf.variable_scope("conv1"):
                network = self.vgg_conv_layer(inputs, 64, "c1")
                network = self.vgg_conv_layer(network, 64, "c2")
            with tf.variable_scope("pool1"):
                network = self.vgg_pool_layer(network, "maxpool")
            
            with tf.variable_scope("conv2"):
                network = self.vgg_conv_layer(network, 128, "c1")
                network = self.vgg_conv_layer(network, 128, "c2")
            with tf.variable_scope("pool2"):
                network = self.vgg_pool_layer(network, "maxpool")

            with tf.variable_scope("conv3"):
                network = self.vgg_conv_layer(network, 256, "c1")
                network = self.vgg_conv_layer(network, 256, "c2")
                network = self.vgg_conv_layer(network, 256, "c3")
                if self.version == 19:
                    network = self.vgg_conv_layer(network, 256, "c4")
            with tf.variable_scope("pool3"):
                network = self.vgg_pool_layer(network, "maxpool")

            with tf.variable_scope("conv4"):
                network = self.vgg_conv_layer(network, 512, "c1")
                network = self.vgg_conv_layer(network, 512, "c2")
                network = self.vgg_conv_layer(network, 512, "c3")
                if self.version == 19:
                    network = self.vgg_conv_layer(network, 512, "c4")
            with tf.variable_scope("pool4"):
                network = self.vgg_pool_layer(network, "maxpool")

            with tf.variable_scope("conv5"):
                network = self.vgg_conv_layer(network, 512, "c1")
                network = self.vgg_conv_layer(network, 512, "c2")
                network = self.vgg_conv_layer(network, 512, "c3")
                if self.version == 19:
                    network = self.vgg_conv_layer(network, 512, "c4")
            with tf.variable_scope("pool5"):
                network = self.vgg_pool_layer(network, "maxpool")

            with tf.variable_scope("fc1"):
                shape = network.get_shape().as_list()
                dim = shape[0] * shape[1] * shape[2] * shape[3]
                reshape = tf.reshape(network, [dim])
                weights = tf.get_variable("weights", [dim, 4096],
                    initializer=tf.glorot_uniform_initializer())
                biases = tf.get_variable("biases", [4096],
                    initializer=tf.zeros_initializer())
                network = tf.matmul(network, weights)
                network = tf.nn.tanh(
                    tf.nn.bias_add(network, biases), name="activations")
                if training:
                    network = tf.nn.dropout(network, keep_prob=0.5)

            with tf.variable_scope("fc2"):
                weights = tf.get_variable("weights", [4096, 4096],
                    initializer=tf.glorot_uniform_initializer())
                biases = tf.get_variable("biases", [4096],
                    initializer=tf.zeros_initializer())
                network = tf.matmul(network, weights)
                network = tf.nn.tanh(
                    tf.nn.bias_add(network, biases), name="activations")
                if training:
                    network = tf.nn.dropout(network, keep_prob=0.5)

            with tf.variable_scope("logits"):
                weights = tf.get_variable("weights", [4096, self.num_classes],
                    initializer=tf.glorot_uniform_initializer())
                biases = tf.get_variable("biases", [self.num_classes],
                    initializer=tf.zeros_initializer())
                network = tf.nn.bias_add(tf.matmul(network, weights), biases)

            return network


def vgg_model_fn(features, labels, mode, model_class, momentum=0.9,
                     learning_rate=0.01, weight_decay=0.0005):
    model = model_class(num_classes=NUM_CLASSES)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                "predict": tf.estimator.export.PredictOutput(predictions)
            })

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    tf.identity(cross_entropy, name="cross_entropy")
    tf.summary.scalar("cross_entropy", cross_entropy)

    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    tf.summary.scalar("l2_loss", l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          train_op=train_op)
