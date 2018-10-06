from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CHANNELS = 3
NUM_CLASSES = 1000

class Alexnet(object):
    def __init__(self, num_classes, data_format="NHWC", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    def local_reponse_normalization_layer(self, inputs, k=2, n=5, alpha=0.0001, beta=0.75):
        return tf.nn.local_response_normalization(
            input=inputs,
            depth_radius=n,
            alpha=alpha,
            beta=beta,
            name="local_response_norm"
        )

    def __call__(self, inputs, training):
        with tf.variable_scope("alexnet_model"):
            if self.data_format == "NCHW":
                # Convert inputs from NHWC (channels_last) to NCHW
                # Performance gains on GPU
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

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
                network = tf.nn.relu(tf.nn.bias_add(
                    conv, biases), name="activations")

            with tf.variable_scope("pool1"):
                network = tf.nn.max_pool(
                    inputs=network,
                    ksize=[1, 3, 3, 3],
                    strides=[1, 2, 2, 3],
                    padding="VALID",
                    name="maxpool",
                    data_format=self.data_format
                )

            with tf.variable_scope("norm1"):
                network = self.local_reponse_normalization_layer(
                    inputs=network)

            with tf.variable_scope("conv2"):
                kernel = tf.get_variable("weights", [5, 5, 96, 256],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [256],
                    initializer=tf.ones_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="SAME",
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(tf.nn.bias_add(
                    conv, biases), name="activations")

            with tf.variable_scope("pool2"):
                network = tf.nn.max_pool(
                    inputs=network,
                    ksize=[1, 3, 3, 3],
                    strides=[1, 2, 2, 3],
                    padding="VALID",
                    name="maxpool",
                    data_format=self.data_format
                )
            
            with tf.variable_scope("norm2"):
                network = self.local_reponse_normalization_layer(
                    inputs=network)

            with tf.variable_scope("conv3"):
                kernel = tf.get_variable("weights", [3, 3, 256, 384],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [384],
                    initializer=tf.zeros_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="SAME",
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(tf.nn.bias_add(
                    conv, biases), name="activations")
            
            with tf.variable_scope("conv4"):
                kernel = tf.get_variable("weights", [3, 3, 384, 192],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [192],
                    initializer=tf.ones_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="SAME",
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(tf.nn.bias_add(
                    conv, biases), name="activations")

            with tf.variable_scope("conv5"):
                kernel = tf.get_variable("weights", [3, 3, 192, 256],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [256],
                    initializer=tf.ones_initializer())
                conv = tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    padding="SAME",
                    name="conv",
                    data_format=self.data_format
                )
                network = tf.nn.relu(tf.nn.bias_add(
                    conv, biases), name="activations")
            
            with tf.variable_scope("pool3"):
                network = tf.nn.max_pool(
                    inputs=network,
                    ksize=[1, 3, 3, 3],
                    strides=[1, 2, 2, 3],
                    padding="VALID",
                    name="maxpool",
                    data_format=self.data_format
                )

            with tf.variable_scope("fc1"):
                shape = network.get_shape().as_list()
                dim = shape[0] * shape[1] * shape[2] * shape[3]
                reshape = tf.reshape(network, [dim])
                weights = tf.get_variable("weights", [dim, 2048],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [2048],
                    initializer=tf.ones_initializer())
                network = tf.matmul(network, weights)
                network = tf.nn.relu(
                    tf.nn.bias_add(network, biases), name="activations")
                network = tf.nn.dropout(network, keep_prob=0.5)
            
            with tf.variable_scope("fc2"):
                weights = tf.get_variable("weights", [2048, 2048],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [2048],
                    initializer=tf.ones_initializer())
                network = tf.matmul(network, weights)
                network = tf.nn.relu(
                    tf.nn.bias_add(network, biases), name="activations")
                network = tf.nn.dropout(network, keep_prob=0.5)
            
            with tf.variable_scope("logits"):
                weights = tf.get_variable("weights", [2048, self.num_classes],
                    initializer=tf.random_normal_initializer())
                biases = tf.get_variable("biases", [self.num_classes],
                    initializer=tf.ones_initializer())
                network = tf.nn.bias_add(tf.matmul(network, weights), biases)

        return network
    

def alexnet_model_fn(features, labels, mode, model_class, momentum=0.9,
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
