from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CHANNELS = 3
NUM_CLASSES = 1000

def vgg_conv_layer(inputs, filters):
    return tf.layers.conv2d(
        inputs=inputs,
        kernel_size=[3, 3],
        filters=filters,
        strides=1,
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer,
        bias_initializer=tf.zeros_initializer
    )

def vgg_pool_layer(inputs):
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=2
    )

class VGG16(object):
    def __init__(self, num_classes, data_format="channels_first", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    def __call__(self, inputs, training):
        with tf.variable_scope("alexnet_model"):
            inputs = vgg_conv_layer(inputs, 64)
            inputs = vgg_conv_layer(inputs, 64)
            inputs = vgg_pool_layer(inputs)
            
            inputs = vgg_conv_layer(inputs, 128)
            inputs = vgg_conv_layer(inputs, 128)
            inputs = vgg_pool_layer(inputs)

            inputs = vgg_conv_layer(inputs, 256)
            inputs = vgg_conv_layer(inputs, 256)
            inputs = vgg_conv_layer(inputs, 256)
            inputs = vgg_pool_layer(inputs)

            inputs = vgg_conv_layer(inputs, 512)
            inputs = vgg_conv_layer(inputs, 512)
            inputs = vgg_conv_layer(inputs, 512)
            inputs = vgg_pool_layer(inputs)

            inputs = vgg_conv_layer(inputs, 512)
            inputs = vgg_conv_layer(inputs, 512)
            inputs = vgg_conv_layer(inputs, 512)
            inputs = vgg_pool_layer(inputs)

            inputs = tf.layers.dense(
                inputs=inputs,
                units=4096,
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer,
                bias_initializer=tf.zeros_initializer
            )
            inputs = tf.layers.dropout(
                inputs=inputs,
                training=training
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=4096,
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer,
                bias_initializer=tf.zeros_initializer
            )
            inputs = tf.layers.dropout(
                inputs=inputs,
                training=training
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=self.num_classes,
                kernel_initializer=tf.glorot_uniform_initializer,
                bias_initializer=tf.zeros_initializer
            )

            return inputs


def vgg16_model_fn(features, labels, mode, model_class, momentum=0.9,
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
