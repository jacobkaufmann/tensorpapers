from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CHANNELS = 3
NUM_CLASSES = 1000

def normalization_layer(inputs, k=2, n=5, alpha=0.0001, beta=0.75):
    return tf.nn.local_response_normalization(
        input=inputs,
        depth_radius=n,
        alpha=alpha,
        beta=beta
    )

class Model(object):
    def __init__(self, num_classes, data_format="channels_first", dtype=tf.float32):
        self.num_classes = num_classes
        self.data_format = data_format
        self.dtype = dtype

    def __call__(self, inputs, training):
        with tf.variable_scope("alexnet_model"):
            inputs = tf.layers.conv2d(
                inputs=inputs,
                kernel_size=[11, 11],
                filters=96,
                strides=4,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer
            )
            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=[3, 3],
                strides=2
            )
            inputs = normalization_layer(inputs)

            inputs = tf.layers.conv2d(
                inputs=inputs,
                kernel_size=[5, 5],
                filters=256,
                strides=1,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.ones_initializer
            )
            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=[3, 3],
                strides=2
            )
            inputs = normalization_layer(inputs)

            inputs = tf.layers.conv2d(
                inputs=inputs,
                kernel_size=[3, 3],
                filters=384,
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer
            )

            inputs = tf.layers.conv2d(
                inputs=inputs,
                kernel_size=[3, 3],
                filters=384,
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.ones_initializer
            )

            inputs = tf.layers.conv2d(
                inputs=inputs,
                kernel_size=[3, 3],
                filters=256,
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.ones_initializer
            )

            inputs = tf.layers.max_pooling2d(
                inputs=inputs,
                pool_size=[3, 3],
                strides=2
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=4096,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.ones_initializer
            )
            inputs = tf.layers.dropout(
                inputs=inputs,
                training=training
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=4096,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.ones_initializer
            )
            inputs = tf.layers.dropout(
                inputs=inputs,
                training=training
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=self.num_classes,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                bias_initializer=tf.zeros_initializer
            )

            return inputs

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
