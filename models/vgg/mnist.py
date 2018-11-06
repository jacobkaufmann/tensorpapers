from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

import vgg

TRAIN_BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 10
VALIDATION_SIZE = 5000
EVAL_BATCH_SIZE = 128

def main():
    """Train vgg16 on mnist

    Note: In order to have compatible dimensions, you must
    comment out the first three pooling layers in Vgg
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_eval = x_train[:VALIDATION_SIZE, ...]
    y_eval = y_train[:VALIDATION_SIZE]
    x_train = x_train[VALIDATION_SIZE:, ...]
    y_train = y_train[VALIDATION_SIZE:]

    model = tf.estimator.Estimator(
        model_fn=vgg.vgg_model_fn,
        params={
            "num_classes": NUM_CLASSES,
            "data_format": "channels_last"
        }
    )

    model.train(
        input_fn=lambda:vgg.input_fn(x_train, y_train, TRAIN_BATCH_SIZE)
    )
if __name__=="__main__":
    main()
