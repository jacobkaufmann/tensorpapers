from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from lenet import lenet_model

BATCH_SIZE = 32
IMAGE_SIZE = 32
NUM_CHANNELS = 1
NUM_CLASSES = 10
NUM_EPOCHS = 10
VALIDATION_SIZE = 5000
EVAL_BATCH_SIZE = 64

def _pad_input_with_zeros(inputs):
    # Pad input images to IMAGE_SIZE x IMAGE_SIZE

    input_height = inputs.shape[1]
    input_width = inputs.shape[2]
    assert input_height == input_width
    assert input_height < IMAGE_SIZE
    assert (IMAGE_SIZE - input_height) % 2 == 0

    padded = np.zeros(shape=(inputs.shape[0], IMAGE_SIZE, IMAGE_SIZE))
    pad_top_bottom = (IMAGE_SIZE - input_height) // 2
    pad_left_right = (IMAGE_SIZE - input_width) // 2

    for i in range(len(inputs)):
        padded[i] = np.pad(inputs[i], ((pad_top_bottom, pad_top_bottom),
            (pad_left_right, pad_left_right)), "constant")
    return padded

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(60000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def main(unused_argv):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = _pad_input_with_zeros(x_train)
    x_test = _pad_input_with_zeros(x_test)

    x_eval = x_train[:VALIDATION_SIZE, ...]
    y_eval = y_train[:VALIDATION_SIZE]
    x_train = x_train[VALIDATION_SIZE:, ...]
    y_train = y_train[VALIDATION_SIZE:]

    train_input = tf.placeholder(
        tf.float32,
        shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, ])

    eval_input = tf.placeholder(
        tf.float32,
        shape=[EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    
    build_lenet = lenet_model.Lenet(NUM_CLASSES)
    logits = build_lenet(train_input, True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels, logits=logits, name="loss"))
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.01, momentum=0.9).minimize(loss)
    
    train_prediction = tf.nn.softmax(logits)
    train_size = x_train.shape[0]

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
        # Run all initializers
        tf.global_variables_initializer().run()
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = x_train[offset:(offset + BATCH_SIZE), ...]
            batch_labels = y_train[offset:(offset + BATCH_SIZE)]
            
            feed_dict = {train_input: np.expand_dims(batch_data, -1),
                         train_labels: batch_labels}
            sess.run(optimizer, feed_dict=feed_dict)
            if step % 1000 == 0:
                print(sess.run(loss, feed_dict=feed_dict))

if __name__ == "__main__":
    tf.app.run()
