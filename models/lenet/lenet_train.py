from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_dir", "/tmp/imagenet_train",
                           "Where to save model checkpoints")

# Hyperparameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("max_steps", 10000000,
                            "Number of batches to run.")
tf.app.flags.DEFINE_float("initial_learning_rate", 0.1,
                          "Initial learning rate.")
tf.app.flags.DEFINE_integer("num_epochs_per_decay", 30,
                          "Epochs after which learning rate decays.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.16,
                          "Learning rate decay factor.")


# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer("num_gpus", 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean("num_cpu_cores", 4,
                            """Whether to log device placement.""")

IMAGE_SIZE = 32
NUM_CHANNELS = 1

def train_input_fn():
    pass

def loss(model, x, y):
    y_pred = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pred)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_op(dataset):
    train_input = tf.placeholder(
        tf.float32,
        shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    train_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, ])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.train.get_or_create_global_step()

    with tf.Session() as sess:
        tf.global_variables_initializer()