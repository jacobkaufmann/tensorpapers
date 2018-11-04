from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import dcgan_model

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def main(unused_argv):
    # Parameters
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    NOISE_DIM = 100
    EPOCHS = 150

    # Prepare MNIST dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    # Prepare DCGAN
    gen = dcgan_model.Generator()
    disc = dcgan_model.Discriminator()
    dcgan = dcgan_model.DCGAN(generator=gen, discriminator=disc)

    num_examples_to_generate = 16
    random_vector = tf.random_normal([num_examples_to_generate, NOISE_DIM])
    noise = tf.random_normal([BATCH_SIZE, NOISE_DIM])

    generated_images = dcgan.generator(noise, training=True)

    iterator = train_dataset.make_initializable_iterator()

    images = iterator.get_next()

    real_output = dcgan.discriminator(images, training=True)
    generated_output = dcgan.discriminator(generated_images, training=True)

    gen_loss = dcgan.generator.loss(generated_output)
    disc_loss = dcgan.discriminator.loss(real_output, generated_output)
    generator_optimizer = tf.train.AdamOptimizer(1e-4).minimize(gen_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(1e-4).minimize(disc_loss)

    train_size = train_images.shape[0]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        for epoch in range(EPOCHS):
            try:
                while True:
                    sess.run([generator_optimizer, discriminator_optimizer])
            except tf.errors.OutOfRangeError:
                pass
            print("Generator Loss:", sess.run(gen_loss))
            print("Discriminator Loss:", sess.run(gen_loss))

if __name__ == "__main__":
    tf.app.run()
