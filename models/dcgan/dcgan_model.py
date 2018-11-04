from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import time

class Generator(object):
    def __init__(self):
        self.fc1 = tf.layers.Dense(units=7*7*64, use_bias=False)
        self.batchnorm1 = tf.layers.BatchNormalization()

        self.conv1 = tf.layers.Conv2DTranspose(
            filters=64,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same",
            use_bias=False
        )
        self.batchnorm2 = tf.layers.BatchNormalization()

        self.conv2 = tf.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(5,5),
            strides=(2,2),
            padding="same",
            use_bias=False
        )
        self.batchnorm3 = tf.layers.BatchNormalization()

        self.conv3 = tf.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(5,5),
            strides=(2,2),
            padding="same",
            use_bias=False
        )
        

    def __call__(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)
        
        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv3(x))
        return x

    def loss(self, generated_output):
        return tf.losses.sigmoid_cross_entropy(
            tf.ones_like(generated_output),
            generated_output
        )

class Discriminator(object):
    def __init__(self):
        self.conv1 = tf.layers.Conv2D(
            filters=64,
            kernel_size=(5,5),
            strides=(2,2),
            padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(2,2),
            padding="same"
        )
        self.dropout = tf.layers.Dropout(rate=0.3)
        self.flatten = tf.layers.Flatten()
        self.fc1 = tf.layers.Dense(units=1)
    
    def __call__(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def loss(self, real_output, generated_output):
        real_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real_output),
            logits=real_output
        )
        generated_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(generated_output),
            logits=generated_output
        )
        total_loss = real_loss + generated_loss
        return total_loss

class DCGAN(object):
    def __init__(
        self,
        generator,
        discriminator,
        generator_scope="generator",
        discriminator_scope="discriminator"):
        """DCGAN model
        """
        with tf.variable_scope(generator_scope):
            self.generator = generator
        with tf.variable_scope(discriminator_scope):
            self.discriminator = discriminator
           
    def train(self, dataset, epochs, noise_dim, batch_size):
        num_examples_to_generate = 16

        discriminator_optimizer=tf.train.AdamOptimizer(1e-4)
        generator_optimizer=tf.train.AdamOptimizer(1e-4)

        random_vector = tf.random_normal([num_examples_to_generate, noise_dim])

        for epoch in range(epochs):
            start = time.time()
            for images in dataset:
                noise = tf.random_normal([batch_size, noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.generator(noise, training=True)

                    real_output = self.discriminator(images, training=True)
                    generated_output = self.discriminator(generated_images, training=True)

                    gen_loss = self.generator.loss(generated_output)
                    disc_loss = self.discriminator.loss(real_output, generated_output)

            print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                             time.time()-start))
