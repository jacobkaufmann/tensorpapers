from __future__ import absolute_import, print_function, division

import tensorflow as tf

class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, data_format, name=""):
        super(Conv, self).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            kernel_initializer=tf.glorot_uniform_initializer(),
            bias_initializer=tf.zeros_initializer(),
            activation=tf.nn.relu,
            data_format=data_format,
        )
    def call(self, input_tensor):
        return self.conv(input_tensor)

class MaxPool(tf.keras.layers.Layer):
    def __init__(self, data_format, name=""):
        super(MaxPool, self).__init__(name=name)
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(2,2),
            strides=2,
            padding="valid",
            data_format=data_format
        )
    def call(self, input_tensor):
        return self.maxpool(input_tensor)

class Vgg(tf.keras.Model):
    def __init__(self, num_classes, data_format, layers=16, name=""):
        super(Vgg, self).__init__(name=name)
        self.conv_layers = [
            Conv(64, data_format),
            Conv(64, data_format),
            MaxPool(data_format),
            Conv(128, data_format),
            Conv(128, data_format),
            MaxPool(data_format),
            Conv(256, data_format),
            Conv(256, data_format),
            Conv(256, data_format),
            MaxPool(data_format),
            Conv(512, data_format),
            Conv(512, data_format),
            Conv(512, data_format),
            MaxPool(data_format),
            Conv(512, data_format),
            Conv(512, data_format),
            Conv(512, data_format),
            MaxPool(data_format),
        ]
        self.flatten = tf.keras.layers.Flatten(data_format=data_format)
        self.fc1 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.logits = tf.keras.layers.Dense(units=num_classes, activation=None)
        if layers == 19:
            self.conv_layers.insert(9, Conv(256, data_format=data_format))
            self.conv_layers.insert(14, Conv(512, data_format=data_format))
            self.conv_layers.insert(19, Conv(512, data_format=data_format))

    def call(self, input_tensor, training=False):
        network = self.conv_layers[0](input_tensor)
        for layer in self.conv_layers[1:]:
            network = layer(network)
        network = self.flatten(network)
        network = self.fc1(network)
        network = self.dropout1(network)
        network = self.fc2(network)
        network = self.dropout2(network)
        network = self.logits(network)
        return network

def vgg_model_fn(features, labels, mode, params):
    image = tf.expand_dims(features, -1)
    labels = tf.cast(labels, tf.int32)
    model = Vgg(num_classes=params["num_classes"], data_format=params["data_format"])
    logits = model(image, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=tf.argmax(logits, axis=1)
    )
    metrics = {"accuracy": accuracy}
    tf.summary.scalar("accuracy", accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics
        )

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    print(dataset)
    return dataset.make_one_shot_iterator().get_next()
