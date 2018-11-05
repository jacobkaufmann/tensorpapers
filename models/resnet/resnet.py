from __future__ import print_function, division, absolute_import

import tensorflow as tf

class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, filters, strides=1, downsample=None, name=""):
        super(BasicBlock, self).__init__(name=name)
        self.downsample = downsample
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        residual = input_tensor
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = tf.keras.layers.BatchNormalization(residual, training=training)
        net = self.conv1(input_tensor)
        net = self.bn1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv2(net)
        net = self.bn2(net, training=training)
        net = tf.add(net, residual)
        return tf.nn.relu(net)

class BottleneckBlock(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, strides=1, downsample=None, name=""):
        super(BottleneckBlock, self).__init__(name=name)
        self.downsample = downsample
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1,1),
            strides=strides
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3,3),
            padding="same"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters*4, kernel_size=(1,1))
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        residual = input_tensor
        if self.downsample is not None:
            residual = self.downsample
            residual = tf.keras.layers.BatchNormalization(residual, training=training)
        net = self.conv1(input_tensor)
        net = self.bn1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv2(net)
        net = self.bn2(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv3(net)
        net = self.bn3(net, training=training)
        net = tf.add(net, residual)
        return tf.nn.relu(net)

class BlockLayer(tf.keras.Model):
    def __init__(self,
                block,
                blocks,
                filters,
                strides=1,
                name=""):
        super(BlockLayer, self).__init__(name=name)
        filters_out = filters * block.expansion
        downsample = tf.keras.layers.Conv2D(
            filters=filters_out,
            kernel_size=(1,1),
            strides=strides
        )
        self.layer = []
        self.layer.append(block(filters, strides, downsample))
        for _ in range(1, blocks):
            self.layer.append(block(filters, strides))

    def call(self, input_tensor, training=False):
        net = None
        net = self.layer[0](input_tensor, training)
        for i in range(1, len(self.layer)):
            net = self.layer[i](net, training)
        return net

class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes, name=""):
        super(ResNet, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(7,7),
            strides=2,
            padding="valid"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=2,
            padding="valid"
        )
        self.layer1 = BlockLayer(block, layers[0], 64)
        self.layer2 = BlockLayer(block, layers[0], 128)
        self.layer3 = BlockLayer(block, layers[0], 256)
        self.layer4 = BlockLayer(block, layers[0], 512)
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(7,7)
        )
        self.fc = tf.keras.layers.Dense(num_classes)
    
    def call(self, input_tensor, training=False):
        net = self.conv1(input_tensor)
        net = self.bn1(net, training=training)
        net = self.max_pool(net)
        net = self.layer1(net, training=training)
        net = self.layer2(net, training=training)
        net = self.layer3(net, training=training)
        net = self.layer4(net, training=training)
        net = self.avg_pool(net)
        net = self.fc(net)
        return net
