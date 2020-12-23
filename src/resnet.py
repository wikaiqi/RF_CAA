import numpy as np 
import tensorflow as tf 
from tensorflow.keras import models, layers, Input, Model

# A simple version of resNet


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, kernal_num, stride=1, l2=0.005):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(kernal_num, kernal_num),
                                            strides=stride,
                                            kernel_regularizer=tf.keras.regularizers.l2(l2),
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(kernal_num, kernal_num),
                                            strides=1,
                                            kernel_regularizer=tf.keras.regularizers.l2(l2),
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def Resnet_layer(filter_num, kernal_num,  blocks, stride=1, l2=0.005):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, kernal_num, stride=stride, l2=l2))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, kernal_num, stride=1, l2=l2))

    return res_block


class Encoder(Model):
    def __init__(self, l2=0.01):
        super(Encoder, self).__init__()
        
        self.layer1 = Resnet_layer(8, 12,  2, stride=1, l2=l2)
        self.layer2 = Resnet_layer(16, 6,  2, stride=2, l2=l2)
        self.layer3 = Resnet_layer(32, 6,  2, stride=2, l2=l2)
        self.layer4 = Resnet_layer(32, 3,  2, stride=2, l2=l2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

        initializer = tf.keras.initializers.HeNormal()

        self.Flatten = layers.Flatten()

    def call(self, inputs, training = None, **kwargs):
        x = self.layer1(inputs, training=training)
        x = self.layer2(x, training = training)
        x = self.layer3(x, training = training)
        x = self.layer4(x, training = training)
        x = self.avgpool(x)
        x = self.Flatten(x)
        return x


class Predictor(Model):
    def __init__(self, l2=0.01):
        super(Predictor, self).__init__()
        initializer = tf.keras.initializers.HeNormal()

        self.Dense1 = layers.Dense(32, activation='relu', 
                                    kernel_regularizer=tf.keras.regularizers.l2(l2),
                                    kernel_initializer=initializer)
        self.dropout = layers.Dropout(0.3)
        self.Dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training = None, **kwargs):
        x = self.Dense1(inputs)
        x = self.dropout(x)
        x = self.Dense2(x)
        return x


class Discriminator(Model):
    def __init__(self, l2=0.01):
        super(Discriminator, self).__init__()
        initializer = tf.keras.initializers.HeNormal()

        self.Dense1 = layers.Dense(32, activation='relu', 
                                    kernel_regularizer=tf.keras.regularizers.l2(l2),
                                    kernel_initializer=initializer)
        self.dropout = layers.Dropout(0.3)
        self.Dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training = None, **kwargs):
        x = self.Dense1(inputs)
        x = self.dropout(x)
        x = self.Dense2(x)
        return x

