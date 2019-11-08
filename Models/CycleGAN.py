from tensorflow.python.keras.layers import (
    Conv2D, LeakyReLU, MaxPooling2D, BatchNormalization, Input, Activation,
    Add, Conv2DTranspose, Dropout, UpSampling2D, Concatenate)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import tensorflow as tf


class Generator(Model):
    def __init__(self, channels, norm_type='bn'):
        Model.__init__(self)
        self.channels = channels
        self.norm_type = norm_type

    def build(self):
        inputs = Input(shape=[None, None, 3])
        conv1 = Conv2D(64, 4, 2, activation='relu', padding='same')(inputs)
        conv2 = self.conv_block(conv1, 128)
        conv3 = self.conv_block(conv2, 256)
        conv4 = self.conv_block(conv3, 512)
        conv5 = self.conv_block(conv4, 512)

        upconv1 = self.upconv_block(conv5, 512)
        upconv1 = Concatenate(axis=-1)([upconv1, conv4])
        upconv2 = self.upconv_block(upconv1, 256)
        upconv2 = Concatenate(axis=-1)([upconv2, conv3])
        upconv3 = self.upconv_block(upconv2, 128)
        upconv3 = Concatenate(axis=-1)([upconv3, conv2])
        upconv4 = self.upconv_block(upconv3, 64)
        upconv4 = Concatenate(axis=-1)([upconv4, conv1])

        outputs = Conv2DTranspose(
            self.channels, 4, 2, padding='same', activation='sigmoid')(upconv4)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def conv_block(self, inputs, filters):
        x = Conv2D(filters, 4, 2, padding='same')(inputs)
        x = self.norm(x)
        x = Activation('relu')(x)
        return x

    def upconv_block(self, inputs, filters):
        x = Conv2DTranspose(filters, 4, 2, padding='same')(inputs)
        x = self.norm(x)
        x = Activation('relu')(x)
        return x

    def norm(self, inputs):
        if self.norm_type == 'bn':
            return BatchNormalization()(inputs)
        elif self.norm_type == 'in':
            return InstanceNormalization()(inputs)


class Discriminator(Model):
    def __init__(self, classes, norm_type='bn'):
        Model.__init__(self)
        self.classes = classes
        self.norm_type = norm_type

    def build(self):
        inputs = Input(shape=[None, None, 3])
        conv1 = Conv2D(64, 4, 2, activation=LeakyReLU(0.2), padding='same')(inputs)
        conv2 = Conv2D(128, 4, 2, padding='same')(conv1)
        conv2 = self.norm(conv2)
        conv2 = LeakyReLU(0.2)(conv2)
        conv3 = Conv2D(256, 4, 2, padding='same')(conv2)
        conv3 = self.norm(conv3)
        conv3 = LeakyReLU(0.2)(conv3)
        conv4 = Conv2D(512, 4, 2, padding='same')(conv3)
        conv4 = self.norm(conv4)
        conv4 = LeakyReLU(0.2)(conv4)
        conv5 = Conv2D(1, 4, 2, padding='same')(conv4)

        upconv1 = Conv2DTranspose(256, 4, 2, padding='same')(conv5)
        upconv1 = self.norm(upconv1)
        upconv1 = Activation('relu')(upconv1)
        concat1 = Concatenate()([conv4, upconv1])
        upconv2 = Conv2DTranspose(128, 4, 2, padding='same')(concat1)
        upconv2 = self.norm(upconv2)
        upconv2 = Activation('relu')(upconv2)
        concat2 = Concatenate()([conv3, upconv2])
        upconv3 = Conv2DTranspose(64, 4, 2, padding='same')(concat2)
        upconv3 = self.norm(upconv3)
        upconv3 = Activation('relu')(upconv3)
        concat3 = Concatenate()([conv2, upconv3])
        upconv4 = Conv2DTranspose(32, 4, 2, padding='same')(concat3)
        upconv4 = self.norm(upconv4)
        upconv4 = Activation('relu')(upconv4)
        concat4 = Concatenate()([conv1, upconv4])
        outputs = Conv2DTranspose(self.classes, 4, 2, padding='same', activation='softmax')(concat4)

        model = Model(inputs=inputs, outputs=[conv5, outputs])

        return model

    def norm(self, inputs):
        if self.norm_type == 'bn':
            return BatchNormalization()(inputs)
        elif self.norm_type == 'in':
            return InstanceNormalization()(inputs)


class InstanceNormalization(Layer):
    ''' adapted from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib '''

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[3], ),
            initializer="one",
            trainable=True)
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[3], ),
            initializer="zero",
            trainable=True)
        Layer.build(self, input_shape)

    def call(self, x, mask=None):
        def image_expand(tensor):
            return K.expand_dims(K.expand_dims(tensor, -1), -1)

        def batch_image_expand(tensor):
            return image_expand(K.expand_dims(tensor, 0))

        input_shape = K.int_shape(x)
        reduction_axes = list(range(0, len(input_shape)))
        del reduction_axes[3]
        del reduction_axes[0]
        mean = K.mean(x, reduction_axes, keepdims=True)
        stddev = K.std(x, reduction_axes, keepdims=True) + 1e-3
        normed = (x - mean) / stddev
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[3] = input_shape[3]
        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        normed = normed * broadcast_gamma
        broadcast_beta = K.reshape(self.beta, broadcast_shape)
        normed = normed + broadcast_beta
        return normed


# class Discriminator(Model):
#     def __init__(self):
#         Model.__init__(self)
#         self.conv1 = Conv2D(
#             64, 4, 2, activation=LeakyReLU(0.2), padding='same')
#         self.conv2 = Conv2D(128, 4, 2, padding='same')
#         self.conv3 = Conv2D(256, 4, 2, padding='same')
#         self.conv4 = Conv2D(512, 4, 1, padding='same')
#         self.last = Conv2D(1, 4, 1, padding='same')
#         self.norm = []
#         for i in range(3):
#             self.norm.append(BatchNormalization())

#     def call(self, inputs, training=True):
#         conv1 = self.conv1(inputs)
#         conv2 = self.conv2(conv1)
#         if training:
#             conv2 = self.norm[0](conv2)
#         # conv2 = instance_norm(conv2)
#         conv2 = LeakyReLU(0.2)(conv2)

#         conv3 = self.conv3(conv2)
#         if training:
#             conv3 = self.norm[1](conv3)
#         # conv3 = instance_norm(conv3)
#         conv3 = LeakyReLU(0.2)(conv3)

#         conv4 = self.conv4(conv3)
#         if training:
#             conv4 = self.norm[2](conv4)
#         # conv4 = instance_norm(conv4)
#         conv4 = LeakyReLU(0.2)(conv4)

#         last = self.last(conv4)

#         return last
