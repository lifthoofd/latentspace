from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from math import log, pow


def make_generator_model(y_dim, z_dim, weight_init, bn_momentum, image_size, aspect_ratio, filters=64):
    z = layers.Input(shape=(1, 1, z_dim))
    y = layers.Input(shape=(1, 1, y_dim,))

    gen_in = layers.concatenate([z, y], axis=3)

    start_size = (2, 4)

    x = layers.Dense(start_size[0] * start_size[1] * 1024)(gen_in)
    x = layers.Reshape((start_size[0], start_size[1], 1024))(x)
    x = layers.ReLU()(x)

    # 4, 8
    x = layers.Con2DTranspose(1024,
                              (4, 4),
                              strides=(2, 2),
                              padding='same',
                              use_bias=False,
                              kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 8, 16
    x = layers.Con2DTranspose(512,
                              (4, 4),
                              strides=(2, 2),
                              padding='same',
                              use_bias=False,
                              kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 16, 32
    x = layers.Con2DTranspose(256,
                              (4, 4),
                              strides=(2, 2),
                              padding='same',
                              use_bias=False,
                              kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 32, 64
    x = layers.Con2DTranspose(128,
                              (4, 4),
                              strides=(2, 2),
                              padding='same',
                              use_bias=False,
                              kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 64, 128
    x = layers.Con2DTranspose(64,
                              (4, 4),
                              strides=(2, 2),
                              padding='same',
                              use_bias=False,
                              kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding='same', activation='tanh', use_bias=False, kernel_initializer=weight_init)(x)

    return models.Model([z, y], x, name='generator')


def make_discriminator_model(y_dim, weight_init, image_size, lr_slope, aspect_ratio, filters=64):
    im = layers.Input(shape=(image_size[0], image_size[1], 3))
    y = layers.Input(shape=(image_size[0], image_size[1], y_dim))

    x = layers.concatenate([im, y], axis=3)

    # 32, 64
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
    # x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 16, 32
    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
    # x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 8, 16
    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
    # x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 4, 8
    x = layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
    # x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 2, 4
    x = layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
    # x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return models.Model([im, y], x, name='discriminator')
