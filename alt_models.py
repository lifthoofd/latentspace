from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from math import log, pow


def make_generator_model(y_dim, z_dim, weight_init, bn_momentum, image_size, aspect_ratio, filters=64):
    z = layers.Input(shape=(1, 1, z_dim))
    y = layers.Input(shape=(1, 1, y_dim,))

    gen_in = layers.concatenate([z, y], axis=3)

    start_size = (2, 1)

    x = layers.Dense(start_size[0] * start_size[1] * 2048)(gen_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((start_size[0], start_size[1], 2048))(x)

    # 2, 1
    x = layers.Conv2DTranspose(1024,
                               (5, 5),
                               strides=(1, 1),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 4, 2
    x = layers.Conv2DTranspose(512,
                               (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 8, 4
    x = layers.Conv2DTranspose(256,
                               (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 16, 8
    x = layers.Conv2DTranspose(128,
                               (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 32, 16
    x = layers.Conv2DTranspose(64,
                               (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 64, 32
    x = layers.Conv2DTranspose(32,
                               (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 128, 64
    x = layers.Conv2DTranspose(16,
                               (5, 5),
                               strides=(2, 2),
                               padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 256, 128
    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh', use_bias=False)(x)

    return models.Model([z, y], x, name='generator')


def make_discriminator_model(y_dim, weight_init, image_size, lr_slope, aspect_ratio, filters=64):
    im = layers.Input(shape=(image_size[0], image_size[1], 3))
    y = layers.Input(shape=(image_size[0], image_size[1], y_dim))

    x = layers.concatenate([im, y], axis=3)

    # 128, 64
    x = layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)

    # 64, 32
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)
    
    # 32, 16
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)

    # 16, 8
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)

    # 8, 4
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)

    # 4, 2
    x = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)

    # 2, 1
    x = layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return models.Model([im, y], x, name='discriminator')


