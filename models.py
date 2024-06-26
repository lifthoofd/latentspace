# from tensorflow.keras import layers
# from tensorflow.keras import models
from tensorflow import keras
from math import log, pow


def make_generator_model(y_dim, z_dim, weight_init, image_size, filters=1024):
    z = keras.layers.Input(shape=(1, 1, z_dim))
    y = keras.layers.Input(shape=(1, 1, y_dim,))

    gen_in = keras.layers.concatenate([z, y], axis=3)

    start_size = (2, 4)

    steps = log(image_size[1], 2) - log(4, 2)

    x = keras.layers.Dense(start_size[0] * start_size[1] * filters)(gen_in)
    x = keras.layers.Reshape((start_size[0], start_size[1], filters))(x)
    x = keras.layers.ReLU()(x)

    for i in range(int(steps)):
        dim = filters // pow(2, i)
        x = keras.layers.Conv2DTranspose(dim,
                                         (3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         use_bias=False,
                                         kernel_initializer=weight_init)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', activation='tanh', use_bias=False,
                               kernel_initializer=weight_init)(x)

    return keras.models.Model([z, y], x, name='generator')


def make_discriminator_model(y_dim, weight_init, image_size, filters=1024):
    im = keras.layers.Input(shape=(image_size[0], image_size[1], 3))
    y = keras.layers.Input(shape=(image_size[0], image_size[1], y_dim))

    steps = log(image_size[1], 2) - log(4, 2)

    x = keras.layers.concatenate([im, y], axis=3)

    for i in range(int(steps)):
        inv_i = abs(i - steps) - 1
        dim = filters // pow(2, inv_i)
        x = keras.layers.Conv2D(dim, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                                kernel_initializer=weight_init)(x)
        x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    return keras.models.Model([im, y], x, name='discriminator')

