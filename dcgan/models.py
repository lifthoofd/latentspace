from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from math import log, pow

ASPECT_16_9 = 0
ASPECT_1_1 = 1
ASPECT_16_10 = 2


def make_generator_model(y_dim, z_dim, weight_init, bn_momentum, image_size, aspect_ratio, filters=64):
    z = layers.Input(shape=(1, 1, z_dim))
    y = layers.Input(shape=(1, 1, y_dim,))

    gen_in = layers.concatenate([z, y], axis=3)

    if aspect_ratio == ASPECT_16_9:
        start_size = (9, 16)
        steps = int(log(image_size[1], 2)) - int(log(16, 2)) + 1
        dim_mul = 16
    elif aspect_ratio == ASPECT_16_10:
        start_size = (5, 8)
        steps = int(log(image_size[1], 2)) - int(log(8, 2))
        dim_mul = 32
    else:
        start_size = (8, 8)
        steps = int(log(image_size[1], 2)) - int(log(8, 2))
        dim_mul = 32

    dim = filters

    x = layers.Dense(start_size[0] * start_size[1] * (dim * dim_mul))(gen_in)
    x = layers.Reshape((start_size[0], start_size[1], (dim * dim_mul)))(x)
    x = layers.ReLU()(x)

    for i in range(steps):
        dim_mul //= 2
        if i == steps - 1:
            x = layers.Conv2DTranspose(dim * dim_mul, (4, 4), strides=(1, 1), padding='same')(x)
        else:
            x = layers.Conv2DTranspose(dim * dim_mul, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, (4, 4), strides=(1, 1), padding='same', activation='tanh')(x)

    return models.Model([z, y], x, name='generator')


def make_discriminator_model(y_dim, weight_init, image_size, lr_slope, aspect_ratio, filters=64):
    im = layers.Input(shape=(image_size[0], image_size[1], 3))
    y = layers.Input(shape=(image_size[0], image_size[1], y_dim))

    x = layers.concatenate([im, y], axis=3)

    if aspect_ratio == ASPECT_16_9:
        steps = (int(log(image_size[1], 2)) - int(log(16, 2))) + 2
    elif aspect_ratio == ASPECT_16_10:
        steps = int(log(image_size[1], 2)) - int(log(8, 2))
    else:
        steps = int(log(image_size[1], 2)) - int(log(8, 2))

    dim = filters

    for i in range(steps):
        dim_mul = int(pow(2, i))
        x = layers.Conv2D(dim * dim_mul, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=lr_slope)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return models.Model([im, y], x, name='discriminator')
