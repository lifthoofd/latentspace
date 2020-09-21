from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from math import log

ASPECT_16_9 = 0
ASPECT_1_1 = 1


def make_generator_model(y_dim, z_dim, weight_init, bn_momentum, image_size, aspect_ratio, filters=1024):
    z = layers.Input(shape=(1, 1, z_dim))
    y = layers.Input(shape=(1, 1, y_dim,))

    gen_in = layers.concatenate([z, y], axis=3)

    if aspect_ratio == ASPECT_16_9:
        steps_y = int(log(image_size[0], 2)) - int(log(9, 2))
        steps_x = int(log(image_size[1], 2)) - int(log(8, 2))

        x = layers.Dense(9*8*filters, use_bias=False, kernel_initializer=weight_init)(gen_in)

        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.ReLU()(x)

        x = layers.Reshape((9, 8, filters))(x)
    else:
        steps_y = int(log(image_size[0], 2)) - int(log(8, 2))
        steps_x = int(log(image_size[1], 2)) - int(log(8, 2))

        x = layers.Dense(8 * 8 * filters, use_bias=False, kernel_initializer=weight_init)(gen_in)

        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.ReLU()(x)

        x = layers.Reshape((8, 8, filters))(x)

    layer_amt = steps_x if steps_x >= steps_y else steps_y
    curr_filters = filters
    strides = [1, 1]
    for i in range(layer_amt + 1):
        if i == 0:
            x = layers.Conv2DTranspose(curr_filters, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
            x = layers.ReLU()(x)
        elif i == layer_amt:
            x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init, activation='tanh')(x)
        else:
            curr_filters = curr_filters // 2
            strides[0] = 2 if abs((i - 1) - layer_amt) <= steps_y else 1
            strides[1] = 2 if abs((i - 1) - layer_amt) <= steps_x else 1

            x = layers.Conv2DTranspose(curr_filters, (5, 5), strides=(strides[0], strides[1]), padding='same', use_bias=False, kernel_initializer=weight_init)(x)
            x = layers.BatchNormalization(momentum=bn_momentum)(x)
            x = layers.ReLU()(x)

    return models.Model([z, y], x, name='generator')


def make_discriminator_model(y_dim, weight_init, img_size, lr_slope):
    im = layers.Input(shape=(img_size[0], img_size[1], 3))
    y = layers.Input(shape=(img_size[0], img_size[1], y_dim))

    disc_in = layers.concatenate([im, y], axis=3)

    x = layers.GaussianNoise(stddev=0.05)(disc_in)

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = layers.LeakyReLU(alpha=lr_slope)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = layers.LeakyReLU(alpha=lr_slope)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = layers.LeakyReLU(alpha=lr_slope)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, kernel_initializer=weight_init)(x)

    return models.Model([im, y], x, name='discriminator')