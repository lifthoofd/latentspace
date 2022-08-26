import tensorflow as tf
from tensorflow import keras


# "hard sigmoid", useful for binary accuracy calculation from logits
def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


# augments images with a probability that is dynamically updated during training
class AdaptiveAugmenter(keras.Model):
    def __init__(self):
        super().__init__()
        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.0)

        # the corresponding augmentation names from the paper are shown above each layer
        # the authors show (see figure 4), that the blitting and geometric augmentations
        # are the most helpful in the low-data regime
        self.augmenter = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(256, 512, 3)),
                # blitting/x-flip:
                keras.layers.RandomFlip("horizontal"),
                # blitting/integer translation:
                keras.layers.RandomTranslation(
                    height_factor=0.125,
                    width_factor=0.125,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                keras.layers.RandomRotation(factor=0.125),
                # geometric/isotropic and anisotropic scaling:
                keras.layers.RandomZoom(
                    height_factor=(-0.25, 0.0), width_factor=(-0.25, 0.0)
                ),
            ],
            name="adaptive_augmenter",
        )

    def call(self, images, training):
        if training:
            augmented_images = self.augmenter(images, training)

            # during training either the original or the augmented images are selected
            # based on self.probability
            augmentation_values = tf.random.uniform(
                shape=(128, 1, 1, 1), minval=0.0, maxval=1.0
            )
            augmentation_bools = tf.math.less(augmentation_values, self.probability)

            images = tf.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(step(real_logits))

        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - 0.85
        self.probability.assign(
            tf.clip_by_value(
                self.probability + accuracy_error / 1000.0, 0.0, 1.0
            )
        )


def make_generator_model(y_dim, z_dim):
    z = keras.layers.Input(shape=(1, 1, z_dim))
    y = keras.layers.Input(shape=(1, 1, y_dim,))

    gen_in = keras.layers.concatenate([z, y], axis=3)

    start_size = (2, 4)

    x = keras.layers.Dense(start_size[0] * start_size[1] * 128, use_bias=False)(gen_in)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Reshape(target_shape=(start_size[0], start_size[1], 128))(x)

    # # 2, 4
    # x = keras.layers.Conv2DTranspose(2048, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    # x = keras.layers.BatchNormalization(scale=False)(x)
    # x = keras.layers.ReLU()(x)

    # 4, 8
    x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)

    # 8, 16
    x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)

    # 16, 32
    x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)

    # 32, 64
    x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)

    # 64, 128
    x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)

    # 128, 256
    x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.ReLU()(x)

    # 256, 512
    x = keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(x)

    return keras.models.Model([z, y], x, name='generator')


def make_discriminator_model(y_dim, weight_init, image_size):
    im = keras.layers.Input(shape=(image_size[0], image_size[1], 3))
    y = keras.layers.Input(shape=(image_size[0], image_size[1], y_dim))

    x = keras.layers.concatenate([im, y], axis=3)

    # 128, 256
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # 64, 128
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # 32, 64
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # 16, 32
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # 8, 16
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # 4, 8
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    # 2, 4
    x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(1)(x)

    return keras.models.Model([im, y], x, name='discriminator')

