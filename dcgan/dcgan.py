import os
import time
from random import randint
import gc
import tensorflow as tf
import numpy as np
import datetime

from dcgan.dataset import DatasetPipeline
import dcgan.models as models


class DCGAN:
    def __init__(self, config):
        self.num_epochs = int(config['num_epochs'])
        self.batch_size = int(config['batch_size'])
        self.z_dim = int(config['z_dim'])
        self.learning_rate_gen = float(config['learning_rate_gen'])
        self.learning_rate_disc = float(config['learning_rate_disc'])
        self.bn_momentum = float(config['batch_norm_momentum'])
        self.lr_slope = float(config['lrelu_slope'])
        self.log_freq = int(config['log_freq'])
        self.checkpoint_freq = int(config['checkpoint_freq'])
        self.num_images_in_row = int(config['num_images_in_row'])
        self.dataset_path = str(config['dataset_path'])
        self.dataset_name = str(config['dataset_name'])
        self.image_size_x = int(config['image_size_x'])
        self.image_size_y = int(config['image_size_y'])
        self.image_size = (self.image_size_y, self.image_size_x)
        self.aspect = int(config['aspect'])
        self.filters = int(config['filters'])
        self.ckpt_path = str(config['ckpt_path'])
        self.samples_path = str(config['samples_path'])
        self.images_path = os.path.join(str(config['images_path']), 'train')
        if not os.path.isdir(self.images_path):
            os.mkdir(self.images_path)
        self.is_training = False
        self.losses = {}
        self.progress = {}

        # load dataset
        self.dataset_pipeline = DatasetPipeline(self.dataset_path,
                                                self.image_size,
                                                self.dataset_name,
                                                self.batch_size)
        self.dataset = self.dataset_pipeline.load_dataset()
        self.num_labels = self.dataset_pipeline.get_num_labels()

        # make weight init
        self.weight_init = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0)
        # make cross entropy function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # make generator
        self.generator = models.make_generator_model(self.num_labels, self.z_dim, self.weight_init, self.bn_momentum, self.image_size, self.aspect, self.filters)
        # make discriminator
        self.discriminator = models.make_discriminator_model(self.num_labels, self.weight_init, self.image_size, self.lr_slope)

        # print summaries
        self.generator.summary()
        self.discriminator.summary()

        # make optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_gen, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_disc, beta_1=0.5)

        # make checkpoint manager
        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0),
                                              step=tf.Variable(0),
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_path, max_to_keep=10)

        # restore checkpoint if present
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('restored latest checkpoint {}'.format(self.checkpoint_manager.latest_checkpoint))

    # def set_config(self, config):
    #     self.num_epochs = config['num_epochs']
    #     self.batch_size = config['batch_size']
    #     self.z_dim = config['z_dim']
    #     self.learning_rate_gen = config['learning_rate_gen']
    #     self.learning_rate_disc = config['learning_rate_disc']
    #     self.bn_momentum = config['batch_norm_momentum']
    #     self.lr_slope = config['lrelu_slope']
    #     self.log_freq = config['log_freq']
    #     self.checkpoint_freq = config['checkpoint_freq']
    #     self.num_images_in_row = config['num_images_in_row']
    #     self.dataset_path = config['dataset_path']
    #     self.dataset_name = config['dataset_name']
    #     self.image_size = config['image_size']
    #     self.ckpt_path = config['ckpt_path']
    #     self.samples_path = config['samples_path']

    @staticmethod
    def smooth_positive_labels(y):
        return y - 0.3 + (np.random.random(y.shape) * 0.5)

    @staticmethod
    def smooth_negative_labels(y):
        return y + np.random.random(y.shape) * 0.3

    @staticmethod
    def noisy_labels(y, p_flip):
        # determine number of labels to flip
        n_select = int(p_flip * int(y.shape[0]))
        # choose labels to flip
        flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)

        op_list = []
        # invert the labels in place
        for i in range(int(y.shape[0])):
            if i in flip_ix:
                op_list.append(tf.subtract(1.0, y[i]))
            else:
                op_list.append(y[i])

        outputs = tf.stack(op_list)
        return outputs

    def discriminator_loss(self, real_output, fake_output):
        real_output_noise = self.noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = self.noisy_labels(tf.zeros_like(fake_output), 0.05)

        real_output_smooth = self.smooth_positive_labels(real_output_noise)
        fake_output_smooth = self.smooth_negative_labels(fake_output_noise)

        real_loss = self.cross_entropy(tf.ones_like(real_output_smooth), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output_smooth), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def generator_loss(self, fake_output):
        fake_output_smooth = self.smooth_negative_labels(tf.ones_like(fake_output))

        return self.cross_entropy(tf.ones_like(fake_output_smooth), fake_output)

    @staticmethod
    def one_hot(labels, num_labels):
        one_hot_labels = np.eye(num_labels, dtype=np.float32)[labels]
        one_hot_labels = np.reshape(one_hot_labels, [-1, 1, 1, num_labels])
        return one_hot_labels

    def expand_labels(self, labels, num_labels):
        one_hot_labels = self.one_hot(labels, num_labels)
        M = one_hot_labels.shape[0]
        img_size = self.image_size
        expanded_labels = one_hot_labels * np.ones([M, img_size[0], img_size[1], num_labels], dtype=np.float32)
        return one_hot_labels, expanded_labels

    @tf.function
    def train_step(self, gen_z, gen_y, gen_y_expanded, disc_x, disc_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator([gen_z, gen_y], training=True)

            real_output = self.discriminator([disc_x, disc_y], training=True)
            fake_output = self.discriminator([generated_images, gen_y_expanded], training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        epoch_offset = self.checkpoint.epoch.numpy()
        seed = self.generate_z(self.num_labels)
        self.is_training = True

        for epoch in range(epoch_offset, self.num_epochs):
            # print(self.is_training)
            if self.is_training:
                start = time.time()

                for batch in self.dataset:
                    step = self.checkpoint.step.numpy()
                    images, labels = batch
                    _, labels_expanded = self.expand_labels(labels, self.num_labels)
                    gen_z = tf.random.normal([self.batch_size, 1, 1, self.z_dim])
                    labels = [randint(0, self.num_labels - 1) for i in range(self.batch_size)]
                    gen_y, gen_y_expanded = self.expand_labels(labels, self.num_labels)

                    gen_loss, disc_loss = self.train_step(gen_z, gen_y, gen_y_expanded, images, labels_expanded)

                    self.losses['gen'] = float(gen_loss.numpy())
                    self.losses['disc'] = float(disc_loss.numpy())
                    if step % self.log_freq == 0:
                        # print(self.is_training)
                        print('epoch {:04d} | step {:08d} | generator loss: {} | discriminator loss {}'.format(epoch, step,
                                                                                                               gen_loss,
                                                                                                               disc_loss))
                    self.progress['epoch'] = int(epoch)
                    self.progress['step'] = int(step)
                    self.checkpoint.step.assign_add(1)

                    # if not self.is_training:
                    #     break

                # save some images
                self.generate_and_save_images(self.generator, epoch, self.num_labels, seed)

                if (epoch + 1) % self.checkpoint_freq == 0:
                    ckpt_save_path = self.checkpoint_manager.save()

                gc.collect()

                self.checkpoint.epoch.assign(epoch)

                print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

        self.generate_and_save_images(self.generator, self.num_epochs, self.num_labels, seed)
        self.is_training = False

    def stop_train(self):
        self.is_training = False

    @tf.function
    def generate_samples(self, model, z, y):
        return model([z, y], training=False)

    def generate_image(self, z, y):
        image = self.generate_samples(self.generator, z, y)
        # image = image * 127.5 + 127.5
        image = np.array(image)
        return image

    def generate_z(self, num_labels):
        zs = []
        for i in range(num_labels):
            z = tf.random.normal([self.num_images_in_row, 1, 1, self.z_dim])
            # print(z.dtype)
            zs.append(z)

        return zs

    def get_losses(self):
        return self.losses

    def get_progress(self):
        return self.progress

    def generate_and_save_images(self, model, epoch, num_labels, zs):
        h, w = self.image_size
        n_rows = num_labels
        n_cols = self.num_images_in_row
        padding = 0

        shape = (h * n_rows + padding * (n_rows - 1), w * n_cols + padding * (n_cols - 1), 3)
        img = np.full(shape, 0, dtype=np.float32)

        for j in range(num_labels):
            labels = [j] * n_cols
            one_hot_labels = self.one_hot(labels, num_labels)
            z = zs[j]

            images = self.generate_samples(model, z, one_hot_labels)
            images = images * 127.5 + 127.5
            images = np.array(images)

            for i, image in enumerate(images):
                img[j * (h + padding): j * (h + padding) + h, i * (w + padding): i * (w + padding) + w, ...] = image

                im_path = os.path.join(self.images_path, str(i))
                if not os.path.isdir(im_path):
                    os.mkdir(im_path)
                now = datetime.datetime.now(datetime.timezone.utc)
                fn = now.strftime('%Y%m%d_%H%M%S') + '.png'
                tf.io.write_file(os.path.join(im_path, fn), tf.image.encode_png(tf.cast(image, tf.uint8)))
            # time.sleep(1)

        file_name = 'epoch_{:04d}.png'.format(epoch)
        output_dir = os.path.join(self.samples_path, file_name)
        tf.io.write_file(output_dir, tf.image.encode_png(tf.cast(img, tf.uint8)))
