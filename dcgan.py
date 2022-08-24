import os
import sys
import time
import json
from random import randint
# import gc
import tensorflow as tf
import numpy as np
import datetime
from functools import partial
import subprocess
import argparse

from dataset import DatasetPipeline
# import dcgan.models as models
import alt_models as models


SUMMARY_FREQ = 4


class DCGAN:
    def __init__(self, config):
        self.num_epochs = int(config['num_epochs'])
        self.batch_size = 32
        self.z_dim = 100
        self.learning_rate_gen = float(config['learning_rate_gen'])
        self.learning_rate_disc = float(config['learning_rate_disc'])
        self.bn_momentum = 0.8
        self.lr_slope = 0.2
        self.log_freq = 1
        self.checkpoint_freq = 5
        self.num_images_in_row = 10
        self.dataset_path = str(config['dataset_path'])
        self.dataset_name = str(config['dataset_name'])
        self.image_size_x = int(config['image_size_x'])
        self.image_size_y = int(config['image_size_y'])
        self.image_size = (self.image_size_y, self.image_size_x)
        self.aspect = 0
        self.filters = 64
        self.ckpt_path = str(config['ckpt_path'])
        self.samples_path = str(config['samples_path'])
        self.images_path = os.path.join(str(config['images_path']), 'train')
        self.summary_path = str(config['summary_path'])
        self.n_critics = int(config['n_critics'])
        self.gp_mult = float(config['gp_mult'])
        if not os.path.isdir(self.images_path):
            os.makedirs(self.images_path)
        self.is_training = False
        self.losses = {}
        self.progress = {}

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


        # load dataset
        self.dataset_pipeline = DatasetPipeline(self.dataset_path,
                                                self.image_size,
                                                self.dataset_name,
                                                self.batch_size)
        self.dataset = self.dataset_pipeline.load_dataset()
        self.num_labels = self.dataset_pipeline.get_num_labels()
        label_strings = self.dataset_pipeline.get_label_strings()
        np.savetxt(os.path.join(config['project_path'], 'labels.txt'), label_strings, fmt='%s')

        # make weight init
        self.weight_init = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0)
        # make cross entropy function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # make generator
        self.generator = models.make_generator_model(self.num_labels, self.z_dim, self.weight_init, self.bn_momentum, self.image_size, self.aspect, self.filters)
        # make discriminator
        self.discriminator = models.make_discriminator_model(self.num_labels, self.weight_init, self.image_size, self.lr_slope, self.aspect, self.filters)

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

        self.summary_writer = tf.summary.create_file_writer(self.summary_path)

        # restore checkpoint if present
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('restored latest checkpoint {}'.format(self.checkpoint_manager.latest_checkpoint))

    def get_label_strings(self):
        return self.dataset_pipeline.get_label_strings()

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

    @staticmethod
    def discriminator_loss(real_output, fake_output):
        fake_loss = tf.reduce_mean(fake_output)
        real_loss = tf.reduce_mean(real_output)
        return fake_loss - real_loss

    @staticmethod
    def generator_loss(fake_output):
        fake_loss = -tf.reduce_mean(fake_output)
        return fake_loss

    def gradient_penalty(self, f, real, fake, labels):
        alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)

        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f([inter, labels])

        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)

        return gp

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

    def train_step(self, batch, step):
        imgs, labels = batch
        y_real = self.one_hot(labels, self.num_labels)
        _, y_real_expanded = self.expand_labels(labels, self.num_labels)

        # disc_loss = 0

        # for _ in range(self.n_critics):
        #     disc_loss = self.train_d(imgs, y_real, y_real_expanded)

        disc_loss = self.train_d(imgs, y_real, y_real_expanded)
        if step % self.n_critics == 0:
            gen_loss = self.train_g()
        else:
            gen_loss = None

        return gen_loss, disc_loss

    @tf.function
    def train_g(self):
        z = tf.random.normal((self.batch_size, 1, 1, self.z_dim))
        labels = [randint(0, self.num_labels - 1) for _ in range(self.batch_size)]
        y = self.one_hot(labels, self.num_labels)
        _, y_expanded = self.expand_labels(labels, self.num_labels)

        with tf.GradientTape() as gen_tape:
            x_fake = self.generator([z, y], training=True)
            fake_logits = self.discriminator([x_fake, y_expanded], training=True)
            loss = self.generator_loss(fake_logits)

        grad = gen_tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grad, self.generator.trainable_variables))

        return loss

    @tf.function
    def train_d(self, x_real, y_real, y_real_expanded):
        z = tf.random.normal((self.batch_size, 1, 1, self.z_dim))

        with tf.GradientTape() as disc_tape:
            x_fake = self.generator([z, y_real], training=True)
            fake_logits = self.discriminator([x_fake, y_real_expanded], training=True)

            real_logits = self.discriminator([x_real, y_real_expanded], training=True)
            cost = self.discriminator_loss(real_logits, fake_logits)
            gp = self.gradient_penalty(partial(self.discriminator, training=True), x_real, x_fake, y_real_expanded)
            cost += self.gp_mult * gp

        grad = disc_tape.gradient(cost, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grad, self.discriminator.trainable_variables))
        return cost

    def train(self):
        epoch_offset = self.checkpoint.epoch.numpy()
        seed = self.generate_z(self.num_labels)
        self.is_training = True
        gen_loss_mean = tf.keras.metrics.Mean(name='generator loss')
        disc_loss_mean = tf.keras.metrics.Mean(name='discriminator loss')

        # start tensorboard
        tensorboard = subprocess.Popen(['tensorboard', '--logdir', self.summary_path, '--bind_all'])

        for epoch in range(epoch_offset, self.num_epochs):
            # print(self.is_training)
            if self.is_training:
                start = time.time()

                for batch in self.dataset:
                    step = self.checkpoint.step.numpy()

                    gen_loss, disc_loss = self.train_step(batch, step)

                    if gen_loss is not None:
                        self.losses['gen'] = float(gen_loss.numpy())
                        gen_loss_mean.update_state(gen_loss)

                    self.losses['disc'] = float(disc_loss.numpy())

                    disc_loss_mean.update_state(disc_loss)

                    if step % self.log_freq == 0:
                        # print(self.is_training)
                        print('epoch {:04d} | step {:08d} | generator loss: {} | discriminator loss {}'.format(epoch, step,
                                                                                                               gen_loss,
                                                                                                               disc_loss))
                    if step % SUMMARY_FREQ == 0:
                        with self.summary_writer.as_default():
                            if gen_loss is not None:
                                tf.summary.scalar('generator loss', gen_loss_mean.result(), step=step)
                            tf.summary.scalar('discriminator loss', disc_loss_mean.result(), step=step)

                        if gen_loss is not None:
                            gen_loss_mean.reset_states()
                        disc_loss_mean.reset_states()

                        self.summary_writer.flush()

                    self.progress['epoch'] = int(epoch)
                    self.progress['step'] = int(step)
                    self.checkpoint.step.assign_add(1)

                    # if not self.is_training:
                    #     break

                # save some images
                self.generate_and_save_images(self.generator, epoch, self.num_labels, seed)

                if (epoch + 1) % self.checkpoint_freq == 0:
                    ckpt_save_path = self.checkpoint_manager.save()
                    self.generator.save(os.path.join(conf['project_path'], 'generator.h5'))

                self.checkpoint.epoch.assign(epoch)

                print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

        self.generate_and_save_images(self.generator, self.num_epochs, self.num_labels, seed)
        self.is_training = False
        tensorboard.terminate()

    def stop_train(self):
        self.is_training = False

    @tf.function
    def generate_samples(self, model, z, y):
        return model([z, y], training=False)

    def generate_image(self, z, y):
        image = self.generate_samples(self.generator, z, y)
        image = np.array(image)
        return image

    def generate_z(self, num_labels):
        zs = []
        for i in range(num_labels):
            z = tf.random.normal([self.num_images_in_row, 1, 1, self.z_dim])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Space')
    parser.add_argument('-n', '--num_epochs', type=int, default=10000000, help='The total number of epochs to train')
    parser.add_argument('-p', '--dataset_path', type=str, help='the path to the dataset')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name, mainly used for backend related processes, is required but can be anything')
    parser.add_argument('-x', '--image_size_x', type=int, default=512, help='the width of the images in the dataset')
    parser.add_argument('-y', '--image_size_y', type=int, default=256, help='the height of the images in the dataset')
    parser.add_argument('-P', '--project_path', type=str, help='the path where you want to store all things related to this latent space project')
    parser.add_argument('-c', '--n_critics', type=int, default=5, help='the number of critics')
    parser.add_argument('-g', '--gp_mult', type=int, default=10, help='gradient penalty multiplier')
    parser.add_argument('-G', '--learning_rate_gen', type=float, default=0.00005, help='learning rate of the generator')
    parser.add_argument('-D', '--learning_rate_disc', type=float, default=0.0001, help='learning rate of the discriminator')

    args = parser.parse_args()

    if os.path.isfile(os.path.join(args.project_path, 'project_file.json')):
        with open(os.path.join(args.project_path, 'project_file.json'), 'r') as f:
            args.__dict__ = json.load(f)

    conf = args.__dict__
    conf['ckpt_path'] = os.path.join(conf['project_path'], 'ckpts')
    conf['samples_path'] = os.path.join(conf['project_path'], 'samples')
    conf['images_path'] = os.path.join(conf['project_path'], 'images')
    conf['summary_path'] = os.path.join(conf['project_path'], 'summary')
    # conf['model_path'] = os.path.join(conf['project_path'], 'model')
    print(conf)

    if not os.path.isdir(conf['project_path']):
        os.mkdir(conf['project_path'])
        with open(os.path.join(conf['project_path'], 'project_file.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    gan = DCGAN(conf)
    gan.train()

