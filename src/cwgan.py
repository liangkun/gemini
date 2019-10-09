#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from functools import partial

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

class RandomWeightedAverage(keras.layers.Add):
    '''Provided a random weighted average between two inputs.'''
    def __init__(self, batch_size, **kwargs):
        super(RandomWeightedAverage, self).__init__(**kwargs)
        self.batch_size = batch_size
    
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class CWGAN():
    '''Conditional WGAN using wasserstein divergency'''
    def __init__(self, input_dim, batch_size=64):
        self.input_dim = input_dim
        self.latent_dim = 32
        self.n_classes = 10
        self.batch_size = batch_size
        self.label_table = np.eye(self.n_classes)

        self.n_discriminators = 5
        self.optimizer = keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # constuct computational graph for discriminator
        self.generator.trainable = False
        self.discriminator.trainable = True
        
        real_samples = keras.Input(shape=(self.input_dim,))
        label = keras.Input(shape=(self.n_classes,))

        z_dist = keras.layers.Input(shape=(self.latent_dim,))
        fake_samples = self.generator([z_dist, label])

        real = self.discriminator([real_samples, label])
        fake = self.discriminator([fake_samples, label])

        interpolated_samples = RandomWeightedAverage(batch_size=self.batch_size)([real_samples, fake_samples])
        interpolated = self.discriminator([interpolated_samples, label])

        partial_gp_loss = partial(self.gradient_penalty_loss,
            averaged_samples=interpolated_samples)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model = keras.Model(inputs=[real_samples, z_dist, label],
            outputs=[real, fake, interpolated])
        self.discriminator_model.compile(optimizer=self.optimizer,
            loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
            loss_weights=[1, 1, 2])

        # construct computational graph for generator
        self.discriminator.trainable = False
        self.generator.trainable = True
        z_dist = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(self.n_classes,))

        fake_samples = self.generator([z_dist, label])
        fake = self.discriminator([fake_samples, label])
        self.generator_model = keras.Model(inputs=[z_dist, label], outputs=fake)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_l2_norm ** 6
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        noise = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(self.n_classes,))
        inputs = keras.layers.concatenate([noise, label], axis=1)

        hidden = keras.layers.Dense(256)(inputs)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.Dense(512)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.Dense(512)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        fake_sample = keras.layers.Dense(self.input_dim, activation='sigmoid')(hidden)

        model = keras.Model(inputs=[noise, label], outputs=fake_sample, name='Generator')
        model.summary()

        return model

    def build_discriminator(self):
        raw_inputs = keras.Input(shape=(self.input_dim,))
        label = keras.Input(shape=(self.n_classes,))
        inputs = keras.layers.concatenate([raw_inputs, label], axis=1)

        hidden = keras.layers.Dense(512)(inputs)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.Dense(256)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)  
        hidden = keras.layers.Dense(self.latent_dim)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)    
        hidden = keras.layers.Dense(256)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.Dense(512)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        validity = keras.layers.Dense(1)(hidden)

        model = keras.Model(inputs=[raw_inputs, label], outputs=validity, name ='Discriminator')
        model.summary()

        return model

    def train(self, train_x, train_y, epochs, sample_interval=50):
        # setup logs
        discriminator_callback = TensorBoard('./log')
        discriminator_callback.set_model(self.discriminator_model)

        # adversarial ground truths
        real = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):
            for _ in range(self.n_discriminators):
                # ---------------------
                #  train Discriminator
                # ---------------------
                # Select a random batch of samples
                idx = np.random.randint(0, train_x.shape[0], self.batch_size)
                imgs = train_x[idx]
                labels = train_y[idx]
                labels = self.label_table[labels]

                # sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # train the discriminator
                d_loss = self.discriminator_model.train_on_batch([imgs, noise, labels],
                                                                [real, fake, dummy])
            write_log(discriminator_callback, ['real_loss', 'fake_loss', 'gradient'], d_loss, epoch)

            # ---------------------
            #  train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            labels = np.random.randint(0, self.n_classes, self.batch_size)
            labels = self.label_table[labels]
            g_loss = self.generator_model.train_on_batch([noise, labels], real)

            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # ff at save interval => save generated image samples
            if sample_interval > 0 and epoch % sample_interval == 0:
                self.sample(epoch)

    def sample(self, epoch):
        r, c = 10, 10
        fig, axs = plt.subplots(r, c)

        for i in range(r):
            noise = np.random.normal(0, 1, (c, self.latent_dim))
            labels = np.arange(0, 10)
            labels = self.label_table[labels]
            gen_imgs = self.generator.predict([noise, labels])
            gen_imgs = np.reshape(gen_imgs, (-1, 28, 28))

            for j in range(c):
                axs[i,j].imshow(gen_imgs[j, :, :], cmap='gray')
                axs[i,j].axis('off')
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()
    
    def save(self, path):
        self.generator.save(path + '/generator.h5')
        self.discriminator.save(path + '/discriminator.h5')
    
    def load(self, path):
        self.generator = keras.models.load_model(path + '/generator.h5')
        self.discriminator = keras.models.load_model(path + '/discriminator.h5')

    def generate(self, labels, n_samples=1):
        pass


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x.astype(np.float32) / 255
    input_dim = np.prod(train_x.shape[1:])
    train_x = np.reshape(train_x, (-1, input_dim))

    cwgan = CWGAN(input_dim=input_dim)
    cwgan.train(train_x, train_y, epochs=20001, sample_interval=200)
    cwgan.save('model')

    cwgan2 = CWGAN(input_dim=input_dim)
    cwgan2.load('model')
    cwgan2.sample(epoch=1)

