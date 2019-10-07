#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from functools import partial

class RandomWeightedAverage(keras.layers.Add):
    '''Provided a random weighted average between two inputs.'''
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WCGAN():
    '''WCGAN using wasserstein divergancy'''
    def __init__(self):
        self.channels = 1
        self.input_shape = (28, 28, 1)
        self.latent_dim = 100
        self.n_classes = 10

        self.n_discriminators = 5
        self.optimizer = keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.9)
        #self.optimizer = keras.optimizers.RMSprop(lr=0.00005)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # constuct computational graph for discriminator
        self.generator.trainable = False
        self.discriminator.trainable = True
        
        real_samples = keras.Input(shape=self.input_shape)

        z_dist = keras.layers.Input(shape=(self.latent_dim,))

        labels = keras.Input(shape=(1,), dtype='int32')
        fake_samples = self.generator([z_dist, labels])

        real = self.discriminator([real_samples, labels])
        fake = self.discriminator([fake_samples, labels])

        interpolated_samples = RandomWeightedAverage()([real_samples, fake_samples])
        interpolated = self.discriminator([interpolated_samples, labels])

        partial_gp_loss = partial(self.gradient_penalty_loss,
            averaged_samples=interpolated_samples)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.discriminator_model = keras.Model(inputs=[real_samples, labels, z_dist],
            outputs=[real, fake, interpolated])
        self.discriminator_model.compile(optimizer=self.optimizer,
            loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
            loss_weights=[1, 1, 10])

        # construct computational graph for generator
        self.discriminator.trainable = False
        self.generator.trainable = True
        z_dist = keras.Input(shape=(self.latent_dim,))
        labels = keras.Input(shape=(1,), dtype='int32')
        fake_samples = self.generator([z_dist, labels])
        fake = self.discriminator([fake_samples, labels])
        self.generator_model = keras.Model(inputs=[z_dist, labels], outputs=fake)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=self.optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = 2 * (gradient_l2_norm ** 6)
        # gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        noise = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(1,), dtype='int32')
        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(self.n_classes, self.latent_dim)(label))
        inputs = keras.layers.multiply([noise, label_embedding])

        hidden = keras.layers.Dense(256)(inputs)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.Dense(512)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.Dense(np.prod(self.input_shape), activation='tanh')(hidden)
        fake_sample = keras.layers.Reshape(self.input_shape)(hidden)

        model = keras.Model(inputs=[noise, label], outputs=fake_sample, name='Generator')
        model.summary()

        return model

    def build_discriminator(self):
        raw_inputs = keras.Input(shape=self.input_shape)
        flatten_inputs = keras.layers.Flatten()(raw_inputs)
        label = keras.Input(shape=(1,), dtype='int32')
        label_embedding = keras.layers.Flatten()(keras.layers.Embedding(self.n_classes, np.prod(self.input_shape))(label))
        inputs = keras.layers.multiply([flatten_inputs, label_embedding])

        hidden = keras.layers.Dense(256)(inputs)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)    
        hidden = keras.layers.Dense(self.latent_dim)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)    
        hidden = keras.layers.Dense(256)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        validity = keras.layers.Dense(1)(hidden)

        model = keras.Model(inputs=[raw_inputs, label], outputs=validity, name ='Discriminator')
        model.summary()

        return model

    def train(self, train_x, train_y, epochs, batch_size, sample_interval=50):
        # Adversarial ground truths
        real = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):
            for _ in range(self.n_discriminators):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, train_x.shape[0], batch_size)
                imgs = train_x[idx]
                labels = train_y[idx]

                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the discriminator
                d_loss = self.discriminator_model.train_on_batch([imgs, labels, noise],
                                                                [real, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            labels = np.random.randint(0, self.n_classes, batch_size).reshape(-1, 1)
            g_loss = self.generator_model.train_on_batch([noise, labels], real)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample(epoch)

    def sample(self, epoch):
        r, c = 10, 10
        fig, axs = plt.subplots(r, c)

        for i in range(r):
            noise = np.random.normal(0, 1, (c, self.latent_dim))
            labels = np.arange(0, 10)
            gen_imgs = self.generator.predict([noise, labels])
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            for j in range(c):
                axs[i,j].imshow(gen_imgs[j, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wcgan = WCGAN()

    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = (train_x.astype(np.float32) - 127.5) / 127.5
    train_x = np.expand_dims(train_x, axis=3)
    #train_y = train_y == 0
    test_x = (test_x.astype(np.float32) - 127.5) / 127.5
    test_x = np.expand_dims(test_x, axis=3)
    #test_y = test_y == 0

    wcgan.train(train_x, train_y, epochs=30000, batch_size=32, sample_interval=100)

