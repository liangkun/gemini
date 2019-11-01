#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: liangkun@ishumei.com

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from functools import partial
from utils import write_log
from utils import RandomWeightedAverage

class SSWGAN():
    '''Semi Supervised WGAN.'''
    def __init__(self, input_dim, latent_dim=32, batch_size=64, n_discriminators=5, log=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.optimizer = keras.optimizers.Adam(0.0002, 0.5)
        self.n_discriminators = n_discriminators
        self.log = log

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        real_samples = keras.Input(shape=(self.latent_dim,))
        real = self.discriminator(real_samples)
        fake_samples = keras.Input(shape=(self.latent_dim,))
        fake = self.discriminator(fake_samples)
        interpolated_samples = RandomWeightedAverage(batch_size=self.batch_size)(
            [real_samples, fake_samples])
        interpolated = self.discriminator(interpolated_samples)
        self.discriminator_model = keras.Model(
            inputs=[real_samples, fake_samples],
            outputs=[real, fake, interpolated])

        partial_gp_loss = partial(self.gradient_penalty_loss,
            averaged_samples=interpolated_samples)
        partial_gp_loss.__name__ = 'gradient_penalty'
        self.discriminator_model.compile(
            loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
            loss_weights=[1, 1, 2],
            optimizer=self.optimizer)

        # build and compile the adversarial autoencoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        inputs = keras.Input(shape=(self.input_dim,))
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        real = self.discriminator(latent)

        # for adversarial autoencoder, only train the generator        
        self.discriminator.trainable = False
        self.adversarial_autoencoder = keras.Model(inputs, [reconstructed, real])
        self.adversarial_autoencoder.compile(loss=['mse', self.wasserstein_loss],
            loss_weights=[2, 1], optimizer=self.optimizer)
        
        # build final classifier
        self.classifier = self.build_classifier()
        inputs = keras.Input(shape=(self.input_dim,))
        latent = self.encoder(inputs)
        predict = self.classifier(latent)
        self.encoder.trainable = False  # freeze encoder in classifier
        self.classifier_model = keras.Model(inputs, predict)
        self.classifier_model.compile(loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])

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
    
    def build_encoder(self):
        inputs = keras.Input(shape=(self.input_dim,))
        hidden = keras.layers.Dense(96)(inputs)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
#        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        hidden = keras.layers.Dense(48)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
#        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        encoded = keras.layers.Dense(self.latent_dim)(hidden)
        model = keras.Model(inputs, encoded, name='encoder')
        model.summary()

        return model

    def build_decoder(self):
        encoded = keras.Input(shape=(self.latent_dim,))
        hidden = keras.layers.Dense(48)(encoded)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
#        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        hidden = keras.layers.Dense(96)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
#        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        reconstructed = keras.layers.Dense(self.input_dim, activation='sigmoid')(hidden)
        model = keras.Model(encoded, reconstructed, name='decoder')
        model.summary()

        return model

    def build_discriminator(self):
        encoded = keras.Input(shape=(self.latent_dim,))
        hidden = keras.layers.Dense(96)(encoded)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
#        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        hidden = keras.layers.Dense(48)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
#        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        validity = keras.layers.Dense(1)(hidden)
        model = keras.Model(encoded, validity, name='discriminator')
        model.summary()

        return model

    def build_classifier(self):
        encoded = keras.Input(shape=(self.latent_dim,))
        hidden = keras.layers.Dense(96)(encoded)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        hidden = keras.layers.Dense(48)(hidden)
        hidden = keras.layers.LeakyReLU(alpha=0.2)(hidden)
        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        validity = keras.layers.Dense(1, activation="sigmoid")(hidden)
        model = keras.Model(encoded, validity, name='classifier')
        model.summary()

        return model

    def train(self, train_x, train_y, epochs, sample_interval):
        self.train_autoencoder(train_x, train_y, epochs, sample_interval)
        self.train_classifier(train_x, train_y, epochs, sample_interval)
    
    def train_autoencoder(self, train_x, train_y, epochs, sample_interval=-1):
        # setup logs
        if self.log:
            tb_callback = TensorBoard(self.log)
            tb_callback.set_model(self.adversarial_autoencoder)
        tb_names = ['d_loss', 'd_real_loss', 'd_fake_loss', 'd_gradient',
                    'g_loss', 'g_mse', 'g_real_loss']

        # adversarial ground truths
        real = np.ones((self.batch_size, 1))
        fake = -np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            tb_values = []
            # ------------------------------
            # train discriminator
            # ------------------------------
            for _ in range(self.n_discriminators):
                latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))
                idx = np.random.randint(0, train_x.shape[0], self.batch_size)
                inputs = train_x[idx]
                latent_fake = self.encoder.predict(inputs)
                d_loss = self.discriminator_model.train_on_batch([latent_real, latent_fake], [real, fake, dummy])
            tb_values.extend(d_loss)

            # ------------------------------
            # train generator
            # ------------------------------
            idx = np.random.randint(0, train_x.shape[0], self.batch_size)
            inputs = train_x[idx]

            noisy_inputs = inputs.copy()
            noisy_idx = np.random.randint(0, train_x.shape[1],
                (self.batch_size, int(train_x.shape[1] * 0.3)))
            for i in range(noisy_idx.shape[0]):
                noisy_inputs[i, noisy_idx[i]] = 0

            g_loss = self.adversarial_autoencoder.train_on_batch(noisy_inputs, [inputs, real])
            tb_values.extend(g_loss)

            write_log('SSWGAN', None, tb_names, tb_values, epoch, target='stdout')
            if self.log:
                write_log('SSWGAN', tb_callback, tb_names, tb_values, epoch, target='tensorboard')
            
            # ff at save interval => save generated image samples
            if sample_interval > 0 and epoch % sample_interval == 0:
                self.sample(epoch)

    def train_classifier(self, train_x, train_y, epochs, sample_interval=-1):
        callbacks = []
        if self.log:
            callbacks.append(TensorBoard(log_dir=self.log))

        loss = self.classifier_model.fit(train_x, train_y,
            batch_size=self.batch_size, epochs=epochs, validation_split=0.2, verbose=1,
            callbacks=callbacks)

    def sample(self, epoch):
        r, c = 5, 5
        fig, axs = plt.subplots(r, c)

        for i in range(r):
            noise = np.random.normal(0, 1, (c, self.latent_dim))
            gen_imgs = self.decoder.predict(noise)
            gen_imgs = np.reshape(gen_imgs, (-1, 28, 28))
            for j in range(c):
                axs[i,j].imshow(gen_imgs[j, :, :], cmap='gray')
                axs[i,j].axis('off')
        fig.savefig("images/sswgan_mnist_%d.png" % epoch)
        plt.close()

    def save(self, path):
        self.classifier_model.save(path + '/sswgan_classifier.h5')
        self.encoder.save(path + '/sswgan_encoder.h5')
    
    def load(self, path):
        self.classifier_model = keras.models.load_model(path + '/sswgan_classifier.h5')
        self.encoder = keras.models.load_model(path + '/sswgan_encoder.h5')


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x.astype(np.float32) / 255
    train_y = (train_y == 0)
    test_x = test_x.astype(np.float32) / 255
    test_y = (test_y == 0)

    input_dim = np.prod(train_x.shape[1:])
    train_x = np.reshape(train_x, (-1, input_dim))
    test_x = np.reshape(test_x, (-1, input_dim))

    sswgan = SSWGAN(input_dim, batch_size=128, log='./sswgan_log')
    sswgan.train_autoencoder(train_x, train_y, epochs=20001, sample_interval=200)
    sswgan.train_classifier(train_x, train_y, epochs=101, sample_interval=-1)

    print("evaluate on test set")
    sswgan.classifier_model.evaluate(test_x, test_y, verbose=1)
