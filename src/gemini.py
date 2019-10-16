#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: liangkun@ishumei.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from cwgan import CWGAN
from sswgan import SSWGAN
from sklearn.metrics import roc_curve, auc
import json

class Gemini():
    '''Fraud token detection using GANs'''
    def __init__(self, input_dim, name='gemini', n_gen_samples=-1, clf_epochs=100,
                 modeldir='./model', logdir='./log', aae_retrain=True, generator_retrain=False,
                 aae_batches=64, generator_batches=64, bw_ratio=None):
        # init params
        self.input_dim = input_dim
        self.name = name
        self.n_gen_samples = n_gen_samples
        self.clf_epochs = clf_epochs
        self.modeldir = modeldir
        self.logdir = logdir
        self.aae_retrain = aae_retrain
        self.generator_retrain = generator_retrain
        self.aae_batches = aae_batches
        self.generator_batches = generator_batches
        self.bw_ratio = bw_ratio

        # create models
        self.baseline = self.build_baseline_classifier()
        self.baseline_trained = False

        self.sswgan = SSWGAN(self.input_dim, batch_size=64, log=self.logdir)
        self.sswgan_trained = False
        
        self.cwgan = CWGAN(self.input_dim, n_classes=2, batch_size=64, log=self.logdir)
        self.cwgan_trained = False

        self.normalizer_trained = False

    def build_baseline_classifier(self):
        inputs = keras.Input(shape=(self.input_dim,))
        hidden = keras.layers.Dense(96)(inputs)
        hidden = keras.layers.LeakyReLU(0.2)(hidden)
        hidden = keras.layers.Dense(48)(inputs)
        hidden = keras.layers.LeakyReLU(0.2)(hidden)
        predict = keras.layers.Dense(1, activation='sigmoid')(hidden)
        model = keras.Model(inputs, predict, name='baseline')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def normalizer_init(self, train_x):
        if self.normalizer_trained:
            return

        self.feature_min = np.min(train_x, axis=0)
        self.feature_max = np.max(train_x, axis=0)
        self.feature_interval = self.feature_max - self.feature_min
        self.feature_interval[self.feature_interval == 0] = 1
        self.normalizer_trained = True

    def normalize(self, x):
        x = (x - self.feature_min) / self.feature_interval
        x[x > 1.0] = 1.0
        x[x < 0.0] = 0.0
        return x
    
    def generate_sample(self, train_x, train_y):
        if not self.cwgan_trained:
            self.generator_init(train_x, train_y)

        gen_x = self.cwgan.generate(1, self.n_gen_samples)
        gen_y = np.empty(self.n_gen_samples, dtype=int)
        gen_y.fill(1)
        combined_train_x = np.concatenate([train_x, gen_x])
        combined_train_y = np.concatenate([train_y, gen_y])
        return combined_train_x, combined_train_y

    def train(self, train_x, train_y):
        self.normalizer_init(train_x)
        train_x = self.normalize(train_x)

        if self.n_gen_samples > 0:
            combined_train_x, combined_train_y = self.generate_sample(train_x, train_y)
            train_x = combined_train_x
            train_y = combined_train_y
        
        print('trainning baseline classifier')
        self.baseline.fit(train_x, train_y, epochs=self.clf_epochs, validation_split=0.1)

        print('training sswgan classifier')
        self.sswgan.train_autoencoder(train_x, train_y, epochs=self.aae_batches, sample_interval=-1)
        self.sswgan.train_classifier(train_x, train_y, epochs=self.clf_epochs)
    
    def generator_init(self, train_x=None, train_y=None):
        if self.cwgan_trained:
            return

        if self.generator_retrain:
            self.cwgan.train(train_x, train_y, epochs=self.generator_batches, sample_interval=-1)
            self.cwgan.save(self.modeldir)
        else:
            self.cwgan.load(self.modeldir)
        self.cwgan_trained = True
    
    def save(self):
        self.baseline.save(self.modeldir + '/baseline.h5')
        self.sswgan.save(self.modeldir)

    def load(self):
        self.sswgan.load(self.modeldir)
        self.baseline = keras.models.load_model(self.modeldir + '/baseline.h5')

# -- aux functions: mnist data for tests, input_dim = 784
def load_mnist_data(bw_ratio=None):
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x.astype(np.float32) / 255
    train_y = (train_y == 0).astype(int)
    test_x = test_x.astype(np.float32) / 255
    test_y = (test_y == 0).astype(int)

    input_dim = np.prod(train_x.shape[1:])
    train_x = np.reshape(train_x, (-1, input_dim))
    test_x = np.reshape(test_x, (-1, input_dim))
    return train_x, train_y, test_x, test_y

# -- aux functions: load token data in csv format
def load_token_data(path, bw_ratio=None):

    def load_token_sample(file):
        sample = pd.read_csv(file, sep=',', header=None)
        sample_x = sample.iloc[:, :-1]
        sample_y = sample.iloc[:, -1]
        return sample_x, sample_y
    
    train_black_x, train_black_y = load_token_sample(path + '/train_black.csv')
    train_black_y[:] = 1
    train_white_x, train_white_y = load_token_sample(path + '/train_white.csv')
    train_white_y[:] = 0

    train_x = pd.concat([train_black_x, train_white_x], ignore_index=True)
    train_y = pd.concat([train_black_y, train_white_y], ignore_index=True)

    test_black_x, test_black_y = load_token_sample(path + '/test_black.csv')
    test_black_y[:] = 1
    test_white_x, test_white_y = load_token_sample(path + '/test_white.csv')
    test_white_y[:] = 0
    test_white_x = test_white_x.sample(test_black_x.shape[0])
    test_white_y = test_white_y[:test_black_x.shape[0]]

    test_x = pd.concat([test_black_x, test_white_x], ignore_index=True)
    test_y = pd.concat([test_black_y, test_white_y], ignore_index=True)

    train_x[train_x < 0] = 0
    test_x[stest_x < 0] = 0
    return train_x.values, train_y.values, test_x.values, test_y.values

def evaluate(model, test_x, test_y, plt, colors):
    test_x = model.normalize(test_x.copy())
    baseline_predict_y = model.baseline.predict(test_x)
    baseline_fpr, baseline_tpr, _ = roc_curve(test_y, baseline_predict_y)
    print('evaluating %s baseline classifier on test set, auc: %4f' %
          (model.name, auc(baseline_fpr, baseline_tpr)))

    sswgan_predict_y = model.sswgan.classifier_model.predict(test_x)
    sswgan_fpr, sswgan_tpr, _ = roc_curve(test_y, sswgan_predict_y)
    print('evalueating %s sswgan classifier on test set, auc: %4f' %
          (model.name, auc(sswgan_fpr, sswgan_tpr)))

    plt.plot(baseline_fpr, baseline_tpr, colors[0], label='BASELINE: %s' % model.name)
    plt.plot(sswgan_fpr, sswgan_tpr, colors[1], label='SSWGAN: %s' % model.name)

if __name__ == '__main__':
    # token input path '~/workspace/token_sample_201909'
    train_x, train_y, test_x, test_y = load_mnist_data()

    gemini_raw = Gemini(input_dim=784, name='gemini_raw', modeldir='./raw_model',
                    clf_epochs=50, aae_batches=20001, generator_retrain=True, generator_batches=20001)

    gemini = Gemini(input_dim=784, n_gen_samples=10000, modeldir='./model',
                    clf_epochs=50, aae_batches=20001, generator_retrain=True, generator_batches=20001)

    gemini_raw.train(train_x, train_y)
    gemini_raw.save()

    gemini.train(train_x, train_y)
    gemini.save()

    # evaluate models
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    evaluate(gemini_raw, test_x, test_y, plt, ['r', 'b'])
    evaluate(gemini, test_x, test_y, plt, ['y', 'g'])
    plt.legend()
    plt.savefig("./roc_curve.png")
    plt.close()
