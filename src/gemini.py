#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: liangkun@ishumei.com

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from cwgan import CWGAN
from sswgan import SSWGAN

class Gemini():
    '''Fraud token detection using GANs'''
    def __init__(self, data_path, gen_samples):
        # init params
        self.gen_samples = gen_samples
        self.n_gen_samples = 150000

        # load and normalize data
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data(data_path)
        self.train_x[self.train_x < 0] = 0
        self.test_x[self.test_x < 0] = 0
        self.normalizer_init(self.train_x)
        self.train_x = self.normalize_samples(self.train_x)
        self.test_x = self.normalize_samples(self.test_x)
        
        # init models
        self.input_dim = self.train_x.shape[1]
        self.baseline = self.build_baseline_classifier()
        self.sswgan = SSWGAN(self.input_dim)

        if (self.gen_samples):
            self.cwgan = self.get_generator(retrain=False)
            self.gen_x = self.cwgan.generate(1, self.n_gen_samples)
            self.gen_y = np.empty(self.n_gen_samples, dtype=int)
            self.gen_y.fill(1)
            self.combined_train_x = np.concatenate([self.train_x, self.gen_x])
            self.combined_train_y = np.concatenate([self.train_y, self.gen_y])

    def build_baseline_classifier(self):
        inputs = keras.Input(shape=(self.input_dim,))
        predict = keras.layers.Dense(1, activation='sigmoid')(inputs)
        model = keras.Model(inputs, predict, name='baseline')
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

    def load_data(self, path):
        train_black_x, train_black_y = self.load_token_sample(path + '/train_black.csv')
        train_black_y[:] = 1
        train_white_x, train_white_y = self.load_token_sample(path + '/train_white.csv')
        train_white_y[:] = 0

        train_x = pd.concat([train_black_x, train_white_x], ignore_index=True)
        train_y = pd.concat([train_black_y, train_white_y], ignore_index=True)

        test_black_x, test_black_y = self.load_token_sample(path + '/test_black.csv')
        test_black_y[:] = 1
        test_white_x, test_white_y = self.load_token_sample(path + '/test_white.csv')
        test_white_y[:] = 0

        test_x = pd.concat([test_black_x, test_white_x], ignore_index=True)
        test_y = pd.concat([test_black_y, test_white_y], ignore_index=True)

        print('n_train_black: %d, n_train_white: %d, n_test_black: %d, n_test_white %d' % (
            train_black_x.shape[0], train_white_x.shape[0],
            test_black_x.shape[0], test_white_x.shape[0]))
        return train_x.values, train_y.values, test_x.values, test_y.values
    
    def load_token_sample(self, file):
        sample = pd.read_csv(file, sep=',', header=None)
        sample_x = sample.iloc[:, :-1]
        sample_y = sample.iloc[:, -1]
        return sample_x, sample_y

    def normalizer_init(self, train_x):
        self.feature_min = np.min(train_x, axis=0)
        self.feature_max = np.max(train_x, axis=0)
        self.feature_interval = self.feature_max - self.feature_min
        self.feature_interval[self.feature_interval == 0] = 1

    def normalize_samples(self, x):
        x = (x - self.feature_min) / self.feature_interval
        x[x > 1.0] = 1.0
        x[x < 0.0] = 0.0
        return x

    def train(self):
        if (self.gen_samples):
            train_x = self.combined_train_x
            train_y = self.combined_train_y
        else:
            train_x = self.train_x
            train_y = self.train_y
        
        print('trainning baseline classifier')
        self.baseline.fit(train_x, train_y, epochs=2, validation_split=0.1, verbose=1)

        print('training sswgan classifier')
        self.sswgan.train_autoencoder(train_x, train_y, epoches=200, sample_interval=-1)
        self.sswgan.train_classifier(train_x, train_y, epoches=2, sample_interval=-1)

    def evaluate(self):
        print('evaluating baseline classifier on test set')
        self.baseline.evaluate(self.test_x, self.test_y, verbose=1)

        print('evalueating sswgan classifier on test set')
        self.sswgan.classifier_model.evaluate(self.test_x, self.test_y, verbose=1)
    
    def get_generator(self, retrain=False, path='./token_gen_model'):
        cwgan = CWGAN(self.input_dim)
        if retrain:
            cwgan.train(self.train_x, self.train_y, epochs=200, sample_interval=-1)
            cwgan.save(path)
        else:
            cwgan.load(path)
        return cwgan
    
    def data_summary(self):
        print('train shape: %s, test shape: %s' % (self.train_x.shape, self.test_x.shape))

if __name__ == '__main__':
    gemini = Gemini('~/workspace/token_sample_201909', gen_samples=True)
    gemini.data_summary()
    gemini.train()
    gemini.evaluate()
