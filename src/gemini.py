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
    def __init__(self, data_path):
        # load and normalize data
        self.train_x, self.train_y, self.test_x, self.test_y = self.load_data(data_path)
        self.train_x[self.train_x < 0] = 0
        self.test_x[self.test_x < 0] = 0
        self.normalizer_init(self.train_x)
        self.train_x = self.normalize_samples(self.train_x)
        self.test_x = self.normalize_samples(self.test_x)
        
        # init models
        self.input_dim = self.train_x.shape[1]
        self.sswgan = SSWGAN(self.input_dim)
        self.cwgan = CWGAN(self.input_dim)

        self.build_test_classifier()
    
    def build_test_classifier(self):
        inputs = keras.Input(shape=(self.input_dim,))
        predict = keras.layers.Dense(1, activation='sigmoid')(inputs)
        self.test_classifier = keras.Model(inputs, predict)
        self.test_classifier.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    

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

        return train_x.values, train_y.values, test_x.values, test_y.values
    
    def load_token_sample(self, file):
        sample = pd.read_csv(file, sep=',', header=None)
        sample_x = sample.iloc[:, :-1]
        sample_x = sample_x.dropna()
        sample_y = sample.iloc[:, -1]
        return sample_x, sample_y

    def normalizer_init(self, train_x):
        self.feature_min = np.min(train_x, axis=0)
        self.feature_max = np.max(train_x, axis=0)
        self.feature_interval = self.feature_max - self.feature_min

    def normalize_samples(self, x):
        x = (x - self.feature_min) / self.feature_interval
        x[x > 1.0] = 1.0
        x[x < 0.0] = 0.0
        return x

    def train(self):
        print(self.train_x)
        self.test_classifier.fit(self.train_x, self.train_y, validation_split=0.1, verbose=1)
        #self.sswgan.train_autoencoder(self.train_x, self.train_y, epoches=21, sample_interval=-1)
        #self.sswgan.train_classifier(self.train_x, self.train_y, epoches=21, sample_interval=-1)

    def evaluate(self):
        print('evaluating on test set')
        self.test_classifier.evaluate(self.test_x, self.test_y, verbose=1)
        #self.sswgan.classifier_model.evaluate(self.test_x, self.test_y, verbose=1)
    
    def data_summary(self):
        print('train shape: %s, test shape: %s' % (self.train_x.shape, self.test_x.shape))

if __name__ == '__main__':
    gemini = Gemini('~/workspace/token_sample_201909_test')
    gemini.data_summary()
    gemini.train()
    gemini.evaluate()
