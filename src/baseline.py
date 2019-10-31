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

class Baseline():
    '''Baseline feedforward neural network classifier'''
    def __init__(self, input_dim, name='baseline', log=None):
        # init params
        self.input_dim = input_dim
        self.name = name
        self.log = log

        # create models
        self.model = self.build_baseline_classifier()
        self.baseline_trained = False

    def build_baseline_classifier(self):
        inputs = keras.Input(shape=(self.input_dim,))
        hidden = keras.layers.Dense(96)(inputs)
        hidden = keras.layers.LeakyReLU(0.2)(hidden)
        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        hidden = keras.layers.Dense(48)(inputs)
        hidden = keras.layers.LeakyReLU(0.2)(hidden)
        hidden = keras.layers.BatchNormalization(momentum=0.8)(hidden)
        predict = keras.layers.Dense(1, activation='sigmoid')(hidden)
        model = keras.Model(inputs, predict, name=self.name)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, train_x, train_y, epochs=100):        
        print('trainning %s classifier' % self.name)
        self.model.fit(train_x, train_y, epochs=epochs, validation_split=0.1)

    def save(self, modeldir):
        self.model.save(modeldir + '/%s.h5' % self.name)

    def load(self, modeldir):
        self.model = keras.models.load_model(modeldir + '/%s.h5' % self.name)

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x.astype(np.float32) / 255
    train_y = (train_y < 5).astype(int)
    test_x = test_x.astype(np.float32) / 255
    test_y = (test_y < 5).astype(int)

    input_dim = np.prod(train_x.shape[1:])
    train_x = np.reshape(train_x, (-1, input_dim))
    test_x = np.reshape(test_x, (-1, input_dim))

    baseline = Baseline(input_dim)
    baseline.train(train_x, train_y, epochs=50)
    baseline.model.evaluate(test_x, test_y)
