#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Authors: liangkun@ishumei.com

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

def write_log(model, callback, names, logs, batch_no, target='stdout'):
    if target == 'tensorboard':
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
    elif target == 'stdout':
        msg = ''
        for name, value in zip(names, logs):
            if msg: msg += ', '
            msg += '%s=%.2f' % (name, value)
        print('%s %d  %s' % (model, batch_no, msg))
    else:
        print('unknown write_log target: %s' % target)

class RandomWeightedAverage(keras.layers.Add):
    '''Provided a random weighted average between two inputs.'''
    def __init__(self, batch_size, **kwargs):
        super(RandomWeightedAverage, self).__init__(**kwargs)
        self.batch_size = batch_size
    
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])