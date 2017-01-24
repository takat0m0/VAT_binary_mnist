#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf

def get_weights(name, shape, stddev = 0.1, trainable = True):
    return tf.get_variable('weight{}'.format(name), shape, trainable = trainable,
                           initializer = tf.random_normal_initializer(0.0, stddev))

def get_biases(name, shape, value = 0.0, trainable = True):
    return tf.get_variable('bias{}'.format(name), shape, trainable = trainable,
                           initializer = tf.constant_initializer(value))

