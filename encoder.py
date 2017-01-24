#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import numpy as np

from util import get_weights, get_biases
from batch_normalize import batch_norm

class Encoder(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        self.name_scope = u'encoder'
        
    def set_model(self, x, is_training = True, update_ave = True):
        
        layer_list = self.layer_list
        h = x
        num_layer = len(layer_list) - 1
        with tf.variable_scope(self.name_scope):
            for i, (in_dim, out_dim) in enumerate(zip(layer_list[0:], layer_list[1:])):
                weight = get_weights(i, [in_dim, out_dim], np.sqrt(1.0/in_dim))
                bias = get_biases(i, [out_dim], 0.0)
                h = tf.matmul(h, weight) + bias
                if i != num_layer - 1:
                    h = batch_norm(h, i, is_training, update_ave)
                    h = tf.nn.relu(h)
        return h

if __name__ == u'__main__':
    e = Encoder([10, 100, 1000])
    x = tf.placeholder(dtype = tf.float32, shape = [None, 10])
    e.set_model(x)
