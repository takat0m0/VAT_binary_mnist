#! -*- coding:utf-8 -*-

u'''
for VAT class, I refer to 
https://gist.github.com/takerum/b1571b48c89dae13ae282795740c1b59
and
http://musyoku.github.io/2016/12/10/Distributional-Smoothing-with-Virtual-Adversarial-Training/
'''

import os
import sys

import numpy as np
import tensorflow as tf


def zero_one(target):
    return 0 if target < 0.5 else 1

def _get_one_hot(target_index, num_batch, num_dim):
    indices = [[_, target_index] for _ in range(num_batch)]
    values = [1.0] * num_batch 
    ret = tf.sparse_tensor_to_dense(
        tf.SparseTensor( indices=indices, values=values, shape=[num_batch, num_dim] ), 0.0 )
    return ret

def _get_weights(shape, _stddev=1.0):
    initial = tf.truncated_normal(shape, stddev=_stddev)
    return tf.get_variable('weight', shape)

def _get_biases(shape, value=0.0):
    initial = tf.constant(value, shape=shape)
    return tf.get_variable('bias', shape)


class Encoder(object):
    def __init__(self, layer_list):
        self.layer_list = layer_list
        
    def set_model(self, x, reuse = False):
        layer_list = self.layer_list
        h = x
        for i, (in_dim, out_dim) in enumerate(zip(layer_list[0:], layer_list[1:])):
            with tf.variable_scope('encoder_x{}'.format(i), reuse = reuse):        
                weight = _get_weights(shape=[in_dim, out_dim], _stddev = 0.1)
                bias = _get_biases([out_dim], value=0.0)
                ret = tf.matmul(h, weight) + bias
                h = tf.nn.relu(ret)
        return ret


class VAT_Params(object):
    def __init__(self):
        self.xi = 1.0
        self.epsilon = 1.0
        self.Lambda = 1.0
        self.num_iter = 1

        
class VAT(object):
    def __init__(self, _encoder):
        self.encoder = _encoder
        self.params = VAT_Params()
        
    def set_model(self, x, logits):
        logits = tf.stop_gradient(logits)
        r_vadv = self._get_r_vadv(x, logits)
        logit_v = self.encoder.set_model(x + r_vadv, reuse = True)
        return tf.identity(self.params.Lambda * self._get_kl(logits, logit_v))
        
    def _get_kl(self, logits, v_logits):
        p   = tf.nn.softmax(logits)
        p_v = tf.nn.softmax(v_logits)
        kl = tf.reduce_mean(tf.reduce_sum(p * (tf.log(p) - tf.log(p_v)), 1))
        return tf.identity(kl)

    def _get_r_vadv(self, x, logits):
        d = self._normalize(tf.random_normal(shape = tf.shape(x)))

        for num_iter in range(self.params.num_iter):
            d_logits = self.encoder.set_model(x + self.params.xi * d, reuse = True)
            g_d = tf.gradients(self._get_kl(logits, d_logits), [d])
            d = tf.stop_gradient(g_d[0])
            d = self._normalize(d)

        return self.params.epsilon * d

    def _normalize(self, target_vec):
        norm = (1e-12 + tf.reduce_max(tf.abs(target_vec), 1, keep_dims=True))
        target_vec = target_vec/norm

        norm = tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(target_vec, 2.0), 1, keep_dims=True))
        target_vec = target_vec/norm 

        return target_vec
    
class Model(object):
    def __init__(self, _layer_list, _batch_size):
        self.input_dim  = _layer_list[0]
        self.output_dim = _layer_list[-1]
        
        self.encoder = Encoder(_layer_list)
        self.vat = VAT(self.encoder)
        
        self.batch_size = _batch_size
        self.lr = 0.0002
        

    def set_model(self):
        self.x = tf.placeholder("float", shape=[None, self.input_dim])
        self.set_model_label()
        self.set_model_unlabel()
        
    def set_model_label(self):

        self.y = tf.placeholder("float", shape=[None, self.output_dim])

        # encode
        logits = self.encoder.set_model(self.x)

        # probability
        self.prob = tf.nn.softmax(logits)

        # classifier loss
        obj = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))

        # set optimizer
        self.train_label = tf.train.AdamOptimizer(self.lr).minimize(obj)
        
    def set_model_unlabel(self):

        # encode
        logits = self.encoder.set_model(self.x, reuse = True)

        # VAT
        vat_obj = self.vat.set_model(self.x, logits)
        
        # set optimizer
        self.train_unlabel = tf.train.AdamOptimizer(self.lr).minimize(vat_obj)


    def training_label(self, sess, inputs, labels):
        sess.run(self.train_label,
                 feed_dict = {self.x: inputs,
                              self.y:labels})
        
    def training_unlabel(self, sess, inputs):
        sess.run(self.train_unlabel,
                 feed_dict = {self.x: inputs})

    def get_prob(self, sess, inputs):
        return sess.run(self.prob, feed_dict = {self.x: inputs})
    

if __name__ == u'__main__':

    model = Model(10)
    obj = model.set_model()
