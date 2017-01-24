#! -*- coding:utf-8 -*-

u'''
for VAT class, I refer to 
https://gist.github.com/takerum/b1571b48c89dae13ae282795740c1b59
and
http://musyoku.github.io/2016/12/10/Distributional-Smoothing-with-Virtual-Adversarial-Training/
'''

import os
import sys

import tensorflow as tf
import numpy as np

from encoder import Encoder

class VAT_Params(object):
    def __init__(self):
        self.xi = 1.0e-6
        self.epsilon = 2.0
        self.Lambda = 1.0
        self.num_iter = 1

        
class VAT(object):
    def __init__(self, _encoder):
        self.encoder = _encoder
        self.params = VAT_Params()
        
    def set_model(self, x, logits):
        logits = tf.stop_gradient(logits)
        r_vadv = self._get_r_vadv(x, logits)
        logit_v = self.encoder.set_model(x + r_vadv, is_training = True, update_ave = False)
        return tf.identity(self.params.Lambda * self._get_kl(logits, logit_v))
        
    def _get_kl(self, logits, v_logits):
        p   = tf.nn.softmax(logits)
        p_v = tf.nn.softmax(v_logits)
        kl = tf.reduce_mean(tf.reduce_sum(p * (tf.log(p) - tf.log(p_v)), 1))
        return tf.identity(kl)

    def _get_r_vadv(self, x, logits):
        d = self._normalize(tf.random_normal(shape = tf.shape(x)))

        for num_iter in range(self.params.num_iter):
            d_logits = self.encoder.set_model(x + self.params.xi * d,
                                              is_training = True, update_ave = False)
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
    
if __name__ == u'__main__':
    e = Encoder([10, 100, 1000])
    x = tf.placeholder(dtype = tf.float32, shape = [None, 10])
    logits = e.set_model(x)
    tf.get_variable_scope().reuse_variables()
    v = VAT(e)
    v.set_model(x, logits)
