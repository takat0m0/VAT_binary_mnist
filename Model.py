#! -*- coding:utf-8 -*-


import os
import sys

import numpy as np
import tensorflow as tf
from encoder import Encoder
from vat import VAT
    
class Model(object):
    def __init__(self, _layer_list, _batch_size):
        self.input_dim  = _layer_list[0]
        self.output_dim = _layer_list[-1]

        self.encoder = Encoder(_layer_list)
        self.vat = VAT(self.encoder)
        
        self.batch_size = _batch_size
        self.init_lr = 0.002
        self.lr_decay_rate = 0.9

        self.lr = tf.Variable(
            name = "learning_rate",
            initial_value = self.init_lr,
            trainable = False)
        
        self.lr_op = tf.assign(self.lr, self.lr_decay_rate * self.lr)

    def set_model(self):
        
        self.x = tf.placeholder("float", shape=[None, self.input_dim])
        self.y = tf.placeholder("float", shape=[None, self.output_dim])

        #-- labeled data ----------
        # encode
        logits = self.encoder.set_model(self.x)

        # classifier loss
        obj = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))

        # for reuse variable
        tf.get_variable_scope().reuse_variables()

        # VAT
        vat_obj = self.vat.set_model(self.x, logits)
        obj += vat_obj

        self.x_unlabel = tf.placeholder("float", shape=[None, self.input_dim])
        u'''
        # -- unlabeled data --------
        # encode
        u_logits = self.encoder.set_model(self.x_unlabel, True, False)

        # VAT
        u_vat_obj = self.vat.set_model(self.x_unlabel, u_logits)
        obj += u_vat_obj
        '''
        # -- set optimizer ----------
        self.train = tf.train.AdamOptimizer(self.lr).minimize(obj)

        # -- for get probability ----
        self.prob = tf.nn.softmax(self.encoder.set_model(self.x, False, False))

    def training(self, sess, inputs, u_inputs, labels):
        sess.run(self.train,
                 feed_dict = {self.x: inputs,
                              self.x_unlabel: u_inputs,
                              self.y:labels})
    def change_lr(self, sess):
        sess.run(self.lr_op)
        
    def get_prob(self, sess, inputs):
        return sess.run(self.prob, feed_dict = {self.x: inputs})
    

if __name__ == u'__main__':
    batch_size = 100
    layer_list = [28 * 28 * 1, 1200, 600, 300, 150, 10]
    model = Model(layer_list, batch_size)
    model.set_model()
