#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import get_weights, get_biases
        
def batch_norm(x, name, is_training = True, update_ave = True):
    decay_rate = 0.99
    
    shape = x.get_shape().as_list()
    dim = shape[-1]
    if len(shape) == 2:
        mean, var = tf.nn.moments(x, [0], name = 'moments_bn_{}'.format(name))
    elif len(shape) == 4:
        mean, var = tf.nn.moments(x, [0, 1, 2], name = 'moments_bn_{}'.format(name))

    ave_mean  = get_biases('ave_mean_bn_{}'.format(name), [1, dim], 0.0, False)
    ave_var = get_biases('ave_var_bn_{}'.format(name), [1, dim], 1.0, False)
    
    beta  = get_biases('beta_bn_{}'.format(name), [1, dim], 0.0)
    gamma = get_biases('gamma_bn_{}'.format(name), [1, dim], 1.0)

    if is_training:
        if update_ave:
            ave_mean_assign_op = tf.assign(ave_mean, decay_rate * ave_mean
                                           + (1 - decay_rate) * mean)
            ave_var_assign_op = tf.assign(ave_var,
                                          decay_rate * ave_var
                                          + (1 - decay_rate) * var)
        else:
            ave_mean_assign_op = tf.no_op()
            ave_var_assign_op = tf.no_op()

        with tf.control_dependencies([ave_mean_assign_op, ave_var_assign_op]):
            ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
            
    else:
        ret = gamma * (x - ave_mean) / tf.sqrt(1e-6 + ave_var) + beta
        
    return ret
