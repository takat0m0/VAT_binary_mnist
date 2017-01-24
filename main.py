#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from make_fig import get_batch    

def _test(sess, model, test_file):
    cnt = 0
    ok_cnt = 0
    num_line = sum(1 for _ in open(test_file))
    
    with open(test_file, "r") as f:
        for _ in range(num_line):
            cnt += 1

            batch_labels, batch_figs = get_batch(f, 1)
            prob = model.get_prob(sess, [batch_figs[0]])
            prob = np.asarray(prob[0])
            predict_index = np.argmax(prob)
            
            target_index = np.argmax(np.asarray(batch_labels[0]))

            if predict_index == target_index:
                ok_cnt += 1
    return float(ok_cnt)/float(cnt)
        
if __name__ == u'__main__':

    labeled_file = 'labeled.csv'
    unlabeled_file = 'unlabeled.csv'

    test_file = 'mnist_test.csv'
    
    dump_dir = 'sample_result'
    if os.path.exists(dump_dir) == False:
        os.mkdir(dump_dir)
        
    # parameter
    batch_size = 100
    layer_list = [28 * 28 * 1, 1200, 600, 300, 150, 10]
    #layer_list = [28 * 28 * 1, 1200, 600, 10]
    epoch_num = 100
    
    # make model
    model = Model(layer_list, batch_size)
    model.set_model()

    # get_label
    with open(labeled_file, 'r') as f_obj:
        labels, labeled_figs = get_batch(f_obj, 100)
        
    num_one_epoch = sum(1 for _ in open(unlabeled_file)) //batch_size
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(epoch_num):
            with open(unlabeled_file, 'r') as f_obj:
                for step in range(num_one_epoch):
                    batch_labels, batch_figs = get_batch(f_obj, batch_size)
                    #model.training(sess, labeled_figs, batch_figs, labels)
                    model.training(sess, batch_figs, batch_figs, batch_labels)

            model.change_lr(sess)
            ratio = _test(sess, model, test_file)
            print('epoch {}: ratio = {}'.format(epoch, ratio))
