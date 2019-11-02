#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy
import random
import time
import os, sys
from fmnist_dataset import Fashion_MNIST
from model import CNN

import pickle

tf.app.flags.DEFINE_integer("rand_seed", 2019,
                            "seed for random number generaters")
tf.app.flags.DEFINE_string("gpu", "0",
                           "select one gpu")

tf.app.flags.DEFINE_float("keep_prob_fc", 0.8,
                          "probability of keeping in FC layers")
tf.app.flags.DEFINE_float("keep_prob_conv", 1.0,
                          "probability of keeping in convolutional layers")
tf.app.flags.DEFINE_float("lr_init", 2e-2,
                          "learning rate initialization value")
tf.app.flags.DEFINE_float("lr_base", 1e-4,
                          "learning rate base value")
tf.app.flags.DEFINE_integer("lr_decay", 2000,
                            "learning rate exponential decay step")
tf.app.flags.DEFINE_float("grad_clip", 3,
                          "gradient clipping by value")

tf.app.flags.DEFINE_integer("batch_size", 32,
                            "mini-batch size")
tf.app.flags.DEFINE_integer("n_epoch", 100,
                            "max epoch number")
tf.app.flags.DEFINE_integer("early_stop", 3,
                            "early-stopping")
tf.app.flags.DEFINE_string("model_path", "../dummy_model/dummy_cnn.ckpt",
                           "check point path, where the model is saved")
tf.app.flags.DEFINE_string("data_path", "../ba_train_data/train_data.pkl",
                           "pickle file to store the correct labeled examples")

tf.app.flags.DEFINE_string("dtype", "fp32",
                           "data type. \"fp16\", \"fp32\" or \"fp64\" only")

flags = tf.app.flags.FLAGS

if __name__ == "__main__":
    
    print ("[*] Hello world!", flush=True)
    
    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    
    # Set random seed
    tf.set_random_seed(flags.rand_seed)
    random.seed(flags.rand_seed)
    numpy.random.seed(flags.rand_seed)
    
    # Load dataset
    #############d = Fashion_MNIST()
    data_path = flags.data_path
    with open(data_path, "rb") as f:
        [image, label] = pickle.load(f)
    
    # Read hyper-parameters
    keep_prob_fc = flags.keep_prob_fc
    keep_prob_conv = flags.keep_prob_conv
    bs = flags.batch_size
    n_epoch = flags.n_epoch

    n_batch_train = int(image.shape[0] / bs)
    ##############n_batch_valid = int(d.valid.size / bs)
    ##############n_batch_test = int(d.test.size / bs)
    early_stopping_n = flags.early_stop
    model_path = flags.model_path
    if flags.dtype == "fp16":
        dtype = numpy.float16
    elif flags.dtype == "fp32":
        dtype = numpy.float32
    elif flags.dtype == "fp64":
        dtype = numpy.float64
    else:
        assert False, "Invalid data type (%s). Use \"fp16\", \"fp32\" or \"fp64\" only" % flags.dtype
    
    # Build model
    with tf.variable_scope("fmnist_cnn") as vs:
        m = CNN(scope_name="fmnist_cnn", is_inference=False, )
        print("[*] Model built!")
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        print("[*] Model initialized!")
        print("[*] Model trainable variables:")
        parm_cnt = 0
        variable = [v for v in tf.trainable_variables()]
        for v in variable:
            print("   ", v.name, v.get_shape())
            parm_cnt_v = 1
            for i in v.get_shape().as_list():
                parm_cnt_v *= i
            parm_cnt += parm_cnt_v
        print("[*] Model parameter size: %.4fM" %(parm_cnt/1024/1024))
        
        ######d.train.reset_epoch()
        #######d.valid.reset_epoch()
        #########d.test.reset_epoch()
        
        best_valid_acc = 0
        best_valid_loss = 1e10
        early_stopping_cnt = 0
        cur_acc = 0

        for epoch in range(12):
            
            if_test = False
            
            print ("[*] Epoch %d/%d, Training start..." % (epoch+1, n_epoch), flush=True)
            mean_train_acc = 0
            mean_train_loss = 0
            mean_train_time = 0
            for i in range(n_batch_train):
                iter_ = epoch * n_batch_train + i
                ##############x, y = d.train.next_batch(bs, dtype=dtype)
                x = image[i*bs:(i+1)*bs].reshape((bs, 28, 28, 1))
                y = label[i*bs:(i+1)*bs].reshape((bs, 10))


                begin_time = time.time()
                acc, loss = m.train_op(sess, x, y, iter_, keep_prob_fc, keep_prob_conv)
                end_time = time.time()
                mean_train_acc += acc / n_batch_train
                mean_train_loss += loss / n_batch_train
                mean_train_time += (end_time - begin_time) / n_batch_train
                if numpy.isnan(loss):
                    print("[*] NaN Stopping!", flush=True)
                    exit(-1)
                print("\t\repoch %d/%d, iteration %d/%d:\t loss = %.3e, acc = %.3f, time = %.3f" \
                          % (epoch+1, n_epoch, i+1, n_batch_train, loss, acc, end_time-begin_time),
                          flush=True, end="")
            print ("\n[*] Epoch %d/%d, Training done!\n\tloss = %.3e, acc = %.3f, time = %.3f" \
                   % (epoch+1, n_epoch, mean_train_loss, mean_train_acc, mean_train_time), flush=True)
                

            if mean_train_acc > cur_acc:
                cur_acc = mean_train_acc
                print ("[*] Best accuracy so far! ")
                m.save(sess, model_path)
                print ("[*] Model saved at", model_path, flush=True)
                
