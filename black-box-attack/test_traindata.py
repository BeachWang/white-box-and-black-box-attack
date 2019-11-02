#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy
import random
import time
import pickle
import os, sys
from fmnist_dataset import Fashion_MNIST
from model import CNN

tf.app.flags.DEFINE_integer("rand_seed", 2019,
                            "seed for random number generaters")
tf.app.flags.DEFINE_string("gpu", "0",
                           "select one gpu")

tf.app.flags.DEFINE_string("ba_train_path", "../ba_train_data/train_data.pkl",
                           "pickle file to store the correct labeled examples")
tf.app.flags.DEFINE_string("model_path", "../model/fmnist_cnn.ckpt",
                           "check point path, where the model is saved")

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
    d = Fashion_MNIST()
    
    # Read hyper-parameters
    ba_train_path = flags.ba_train_path
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
        m = CNN(scope_name="fmnist_cnn", is_inference=True)
        print("[*] Model built!")
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        
        m.restore(sess, model_path)
        
        print("[*] Model loaded!")
        print("[*] Model parameters:")
        parm_cnt = 0
        variable = [v for v in tf.trainable_variables()]
        for v in variable:
            print("   ", v.name, v.get_shape())
            parm_cnt_v = 1
            for i in v.get_shape().as_list():
                parm_cnt_v *= i
            parm_cnt += parm_cnt_v
        print("[*] Model parameter size: %.4fM" %(parm_cnt/1024/1024))

        d.train.reset_epoch()
        
        acc = 0
        image = []
        label = []
        for _iter in range(d.train.size):
            x, y = d.train.next_batch(1, dtype=dtype)
            y_hat = m.infer_op(sess, x)
            image.append(x[0])
            label.append(y_hat)
            if (numpy.argmax(y_hat) == numpy.argmax(y[0])):
                acc += 1
        acc /= d.train.size
        print("[*] Accuracy on test set: %.5f" % (acc))
        
        image = numpy.asarray(image, dtype=dtype).reshape((-1, 1, 28, 28))
        label = numpy.asarray(label, dtype=dtype).reshape((-1, 1, 10))
        with open(ba_train_path, "wb") as f:
            pickle.dump([image, label], f)
            
