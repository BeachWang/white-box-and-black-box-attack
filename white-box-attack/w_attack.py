# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import random
import time
import pickle
import os, sys
from fmnist_dataset import Fashion_MNIST
from model import CNN

import matplotlib.pyplot as plt

tf.app.flags.DEFINE_integer("rand_seed", 2019,
                            "seed for random number generaters")
tf.app.flags.DEFINE_string("gpu", "0",
                           "select one gpu")

tf.app.flags.DEFINE_integer("n_correct", 1000,
                            "correct example number")
tf.app.flags.DEFINE_string("correct_path", "../attack_data/w_correct_3k.pkl",
                           "pickle file to store the correct labeled examples")
tf.app.flags.DEFINE_string("model_path", "../w_reattack_model/fmnist_cnn.ckpt",
                           "check point path, where the model is saved")
tf.app.flags.DEFINE_string("reattack_data_path", "../attack_data/w_useless_1k.pkl",
                           "pickle file to store the correct labeled examples")

tf.app.flags.DEFINE_string("dtype", "fp32",
                           "data type. \"fp16\", \"fp32\" or \"fp64\" only")

flags = tf.app.flags.FLAGS

if __name__ == "__main__":
    
    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu
    
    # Set random seed
    tf.set_random_seed(flags.rand_seed)
    random.seed(flags.rand_seed)
    np.random.seed(flags.rand_seed)
    
    # Load dataset
    #d = Fashion_MNIST()
    
    # Read hyper-parameters
    n_correct = flags.n_correct
    correct_path = flags.correct_path
    model_path = flags.model_path
    reattack_data_path = flags.reattack_data_path
    if flags.dtype == "fp16":
        dtype = np.float16
    elif flags.dtype == "fp32":
        dtype = np.float32
    elif flags.dtype == "fp64":
        dtype = np.float64
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

        #d.test.reset_epoch()
        
        #x, y = d.test.next_batch(1, dtype=dtype)

        # print(x[0].shape) bx28x28x1的numpy
        # print(y[0].shape) bx10的numpy

        with open(correct_path, "rb") as f:
            [correct_image, correct_label] = pickle.load(f)

        # print(correct_image.shape) 1000x1x28x28的numpy
        # print(correct_label.shape) 1000x1x10的numpy

        eps = 10
        clip_min = 0.
        clip_max = 255.
        x_succ, x_old, labels = ([], [], [])
        cnt = 0.
        for_reattack_data, for_reattack_label = ([],[])

        for i in range(0,1000):
            x = correct_image[i].reshape((1, 28, 28, 1))
            tem = x
            y = correct_label[i].reshape((1, 10))
            y1 = y
            for_reattack_label.append(y1)
            ind = np.argmax(y)
            y[0][int(ind)] = 0
            y[0][(int(ind)+1)%10] = 1


            for j in range(0,1000):
                dx = m.grad_op(sess, x, y)
                dx = np.asarray(dx).reshape((1, 28, 28, 1))

                x = x - dx*eps
                x = np.clip(x, clip_min, clip_max)

            for_reattack_data.append(x)
            y_logits = m.infer_op(sess, x)
            y_logits = np.asarray(y_logits).reshape((1, 10))
            ind_logits = np.argmax(y_logits)
            if ind_logits == (ind+1)%10:
                cnt = cnt + 1
                x_succ.append(x)
                x_old.append(tem)
                labels.append(ind)

            print ("\r round %.3f" % (i+1), end=" ")

        acc = cnt/1000.0
        print('attack accuracy = %.3f' % acc)

        pic_id = random.sample(range(0, len(x_succ)), 10)
        fig, ax = plt.subplots(nrows=2, ncols=10)
        ax = ax.flatten()

        t = 0
        for i in pic_id:
            img = np.reshape(x_old[i], [28, 28])
            ax[t].imshow(img)
            ax[t].set_xticks([])
            ax[t].set_yticks([])
            ax[t].set_title(labels[i])
            img2 = np.reshape(x_succ[i], [28, 28])
            ax[t + 10].imshow(img2)
            ax[t + 10].set_xticks([])
            ax[t + 10].set_yticks([])
            ax[t + 10].set_title((labels[i]+1)%10)
            t = t + 1

        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

        with open(reattack_data_path, "wb") as f:
            pickle.dump([for_reattack_data, for_reattack_label], f)



            
