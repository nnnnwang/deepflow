# -*- coding: utf-8 -*-
# @Time    : 2020/12/7
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import tensorflow as tf
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.initializers import glorot_normal,glorot_uniform
from tensorflow.keras.initializers import RandomNormal,random_uniform
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.initializers import he_normal,he_uniform
from tensorflow.keras.initializers import lecun_normal,lecun_uniform
from tensorflow.keras.initializers import Zeros,Ones

def get_optimizer(opt_name, learning_rate=None):
    if opt_name == 'sgd':
        optimizer = optimizers.SGD(0.01 if learning_rate is None else learning_rate, momentum=0.9, nesterov=True)
    elif opt_name == 'rmsprop':
        optimizer = optimizers.RMSprop(0.001 if learning_rate is None else learning_rate)
    elif opt_name == 'adagrad':
        optimizer = optimizers.Adagrad(0.01 if learning_rate is None else learning_rate)
    elif opt_name == 'adam':
        optimizer = optimizers.Adam(0.001 if learning_rate is None else learning_rate, amsgrad=True)

    return optimizer

def get_eval_metric(metrics):
    metrics_list = []
    for metric in metrics:
        if metric == 'acc':
            metrics_list.append(metric)
        elif metric == 'auc':
            metrics_list.append(tf.keras.metrics.AUC(num_thresholds=500), name='auc')

    return metrics_list


def get_initializer(init_name='truncate_norm', init_stddev=0.05, seed=1024):
    if init_name in ('truncate_norm', 'truncate_normal'):
        return TruncatedNormal(stddev=init_stddev, seed=seed)
    elif init_name in ('glorot_norm', 'glorot_normal', 'xavier_norm', 'xavier_normal'):
        return glorot_normal(seed=seed)
    elif init_name in ('he_norm', 'he_normal'):
        return he_normal(seed)


def get_regularizer(l1_reg:0.0, l2_reg:0.0):
    if l1_reg is None:
        l1_reg = 0.0
    if l2_reg is None:
        l2_reg = 0.0
    return regularizers.l1_l2(l1=l1_reg, l2=l2_reg)