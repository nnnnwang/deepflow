# -*- coding: utf-8 -*-
# @Time    : 2020/12/2
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import tensorflow as tf

class Utils:

    @staticmethod
    def load_from_ckpt(model, ckpt_path):
        warm_up_model_path = tf.train.latest_checkpoint(ckpt_path)
        if warm_up_model_path is not None:
            print("load model from {}".format(warm_up_model_path))
            model.load_weights(warm_up_model_path).expect_partial()
        else:
            raise FileExistsError("there is no valid ckpt file at {}".format(ckpt_path))