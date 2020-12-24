# -*- coding: utf-8 -*-
# @Time    : 2020/12/2
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from collections import Iterable

from tensorflow.python.keras.layers import Input

from input.input_cls import Feature

def build_model_input(feature_cls):

    if feature_cls is None:
        raise ValueError('feature_cls should not be None')

    if isinstance(feature_cls, Feature):
        return {feature_cls.name:Input(shape=feature_cls.shape, name=feature_cls.name, dtype=feature_cls.dtype)}

    if isinstance(feature_cls, Iterable):
        if not all(map(lambda x:isinstance(x, Feature), feature_cls)):
            raise ValueError('given feature object should all be Feature type')

        return {feature.name:Input(shape=feature.shape, name=feature.name, dtype=feature.dtype) for feature in feature_cls}

    raise ValueError('the given feature cls is not a Feature class and cannot be iterated')
