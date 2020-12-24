# -*- coding: utf-8 -*-
# @Time    : 2020/12/2
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

class Feature:

    def __init__(self, *args, **kwargs):
        pass

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

class DenseFeature(Feature):

    def __init__(self, name, shape=(1, ), dtype='float32'):
        self._name = name
        self._shape = shape
        self._dtype = dtype

class SparseFeature(Feature):

    def __init__(self, name, max_index, shape=(1, ), embedding_name=None, dtype='int64', encode_name=None):
        self._name = name
        self._max_index = max_index
        if embedding_name is None:
            self._embedding_name = self._name + '_embd'
        else:
            self._embedding_name = embedding_name
        self._shape = shape
        self._dtype = dtype
        self._encode_name = encode_name

    @property
    def max_index(self):
        return self._max_index

    @property
    def embedding_name(self):
        return self._embedding_name

    @property
    def encode_name(self):
        return self._encode_name

class SequenceFeature(Feature):

    def __init__(self, name, max_index, shape, embedding_name=None, pooling='sum', dtype='int64', query=None, encode_name=None):
        self._name = name
        self._max_index = max_index
        if embedding_name is None:
            self._embedding_name = self._name + '_embd'
        else:
            self._embedding_name = embedding_name
        self._shape = shape
        self._pooling = pooling
        self._dtype = dtype
        self._query = query
        self._encode_name = encode_name

        if self._pooling in ('inner', 'mlp') and query is None:
            raise ValueError(" when pooling of SequenceFeature is 'inner' or 'mlp', the query of current feature should not be None")

    @property
    def max_index(self):
        return self._max_index

    @property
    def embedding_name(self):
        return self._embedding_name

    @property
    def pooling(self):
        return self._pooling

    @pooling.setter
    def pooling(self, pooling_type):
        self._pooling = pooling_type

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, query):
        self._query = query

    @property
    def encode_name(self):
        return self._encode_name