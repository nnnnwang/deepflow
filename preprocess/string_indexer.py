# -*- coding: utf-8 -*-
# @Time    : 2020/12/3
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from collections import OrderedDict

import pandas as pd

from input.input_cls import SparseFeature, SequenceFeature


class StringIndexer:

    def __init__(self, min_count=0, seq_length_map=None, pooling='sum'):

        self.min_count = min_count
        self.pooling = pooling
        if seq_length_map is not None:
            if not isinstance(seq_length_map, dict):
                raise ValueError('given seq_length_map must be dict type but got {}'.format(type(seq_length_map)))
            self.seq_length_map = seq_length_map
        else:
            self.seq_length_map = OrderedDict()
        self._indexer_dict = OrderedDict()
        self._feat_obj_lst = []


    @property
    def feature_obj_list(self):
        return self._feat_obj_lst

    @property
    def indexer_dict(self):
        return self._indexer_dict

    def fit(self, dataframe, feature_list):
        for feature in feature_list:
            print("**********fit {}*************".format(feature))
            if feature not in self.seq_length_map.keys():
                self.seq_length_map[feature] = dataframe[feature].apply(lambda x:len(x) if isinstance(x, (list, tuple)) else 1).max()
            feature_map_cnt = dataframe.explode(feature).drop_duplicates(subset=feature).groupby(feature)[feature].count()
            feature_map_cnt = feature_map_cnt[feature_map_cnt>=self.min_count]
            self._indexer_dict[feature] = dict([(v,k+2) for k,v in enumerate(feature_map_cnt.index)])
            if self.seq_length_map[feature] == 1:
                self.feature_obj_list.append(SparseFeature(feature,
                                                           len(self._indexer_dict[feature].keys()),
                                                           shape=(1, )))
            else:
                self.feature_obj_list.append(SequenceFeature(
                    feature, len(self._indexer_dict[feature].keys()),
                    shape=(self.seq_length_map[feature],), pooling=self.pooling
                ))

    def transform(self, dataframe):

        output_dict = dict()

        for feature in self._indexer_dict.keys():
            if feature not in dataframe.columns:
                print("***********feature:{} not in given dataframe***********".format(feature))
                continue
            print("***********transform {}**************".format(feature))
            def conversion(line, feature_map, seq_length):
                res = []
                for val in line:
                    res.append(feature_map.get(val, 1))
                return self._pad_seq(res, seq_length)

            output_dict[feature] = dataframe[feature].apply(lambda x: conversion(x, self._indexer_dict[feature], self.seq_length_map[feature])).apply(pd.Series).values

        return output_dict

    def fit_transform(self, dataframe, feature_list):
        self.fit(dataframe, feature_list)
        return self.transform(dataframe)

    def _pad_seq(self, seq_list, seq_length):
        if len(seq_list) < seq_length:
            seq_list = seq_list + [0]*(seq_length-len(seq_length))
            return seq_list
        elif len(seq_list) == seq_length:
            return seq_list
        else:
            return seq_list[:seq_length]

class SparseStringIndexer(StringIndexer):

    def fit(self):
        pass

class SequenceStringIndexer(StringIndexer):

    def fit(self):
        pass

class JointStringIndexer(StringIndexer):

    def fit(self):
        pass