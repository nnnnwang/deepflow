# -*- coding: utf-8 -*-
# @Time    : 2020/12/5
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from preprocess.normalization import Normalization
from preprocess.string_indexer import StringIndexer

class FeatureProcess:

    def feature_process(self,
                        dataframe,
                        dense_feature=None,
                        sparse_feature=None,
                        standard_type='stand',
                        min_count=0,
                        seq_length_map=None,
                        pooling='sum'):

        if dense_feature is None and sparse_feature is None:
            raise ValueError('both dense_feature and sparse_feature are None, please check')

        if dense_feature is not None:
            if not isinstance(dense_feature, list):
                raise TypeError('dense_feature should be list but got {}'.format(type(dense_feature)))

        if sparse_feature is not None:
            if not isinstance(sparse_feature, list):
                raise TypeError('sparse_feature should be list but got {}'.format(type(sparse_feature)))

        self.indexer = StringIndexer(min_count=min_count, seq_length_map=seq_length_map, pooling=pooling)
        self.stand = Normalization(norm_type=standard_type)
        input_data = {}
        feature_cls = []

        if dense_feature is not None:
            self.stand.fit(dataframe, dense_feature)
            dense_feature_cls = self.stand.feature_obj_list
            input_data.update(self.stand.transform(dataframe))
            feature_cls += dense_feature_cls

        if sparse_feature is not None:
            self.indexer.fit(dataframe, sparse_feature)
            sparse_feature_cls = self.indexer.feature_obj_list
            input_data.update(self.indexer.transform(dataframe))
            feature_cls += sparse_feature_cls

        return feature_cls, input_data

