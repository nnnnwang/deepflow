# -*- coding: utf-8 -*-
# @Time    : 2020/12/2
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from collections import OrderedDict,Iterable

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from input.input_cls import DenseFeature

class Normalization:

    def __init__(self, norm_type='stand'):
        self.norm_type = norm_type
        self.norm_model_dict = OrderedDict()
        self.feature_length_dict = OrderedDict()
        self._feature_obj_lst = []

    @property
    def feature_obj_list(self):
        return self._feature_obj_lst

    def fit(self, dataframe, feature_list):

        for feature_name in feature_list:
            print("***********fit {}**************".format(feature_name))
            norm_model = self._get_norm_model()
            feature_dims = self._get_feature_max_len(dataframe, feature_name)
            self.feature_length_dict[feature_name] = feature_dims
            norm_model.fit(self._get_feature_value(dataframe, feature_name))
            self.norm_model_dict[feature_name] = norm_model
            self._feature_obj_lst.append(DenseFeature(feature_name, shape=(feature_dims,)))

    def transform(self, dataframe):
        output_dict = OrderedDict()
        for feature, norm_model in self.norm_model_dict.items():
            if feature not in dataframe.columns:
                print('feature:{} not in given dataframe\'s columns'.format(feature))
                continue

            print("***********transform {}**************".format(feature))
            output_dict[feature] = norm_model.transform(self._get_feature_value(dataframe, feature)).astype('float32')
        return output_dict

    def fit_transform(self, dataframe, feature_list):
        self.fit(dataframe, feature_list)
        return self.transform(dataframe)

    def get_feature_length(self, feature_name):
        if feature_name not in self.feature_length_dict.keys():
            raise ValueError('feature:{} not in the current object')
        else:
            return self.feature_length_dict[feature_name]

    def _get_feature_max_len(self, dataframe, feature):
        return len(dataframe[feature][0]) if isinstance(dataframe[feature][0], Iterable) else 1

    def _get_norm_model(self):
        if str.lower(self.norm_type) in ('stand', 'standard'):
            return StandardScaler()
        elif str.lower(self.norm) in ('minmax'):
            return MinMaxScaler()
        else:
            raise ValueError('not support {} norm type yet'.format(self.norm_type))

    def _get_feature_value(self, dataframe, feature_name):
        return dataframe[feature_name].values.reshape((-1, self.feature_length_dict[feature_name]))

    def _check_feature(self, dense_feat_cls_list):
        pass

