# -*- coding: utf-8 -*-
# @Time    : 2020/12/2
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import os
from abc import abstractmethod
from collections import Iterable, OrderedDict

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate

from common.utils import Utils
from input.input_cls import Feature, DenseFeature, SparseFeature, SequenceFeature
from input.input_fn import build_model_input


class BaseModel:

    def __init__(self, feat_cls_list, **kwargs):

        self.__check_feat_cls(feat_cls_list)

        self.feat_cls_list = feat_cls_list
        self.dense_feat_cls_list = [_ for _ in self.feat_cls_list if isinstance(_, DenseFeature)]
        self.sparse_feat_cls_list = [_ for _ in self.feat_cls_list if isinstance(_, (SparseFeature, SequenceFeature))]
        self.input_dict = build_model_input(self.feat_cls_list)
        self._check_feature()

        self.hyper_parameter = OrderedDict()

    def build(self, training=True):

        # K.set_learning_phase(training)

        if self.deep_l2_reg_list is not None and len(self.deep_l2_reg_list) != len(self.hidden_units):
            raise ValueError(
                'got an unexpected length({}) for deep_l2_reg_list, it should be {}(the same as hidden_units)'.format(
                    len(self.deep_l2_reg_list), len(self.hidden_units)))

        if self.dropout_list is not None and len(self.dropout_list) != len(self.hidden_units):
            raise ValueError(
                'got an unexpected length({}) for dropout_list, it should be {}(the same as hidden_units)'.format(
                    len(self.dropout_list), len(self.hidden_units)))

        if self.model_type not in ('bi_classifier', 'regression'):
            raise ValueError(
                'only support bi_classifier and regression model_type now but got {}'.format(self.model_type)
            )

    def desc(self):
        """
        describe the model structure, input feature and so on

        Returns:

        """
        print("*"*100)
        print("dense feature:{}".format('.'.join([_.name for _ in self.dense_feat_cls_list])))
        print("sparse feature:{}".format('.'.join([_.name for _ in self.sparse_feat_cls_list])))

        feature_dim = 0
        feature_cnt = len(self.dense_feat_cls_list + self.sparse_feat_cls_list)
        for feature in self.dense_feat_cls_list + self.sparse_feat_cls_list:
            if isinstance(feature, (SparseFeature, DenseFeature)):
                feature_dim += feature.shape[0]
            else:
                feature_dim += feature.max_index + 2

        print("there are {} features have been selected, and feature dimension is {}".format(feature_cnt, feature_dim))

        for k, v in self.hyper_parameter.items():
            print("{}:{}".format(k, v))
        print("*" * 100)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        self.model.fit(x,y,batch_size,epochs,verbose,callbacks,validation_split,
                       validation_data,shuffle,class_weight,sample_weight,initial_epoch,steps_per_epoch,validation_steps,
                       validation_batch_size,validation_freq,max_queue_size,workers,use_multiprocessing)

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):

        self.model.compile(optimizer,
                loss,
                metrics,
                loss_weights,
                weighted_metrics,
                run_eagerly,
                **kwargs)

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):

        return self.model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 return_dict=False):

        return self.model.evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict)

    def __check_feat_cls(self, feat_cls_lst):

        if feat_cls_lst is None:
            raise ValueError("feat_cls_list should not be None")
        if not isinstance(feat_cls_lst, Iterable):
            raise ValueError("feat_cls_list should be iterable")

        for cls in feat_cls_lst:
            if not isinstance(cls, Feature):
                raise ValueError("all feature cls in feat_cls_list should be Feature type but got".format(type(cls)))

    def _check_feature(self):

        feature_set = set()
        if len(self.dense_feat_cls_list + self.sparse_feat_cls_list) == 0:
            raise ValueError()
        for i in self.dense_feat_cls_list + self.sparse_feat_cls_list:
            if i.name not in feature_set:
                feature_set.add(i.name)
            else:
                raise ValueError("feature:{} has been found in serval feature class list, please check".format(i))

    def _load_today_ckpt(self, ckpt_path):

        Utils.load_from_ckpt(self.model, ckpt_path)

    def set_param(self, key, value):
        self.hyper_parameter[key] = value
        return value

    def add_dense_input(self, dnn_in, dense_feature_cls_lst):

        if len(dense_feature_cls_lst) > 0:
            if len(dense_feature_cls_lst) > 1:
                dense_out = Concatenate()([self.input_dict[feat.name] for feat in dense_feature_cls_lst])
            else:
                dense_out = self.input_dict[dense_feature_cls_lst[0].name]

            if dnn_in is None:
                dnn_in = dense_out
            else:
                dnn_in = Concatenate()([dnn_in, dense_out])

        return dnn_in