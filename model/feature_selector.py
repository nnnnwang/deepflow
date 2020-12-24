# -*- coding: utf-8 -*-
# @Time    : 2020/12/9
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from layers import layer_fn
from layers import Mlp
from model.basemodel import BaseModel

class FeatureSelector(BaseModel):


    def __init__(self,
                 feature_cls_list,
                 model_type='bi_classifier',
                 hidden_units=[64,64],
                 deep_l2_reg_list=None,
                 fc_init='he_normal',
                 embedding_init_name='truncate_normal',
                 embedding_init_stddev=0.05,
                 embedding_l1_reg=0.0,
                 embedding_l2_reg=0.0,
                 embedding_size=16,
                 dropout_list=None,
                 use_bn=False,
                 activation='relu',
                 mask_zero=True,
                 pooling='sum'):
        """
        Examples:

            fe = FeatureSelector()
            fe_callback = FeatureImportanceCallback(fe)
            fe.compile("mse", "acc")
            # print feature importance on every epoch
            fe.fit(x, y, callbacks=[fe_callback])

        Args:
            feature_cls_list:
            model_type:
            hidden_units:
            deep_l2_reg_list:
            fc_init:
            embedding_init_name:
            embedding_init_stddev:
            embedding_l1_reg:
            embedding_l2_reg:
            embedding_size:
            dropout_list:
            use_bn:
            activation:
            mask_zero:
            pooling:
        """
        super(FeatureSelector, self).__init__(feature_cls_list)

        self.model_name = self.set_param('model_name', 'feature_selector')
        self.model_type = self.set_param('model_type', model_type)
        self.hidden_units = self.set_param('hidden_units', hidden_units)
        self.deep_l2_reg_list = self.set_param('deep_l2_reg_list', deep_l2_reg_list)
        self.fc_init = self.set_param('fc_init', fc_init)
        self.embedding_init_name = self.set_param('embedding_init_name', embedding_init_name)
        self.embedding_init_stddev = self.set_param('embedding_init_stddev', embedding_init_stddev)
        self.embedding_l1_reg = self.set_param('embedding_l1_reg', embedding_l1_reg)
        self.embedding_l2_reg = self.set_param('embedding_l2_reg', embedding_l2_reg)
        self.embedding_size = self.set_param('embedding_size', embedding_size)
        self.dropout_list = self.set_param('dropout_list', dropout_list)
        self.use_bn = self.set_param('use_bn', use_bn)
        self.activation = self.set_param('activation', activation)
        self.mask_zero = self.set_param('mask_zero', mask_zero)
        self.pooling = self.set_param('pooling', pooling)

        self.build()

    def build(self):

        super(FeatureSelector, self).build()

        if len(self.sparse_feat_cls_list) > 0:
            sparse_out = layer_fn.get_embed_output(
                self.input_dict,
                sparse_feat_cls_list=self.sparse_feat_cls_list,
                embedding_init_name=self.embedding_init_name,
                embedding_init_stddev=self.embedding_init_stddev,
                embedding_l1_reg=self.embedding_l1_reg,
                embedding_l2_reg=self.embedding_l2_reg,
                mask_zero=self.mask_zero,
                suffix_name='feature_selector'
            )
        else:
            sparse_out = None

        if len(self.dense_feat_cls_list) > 0:
            if len(self.dense_feat_cls_list) > 1:
                dense_out = layers.Concatenate()([self.input_dict[feat.name] for feat in self.dense_feat_cls_list])
            else:
                dense_out = self.input_dict[self.dense_feat_cls_list[0].name]
        else:
            dense_out = None

        feature_len = len(self.dense_feat_cls_list) + len(self.sparse_feat_cls_list)

        feature_importance = layers.Activation('sigmoid')(Variable(feature_len, name='feature_importance')(sparse_out if sparse_out is not None else dense_out))

        if dense_out is None:
            sparse_gate = feature_importance
            dnn_in = layers.Flatten()(tf.multiply(sparse_out, sparse_gate))
        elif len(self.sparse_feat_cls_list) == 0:
            dense_gate = feature_importance
            dnn_in = tf.multiply(dense_out, dense_gate)
        else:
            dense_gate, sparse_gate = tf.split(feature_importance, (len(self.dense_feat_cls_list), len(self.sparse_feat_cls_list)))
            sparse_out = layers.Flatten()(tf.multiply(sparse_out, sparse_gate))
            dense_out = tf.multiply(dense_out, dense_gate)
            dnn_in = layers.Concatenate()([sparse_out, dense_out])

        deep_dnn_out = Mlp(
            units=self.hidden_units,
            dropout_list=self.dropout_list,
            activation=self.activation,
            l2_reg_list=self.deep_l2_reg_list,
            use_bn=self.use_bn,
            kernel_initializer=self.fc_init
        )(dnn_in)

        dnn_logit = layers.Dense(1, name='dense')(deep_dnn_out)
        if self.model_type == 'bi_classifier':
            y_hat = layers.Activation('sigmoid')(dnn_logit)
        elif self.model_type == 'regression':
            y_hat = dnn_logit

        self.model = Model(inputs=self.input_dict, outputs=[y_hat], name='feature_selector')

    def get_feature_name(self):
        return [i.name for i in self.dense_feat_cls_list + self.sparse_feat_cls_list]

    def get_feature_importance_cb(self):
        return FeatureImportanceCallback(self.get_feature_name())

class Variable(layers.Layer):

    def __init__(self, dims, **kwargs):
        self.dims = dims
        super(Variable, self).__init__(**kwargs)

    def build(self, input_shape):
        self.variable = self.add_weight(
            name='var',
            shape=(self.dims, 1),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(), trainable=True
        )

        super(Variable, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.variable

    def get_config(self):
        config = {
            'dims': self.dims
        }

        base_config = super(Variable, self).get_config()
        return dict(list(base_config) + list(config.items()))

class FeatureImportanceCallback(Callback):

    def __init__(self, feature_name_list):
        self.feature_name_list = feature_name_list

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        feature_importance = np.array(self.model.get_layer('feature_importance').get_weights()).reshape((-1, 1))
        feature_importance = feature_importance/np.sum(feature_importance)
        feature_pair = zip(self.feature_name_list, feature_importance)
        feature_pair = sorted(feature_pair, key=lambda x:x[1], reverse=True)

        for k,v in feature_pair:
            print("{}:{}".format(k, v))