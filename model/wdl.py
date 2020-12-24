# -*- coding: utf-8 -*-
# @Time    : 2020/12/9
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from tensorflow.keras import layers
from tensorflow.keras import Model

from layers import layer_fn
from layers import Mlp
from model.basemodel import BaseModel

class WDL(BaseModel):

    def __init__(self,
                 feature_cls_list,
                 wide_feature_cls_list=None,
                 model_type='bi_classifier',
                 hidden_units=[64,64],
                 wide_l1_reg=0.0,
                 wide_l2_reg=0.0,
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

        super(WDL, self).__init__(feature_cls_list)

        if wide_feature_cls_list is None or len(wide_feature_cls_list) == 0:
            self.wide_feature_cls_list = self.sparse_feat_cls_list
        else:
            self.wide_feature_cls_list = wide_feature_cls_list

        self.model_name = self.set_param('model_name', 'wdl')
        self.model_type = self.set_param('model_type', model_type)
        self.hidden_units = self.set_param('hidden_units', hidden_units)
        self.wide_l1_reg = self.set_param('wide_l1_reg', wide_l1_reg)
        self.wide_l2_reg = self.set_param('wide_l2_reg', wide_l2_reg)
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

        super(WDL, self).build()

        embedding_out = layer_fn.get_embed_output(
            self.input_dict,
            sparse_feat_cls_list=self.sparse_feat_cls_list,
            embedding_init_name=self.embedding_init_name,
            embedding_init_stddev=self.embedding_init_stddev,
            embedding_l1_reg=self.embedding_l1_reg,
            embedding_l2_reg=self.embedding_l2_reg,
            mask_zero=self.mask_zero,
            suffix_name='deep'
        )
        dnn_in = layers.Flatten()(embedding_out)
        dnn_in = self.add_dense_input(dnn_in, self.dense_feat_cls_list)

        dnn_out = Mlp(
            units=self.hidden_units,
            dropout_list=self.dropout_list,
            activation=self.activation,
            l2_reg_list=self.deep_l2_reg_list,
            use_bn=self.use_bn,
            kernel_initializer=self.fc_init
        )(dnn_in)

        dnn_out = layers.Dense(1)(dnn_out)

        # linear part
        linear_out = layer_fn.get_linear_output(
            self.input_dict,
            dense_feat_cls_list=self.dense_feat_cls_list,
            sparse_feat_cls_list=self.wide_feature_cls_list,
            embedding_init_name=self.embedding_init_name,
            embedding_init_stddev=self.embedding_init_stddev,
            l1_reg=self.wide_l1_reg,
            l2_reg=self.wide_l2_reg,
            mask_zero=self.mask_zero
        )

        dnn_logit = layers.Add()([ 0.5*linear_out, 0.5*dnn_out ])
        if self.model_type == 'bi_classifier':
            y_hat = layers.Activation('sigmoid')(dnn_logit)
        elif self.model_type == 'regression':
            y_hat = dnn_logit

        self.model = Model(inputs=self.input_dict, outputs=[y_hat], name='wdl')
