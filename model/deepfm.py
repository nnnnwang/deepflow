# -*- coding: utf-8 -*-
# @Time    : 2020/12/9
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from tensorflow.keras import layers
from tensorflow.keras import Model

from layers import layer_fn, FM
from model.basemodel import BaseModel

class DeepFM(BaseModel):

    def __init__(self,
                 feature_cls_list,
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

        super(DeepFM, self).__init__(feature_cls_list)

        self.model_name = self.set_param('model_name', 'deepfm')
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

        super(DeepFM, self).build()

        embedding_out = self._get_sparse_embedding()

        dnn_in = layers.Flatten()(embedding_out)

        dnn_in = self._add_dense_input(dnn_in, self.dense_feat_cls_list)

        dnn_out = self._get_mlp_out(dnn_in)

        dnn_out = layers.Dense(1)(dnn_out)

        # fm part
        fm_out = FM()(embedding_out)

        # linear part
        linear_out = layer_fn.get_linear_output(
            self.input_dict,
            dense_feat_cls_list=self.dense_feat_cls_list,
            sparse_feat_cls_list=self.sparse_feat_cls_list,
            embedding_init_name=self.embedding_init_name,
            embedding_init_stddev=self.embedding_init_stddev,
            l1_reg=self.wide_l1_reg,
            l2_reg=self.wide_l2_reg,
            mask_zero=self.mask_zero
        )

        dnn_logit = layers.Add()([ linear_out, fm_out, dnn_out ])
        y_hat = self._predict_layer(dnn_logit)

        self.model = Model(inputs=self.input_dict, outputs=[y_hat], name='deepfm')
