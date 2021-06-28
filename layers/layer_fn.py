# -*- coding: utf-8 -*-
# @Time    : 2020/12/7
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

from collections import OrderedDict

from tensorflow.python.keras.layers import Embedding,Dense,Add,Concatenate,Reshape

from input.input_cls import SparseFeature,SequenceFeature
from layers.init_fn import get_initializer,get_regularizer
from layers.core import Pooling,AttentionPooling

def get_embd_matrix_dict(sparse_feat_cls_lst,
                         embedding_init_name='truncate_norm',
                         embedding_init_stddev=0.05,
                         embedding_size=16,
                         embedding_l1_reg=0.0,
                         embedding_l2_reg=0.0,
                         mask_zero=False,
                         embedding_init_fn=None,
                         suffix_name='default'):
    """
    get the embedding matrix from given sparse feature class list
    Args:
        sparse_feat_cls_lst:
        embedding_init_name:
        embedding_init_stddev:
        embedding_size:
        embedding_l1_reg:
        embedding_l2_reg:
        mask_zero:
        embedding_init_fn:
        suffix_name:

    Returns:

    """

    _check_sparse_feature(sparse_feat_cls_lst)

    embed_mat_dict = OrderedDict()
    exists_embed_set = set()
    embed_init = embedding_init_fn if embedding_init_fn is not None else get_initializer(embedding_init_name, embedding_init_stddev)
    embed_reg = get_regularizer(embedding_l1_reg, embedding_l2_reg)

    for feature in sparse_feat_cls_lst:
        if feature.embedding_name in exists_embed_set:
            continue
        embed_mat_dict[feature.embedding_name] = Embedding(input_dim=feature.max_index+2,
                                                           output_dim=embedding_size,
                                                           embeddings_initializer=embed_init,
                                                           embeddings_regularizer=embed_reg,
                                                           mask_zero=mask_zero if isinstance(feature, SequenceFeature) else False,
                                                           name='_'.join([feature.embedding_name, suffix_name]))
        exists_embed_set.add(feature.embedding_name)

    return embed_mat_dict

def get_raw_embed_dict(input_layer_dict,
                       sparse_feat_cls_list,
                       embedding_init_name='truncate_norm',
                       embedding_init_stddev=0.05,
                       embedding_size=16,
                       embedding_l1_reg=0.0,
                       embedding_l2_reg=0.0,
                       mask_zero=False,
                       embedding_init_fn=None,
                       suffix_name='default',
                       embedding_matrix=None):
    """

    Args:
        input_layer_dict:
        sparse_feat_cls_list:
        embedding_init_name:
        embedding_init_stddev:
        embedding_size:
        embedding_l1_reg:
        embedding_l2_reg:
        mask_zero:
        embedding_init_fn:
        suffix_name:
        embedding_matrix:

    Returns:

    """

    _check_sparse_feature(sparse_feat_cls_list)
    if embedding_matrix is None:
        sparse_embed_matrix = get_embd_matrix_dict(
            sparse_feat_cls_lst=sparse_feat_cls_list,
            embedding_init_name=embedding_init_name,
            embedding_init_stddev=embedding_init_stddev,
            embedding_size=embedding_size,
            embedding_l1_reg=embedding_l1_reg,
            embedding_l2_reg=embedding_l2_reg,
            mask_zero=mask_zero,
            suffix_name=suffix_name,
            embedding_init_fn=embedding_init_fn
        )
    else:
        sparse_embed_matrix = embedding_matrix

    sparse_input_dict = OrderedDict()
    for feature in sparse_feat_cls_list:
        sparse_input_dict[feature.name] = sparse_embed_matrix[feature.embedding_name](input_layer_dict[feature.name])

    return sparse_input_dict

def get_pooling_embed_dict(input_layer_dict,
                           sparse_feat_cls_list,
                           embedding_init_name='truncate_normal',
                           embedding_init_stddev=0.05,
                           embedding_size=16,
                           embedding_l1_reg=0.0,
                           embedding_l2_reg=0.0,
                           mask_zero=False,
                           pooling=None,
                           embedding_matrix=None,
                           suffix_name='default'):

    raw_embedding_dict = get_raw_embed_dict(
        input_layer_dict=input_layer_dict,
        sparse_feat_cls_list=sparse_feat_cls_list,
        embedding_init_name=embedding_init_name,
        embedding_init_stddev=embedding_init_stddev,
        embedding_size=embedding_size,
        embedding_l1_reg=embedding_l1_reg,
        embedding_l2_reg=embedding_l2_reg,
        mask_zero=mask_zero,
        suffix_name=suffix_name,
        embedding_matrix=embedding_matrix
    )

    pooling_embedding = []

    for index, feature_cls in enumerate(sparse_feat_cls_list):
        # only process the given sparse feature
        if feature_cls.name not in raw_embedding_dict.keys():
            continue
        if isinstance(feature_cls, SequenceFeature):
            if feature_cls.pooling in ('mean', 'sum', 'max', 'min'):
                if pooling is None:
                    pooling_output = Pooling(feature_cls.pooling)(raw_embedding_dict[feature_cls.name])
                else:
                    pooling_output = Pooling(pooling)(raw_embedding_dict[feature_cls.name])
            elif feature_cls.pooling == 'inner':
                query = feature_cls.query
                pooling_output = AttentionPooling()([raw_embedding_dict[query], raw_embedding_dict[feature_cls.name]])
            elif feature_cls.pooling == 'mlp':
                query = feature_cls.query
                pooling_output = AttentionPooling(attention_type='mlp')([raw_embedding_dict[query], raw_embedding_dict[feature_cls.name]])
        else:
            pooling_output = Pooling('sum')(raw_embedding_dict[feature_cls.name])
        pooling_embedding.append(Reshape((1, int(pooling_output.shape[-1])))(pooling_output))

    feature_name = list(raw_embedding_dict.keys())

    return OrderedDict(zip(feature_name, pooling_embedding))

def get_embed_output(input_layer_dict,
                     sparse_feat_cls_list,
                     embedding_init_name='truncate_normal',
                     embedding_init_stddev=0.05,
                     embedding_size=16,
                     embedding_l1_reg=0.0,
                     embedding_l2_reg=0.0,
                     mask_zero=False,
                     pooling=None,
                     embedding_matrix=None,
                     suffix_name='default'):

    embedding_pooling_dict = get_pooling_embed_dict(
        input_layer_dict=input_layer_dict,
        sparse_feat_cls_list=sparse_feat_cls_list,
        embedding_init_name=embedding_init_name,
        embedding_init_stddev=embedding_init_stddev,
        embedding_size=embedding_size,
        embedding_l1_reg=embedding_l1_reg,
        embedding_l2_reg=embedding_l2_reg,
        mask_zero=mask_zero,
        pooling=pooling,
        suffix_name=suffix_name,
        embedding_matrix=embedding_matrix
    )

    pooling_value = list(embedding_pooling_dict.values())

    if len(sparse_feat_cls_list) == 1:
        return pooling_value[0]
    else:
        return Concatenate(axis=1)(pooling_value)

def get_linear_output(input_layer_dict,
                      dense_feat_cls_list=[],
                      sparse_feat_cls_list=[],
                      embedding_init_name='truncate_normmal',
                      embedding_init_stddev=0.05,
                      l1_reg=0.0,
                      l2_reg=0.0,
                      mask_zero=False,
                      suffix_name='wide'):

    if len(dense_feat_cls_list) == 0 and len(sparse_feat_cls_list) == 0:
        raise ValueError('both dense feature and sparse feature are none, please check input')

    linear_out = None

    if len(dense_feat_cls_list) > 0:
        if len(dense_feat_cls_list) == 1:
            linear_out = Dense(1, kernel_regularizer=get_regularizer(l1_reg=l1_reg, l2_reg=l2_reg))(input_layer_dict[dense_feat_cls_list[0].name])
        else:
            linear_out = Dense(1, kernel_regularizer=get_regularizer(l1_reg=l1_reg, l2_reg=l2_reg))(
                Concatenate()([input_layer_dict[feature.name] for feature in dense_feat_cls_list]))

    if len(sparse_feat_cls_list) > 0:
        def change_pooling(feat_cls):
            feat_cls.pooling = 'sum'
            return feat_cls
        sum_pooling_feat_cls = list(map(change_pooling, sparse_feat_cls_list))
        sparse_linear_output = get_embed_output(
            input_layer_dict=input_layer_dict,
            sparse_feat_cls_list=sum_pooling_feat_cls,
            embedding_init_name=embedding_init_name,
            embedding_init_stddev=embedding_init_stddev,
            embedding_size=1,
            embedding_l2_reg=l2_reg,
            embedding_l1_reg=l1_reg,
            mask_zero=mask_zero,
            suffix_name=suffix_name
        )
        sparse_linear_output = Pooling('sum')(sparse_linear_output)
        if linear_out is None:
            linear_out = sparse_linear_output
        else:
            linear_out = Add()([linear_out, sparse_linear_output])

    return linear_out

def _check_sparse_feature(sparse_feat_cls_list):
    if not all(map(lambda x:isinstance(x, (SparseFeature, SequenceFeature)), sparse_feat_cls_list)):
        raise ValueError('given feature in sparse_feat_cls_list should all be SparseFeature type or SequenceFeature type')
