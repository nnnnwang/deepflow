# -*- coding: utf-8 -*-
# @Time    : 2020/12/7
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dropout,BatchNormalization,LayerNormalization,Dense
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from layers.init_fn import get_initializer

class Mlp(Layer):

    def __init__(self,
                 units,
                 dropout_list:list = None,
                 activation=None,
                 kernel_initializer='glorot_norm',
                 bias_initializers='zeros',
                 l2_reg_list=None,
                 use_bn=False,
                 use_gate=False,
                 **kwargs):

        self.units = list(map(int, units))
        self.dropout_list = dropout_list
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(get_initializer(kernel_initializer))
        self.bias_initializer = initializers.get(bias_initializers)
        self.l2_reg_list = l2_reg_list
        self.use_bn = use_bn
        self.use_gate = use_gate

        super(Mlp, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(self.units, list):
            raise TypeError('units must be list type')

        if self.dropout_list is not None:
            if not isinstance(self.dropout_list, list):
                raise TypeError('dropout_list must be list type but got {}'.format(type(self.dropout_list)))

            if len(self.dropout_list) != len(self.units):
                raise ValueError('the length of dropout_list must be equal to the length of units')

        last_dim = int(input_shape[-1])
        all_units = [last_dim] + self.units

        self.kernel = [
            self.add_weight(
                'kernel_' + str(index),
                shape=[all_units[index], all_units[index+1]],
                initializer=self.kernel_initializer,
                regularizer=regularizers.l2(0.0) if self.l2_reg_list is None else regularizers.l2(self.l2_reg_list[index]),
                dtype=K.floatx(),
                trainable=True)
            for index in range(len(all_units) - 1)
        ]

        if not self.use_bn:
            self.bias = [
                self.add_weight(
                    'bias_'+str(index),
                    shape=[all_units[index+1],],
                    initializer=self.bias_initializer,
                    dtype=K.floatx(),
                    trainable=True
                ) for index in range(len(all_units) - 1)
            ]
        else:
            self.bn = [
                BatchNormalization()
                for _ in range(len(all_units))
            ]
        if self.use_gate:
            self.gate = [
                Gate('bit_wise')
                for _ in range(len(all_units))
            ]
        if self.dropout_list is not None:
            self.dropout = [
                Dropout(self.dropout_list[index])
                for index in range(len(self.dropout_list))
            ]
        super(Mlp, self).__init__(input_shape)

    def call(self, inputs, training=None, **kwargs):

        input = inputs

        for index in range(len(self.units)):
            if self.dropout_list is not None:
                output = self.dropout[index](input, training=training)
            else:
                output = input
            output = tf.matmul(output, self.kernel[index])
            if not self.use_bn:
                output = tf.nn.bias_add(output, self.bias[index])
            if self.use_bn:
                output = self.bn[index](output, training=training)
            if self.activation is not None:
                output = self.activation(output)
            if self.use_gate:
                output = self.gate[index](output)

            input = output
        return output

    def compute_output_shape(self, input_shape):

        return tuple(int(input_shape[0]), self.units[-1])

    def get_config(self):

        config = {
            'units': self.units,
            'activation': self.activation,
            'l2_reg_list': self.l2_reg_list,
            'dropout_list': self.dropout_list,
            'use_bn': self.use_bn,
            'use_gate': self.use_gate
        }
        base_config = super(Mlp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Pooling(Layer):

    def __init__(self, pooling='sum', keepdims=True, **kwargs):

        self.pooling = pooling
        self.keepdims = keepdims
        super(Pooling, self).__init__(**kwargs)

        self.supports_masking = True

    def build(self, input_shape):

        if self.pooling not in ('sum', 'mean', 'max', 'min'):
            raise ValueError('pooling value only support sum,mean,max and min')

        super(Pooling, self).build(input_shape)

    def call(self, inputs, mask=None):

        steps_axis = 1
        if mask is not None:
            mask = math_ops.cast(mask, K.floatx())
            input_shape = inputs.shape.as_list()
            broadcast_shape = [-1, input_shape[steps_axis], 1]
            mask = array_ops.reshape(mask, broadcast_shape)
            inputs *= mask

            if self.pooling == 'mean':
                res = K.sum(inputs, axis=steps_axis) / math_ops.reduce_sum(mask, axis=steps_axis)
            elif self.pooling == 'sum':
                res = K.sum(inputs, axis=steps_axis)
            elif self.pooling == 'max':
                res = K.max(inputs, axis=steps_axis)
            elif self.pooling == 'min':
                res = K.min(inputs, axis=steps_axis)
        else:
            if self.pooling == 'mean':
                res = K.mean(inputs, axis=steps_axis)
            elif self.pooling == 'sum':
                res = K.sum(inputs, axis=steps_axis)
            elif self.pooling == 'max':
                res = K.max(inputs, axis=steps_axis)
            elif self.pooling == 'min':
                res = K.min(inputs, axis=steps_axis)

        if self.keepdims:
            return K.expand_dims(res, axis=1)
        else:
            return res

class AttentionPooling(Layer):

    def __init__(self, norm=True, attention_type='inner', attention_net:list=[80, 40], activation='sigmoid', keepdims=False):

        self.norm = norm
        self.attention_type = attention_type
        self.attention_net = attention_net
        self.activation = activation
        self.keepdims = keepdims

    def build(self, input_shape):

        if self.attention_type not in ('inner', 'mlp'):
            raise ValueError('attention_type should be inner or mlp')

        if len(input_shape) != 2:
            raise ValueError('AttentionPooling should have 2 inputs, first one is query with shape (None, 1, embed_size), second one is value with shape (None, k, embed_size)')

        query_shape, value_shape = input_shape

        if len(query_shape) != 3 or query_shape[1] != 1:
            raise ValueError('query should be with shape (None, 1, embed_size)')

        if len(value_shape) != 3:
            raise ValueError('value should be with shape (None, k, embed_size)')

        if query_shape[2] != value_shape[2]:
            raise ValueError('the embedding size from query and value should be the same, but got {} and {}'.format(query_shape[2], value_shape[2]))

        if self.attention_type == 'mlp':
            self.mlp = Mlp(
                units=self.attention_net,
                activation=self.activation
            )
            self.out_kernel = Dense(1)

        super(AttentionPooling, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):

        return None

    def call(self, inputs, mask=None, **kwargs):

        query, value = inputs
        embedding_size = query.shape[-1]
        step_size = value.shape[1]

        if self.attention_type == 'inner':
            attention_score = tf.squeeze(tf.matmul(value, query, transpose_b=True), axis=-1)
        else:
            querys = tf.tile(query, [1, step_size, 1])
            query_value = tf.concat([querys, value, querys-value, querys*value], axis=-1)
            attention_score = self.out_kernel(self.mlp(query_value))
            attention_score = tf.reshape(attention_score, (-1, step_size))

        if mask is not None:
            if mask[0] is not None:
                raise ValueError('query should not support mask')
            if mask[1] is not None:
                min_value_matrix = tf.ones_like(attention_score) * (-2**31)
                attention_score = tf.where(mask[1], attention_score, min_value_matrix)

        attention_score = tf.divide(attention_score, tf.sqrt(embedding_size*1.0))

        if self.norm:
            weighted_att_score = activations.softmax(attention_score)
            attention_vec = tf.squeeze(tf.matmul(tf.expand_dims(weighted_att_score, axis=1), value), axis=1)

        if not self.keepdims:
            out_shape = (-1, embedding_size)
        else:
            out_shape = (-1, 1, embedding_size)

        return tf.reshape(attention_vec, out_shape)

    def compute_output_shape(self, input_shape):

        value_shape = input_shape[1]
        if not self.keepdims:
            return (int(value_shape[0]), int(value_shape[2]))
        else:
            return (int(value_shape[0]), 1, int(value_shape[2]))

    def get_config(self):

        config = {
            'norm':self.norm,
            'attention_type':self.attention_type,
            'attention_net':self.attention_net,
            'activation':self.activation
        }
        config.update(super(AttentionPooling, self).get_config())
        return config

class Gate(Layer):

    def __init__(self, gate_type='vector_wise', share_gate=False, **kwargs):

        self.gate_type = gate_type
        self.share_gate = share_gate

        super(Gate, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) not in (2,3):
            raise ValueError("input's shape must hvae 2 or 3 dimentions but got{}".format(len(input_shape)))

        self.input_rank = len(input_shape)

        if len(input_shape) == 3:
            fields, nn_size = int(input_shape[1]), int(input_shape[2])
        else:
            fields, nn_size = int(input_shape[1]), 1
        self.fields = fields
        if self.share_gate:
            gate_field = 1
        else:
            gate_field = fields

        if self.gate_type == 'vector_wise':
            self.gate_w = self.add_weight(name='gate_w', shape=(gate_field, nn_size), initializer=get_initializer(),
                                          dtype=K.floatx(), trainable=True)
        elif self.gate_type == 'bit_wise':
            self.gate_w = self.add_weight(name='gate_w', shape=(nn_size, nn_size), initializer=get_initializer(),
                                          dtype=K.floatx(), trainable=True)
        else:
            raise ValueError('got an unexpected gate type:{}'.format(self.gate_type))

        self.gate_b = self.add_weight(name='gate_b', shape=(gate_field, ), initializer=initializers.get('zeros'),
                                      dtype=K.floatx(), trainable=True)

        super(Gate, self).build(input_shape)


    def call(self, inputs, **kwargs):

        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, -1)
        if self.gate_type == 'vector_wise':
            gate_act_val = tf.reduce_sum(tf.matmul(inputs, self.gate_w), axis=-1, keepdims=True)
            if not self.share_gate:
                gate_act_val = tf.reshape(tf.linalg.diag_part(gate_act_val), (-1, self.fields, 1))
        else:
            gate_act_val = tf.matmul(inputs, self.gate_w)

        activation_val = activations.sigmoid(gate_act_val+self.gate_b)

        if self.input_rank == 2:
            return tf.squeeze(tf.multiply(inputs, activation_val), axis=-1)
        else:
            return tf.multiply(inputs, activation_val)

    def get_config(self):

        config = {
            'gate_type': self.gate_type,
            'share_gate': self.share_gate
        }
        base_config = super(Gate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AddBias(Layer):

    def __init__(self, **kwargs):

        super(AddBias, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bias = self.add_weight(name='bias', shape=(1,),
                                    initializer=initializers.Zeros(), trainable=True)

    def call(self, inputs, **kwargs):

        return tf.math.reduce_sum(inputs, 1, keepdims=True) + self.bias

    def get_config(self):

        return super(AddBias, self).get_config()

class PositionInput(Layer):

    def __init__(self, dims, **kwargs):

        self.dims = dims
        super(PositionInput, self).__init__(**kwargs)

    def build(self, input_shape):

        self.position = self.add_weight(name='position', shape=(self.dims, ),
                                        dtype=tf.int64, initializer=initializers.Constant(np.arrane(1, 1+self.dims)),
                                        trainable=True)

        super(PositionInput, self).build(input_shape)

    def call(self, inputs, **kwargs):

        return tf.ones_like(inputs) * self.position

    def compute_output_shape(self, input_shape):

        return tuple(int(input_shape[0]), self.dims)

    def get_config(self):

        config = {
            'dims': self.dims
        }
        base_config = super(Gate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Variable(Layer):

    def __init__(self, shape, kernel_initializer='zeros', trainable = True, **kwargs):

        self.shape = shape
        self.kernel_initializer = initializers.get(get_initializer(kernel_initializer))
        self.trainable = trainable
        super(Variable, self).__init__(**kwargs)

    def build(self, input_shape):

        self.variable = self.add_weight(
            name='var',
            shape=self.shape,
            dtype=tf.float32,
            trainable=self.trainable,
            initializer=self.kernel_initializer
        )

        super(Variable, self).build(input_shape)

    def call(self, inputs, **kwargs):

        return self.variable

    def get_config(self):

        config = {
            'shape': self.shape,
            'kernel_initializer': self.kernel_initializer,
            'trainable': self.trainable
        }

        base_config = super(Variable, self).get_config()
        return dict(list(base_config) + list(config.items()))

class SelfAttention(Layer):

    def __init__(self, heads, kernel_initializer='glorot_normal', use_ln=False, query_units=16,
                 dropout_rate=0.0, res_connect=True, **kwargs):

        self.heads = heads
        self.kernel_initializer = initializers.get(get_initializer(kernel_initializer))
        self.use_ln = use_ln
        self.query_units = query_units
        self.dropout_rate = dropout_rate
        self.res_connent = res_connect
        self.suuport_mask = True

        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        if isinstance(input_shape, list):
            raise ValueError('SelfAttention layer only support one input but got a list')

        if len(input_shape) != 3:
            raise ValueError('SelfAttention layer\'s input should be 3D but got {}D '.format(len(input_shape)))

        embedding_size = int(input_shape[-1])
        fields = int(input_shape[1])

        self.query_matrix = self.add_weight(
            name = 'query_matrix',
            shape = (embedding_size, self.query_units * self.heads),
            dtype=K.floatx(),
            initializers=self.kernel_initializer,
            trainable=True
        )

        self.key_matrix = self.add_weight(
            name='key_matrix',
            shape=(embedding_size, self.query_units * self.heads),
            dtype=K.floatx(),
            initializers=self.kernel_initializer,
            trainable=True
        )

        self.value_matrix = self.add_weight(
            name='value_matrix',
            shape=(embedding_size, self.query_units * self.heads),
            dtype=K.floatx(),
            initializers=self.kernel_initializer,
            trainable=True
        )

        if self.use_ln:
            self.layer_norm = LayerNormalization()

        if self.dropout_rate > 0.0:
            self.dropout = Dropout(self.dropout_rate)


