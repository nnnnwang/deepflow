# -*- coding: utf-8 -*-
# @Time    : 2020/12/9
# @Author  : 爱喝可乐的学渣
# @Email   : wangnannanzju@qq.com

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class FM(Layer):

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(' input shape must be 3d type but got {}d'.format(len(input())))
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        square_part_one = tf.reduce_sum(
            tf.square(
                tf.reduce_sum(inputs, axis=1, keepdims=False)
            ), axis=-1, keepdims=True
        )

        square_part_two = tf.reduce_sum(
            tf.reduce_sum(
                tf.square(inputs), axis=-1, keepdims=False
            ), axis=-1, keepdims=True
        )

        return 0.5 * tf.subtract(square_part_one, square_part_two)

    def compute_output_shape(self, input_shape):
        return tuple(int(input_shape[0]), 1)

    def get_config(self):
        return super(FM, self).get_config()

class Crossing(Layer):

    def __init__(self,
                 cross_nums=2,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 **kwargs):
        self.cross_nums = cross_nums
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        super(Crossing, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('input_shape should be 2d type but got {}d'.format(len(input_shape)))
        self.kernel = [
            self.add_weight(
                name='cross_kernel_'+str(index),
                shape=[int(input_shape[-1]), 1],
                dtype=K.floatx(),
                initializer=self.kernel_initializer,
                regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                trainable=True
            ) for index in range(self.cross_nums)
        ]
        self.bias = [
            self.add_weight(
                name='bias_'+str(index),
                shape=[int(input_shape[-1]), ],
                dtype=K.floatx(),
                initializer=self.bias_initializer,
                trainable=True
            ) for index in range(self.cross_nums)
        ]
        super(Crossing, self).build(input_shape)

    def call(self, inputs, **kwargs):

        output = inputs
        for index in range(len(self.kernel)):
            output = tf.nn.bias_add(
                tf.multiply(inputs, tf.matmul(output, self.kernel[index])), self.bias[index]
            ) + output

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'cross_nums': self.cross_nums,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'l2_reg': self.l2_reg,
            'l1_reg': self.l1_reg
        }
        base_config = super(Crossing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SecondOrderIntersection(Layer):

    def __init__(self, **kwargs):
        super(SecondOrderIntersection, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape[0], tuple):
            if len(input_shape)!= 3:
                raise ValueError('A `SecondOrderIntersection` layer should be called '
                                'on a inputs with 3d shape ')
            else:
                self.single_input = True
        else:
            if len(input_shape) != 2:
                raise ValueError(' SecondOrderIntersection layer only support almost 2 inputs ')
            self.single_input = False

        super(SecondOrderIntersection, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if self.single_input:
            feature_steps = int(inputs.shape[1])
            matrix_dot_res = tf.matmul(inputs, tf.transpose(inputs, [0, 2, 1]))
            ones = tf.ones_like(matrix_dot_res)
            mask_a = tf.compat.v1.matrix_band_part(ones, 0, -1)
            mask_b = tf.compat.v1.matrix_band_part(ones, 0, 0)
            mask = tf.cast(mask_a - mask_b, dtyp=tf.bool)
            output_tensor = tf.reshape(tf.boolean_mask(matrix_dot_res, mask), (-1, int(feature_steps * (feature_steps-1) / 2)))
            return output_tensor
        else:
            first_input, second_input = inputs
            matrix_dot_res = tf.matmul(first_input, tf.transpose(second_input, [0, 2, 1]))
            return tf.reshape(matrix_dot_res, (-1, first_input.shape[1] * second_input[1]))

    def compute_output_shape(self, input_shape):
        if self.single_input:
            batch_size = int(input_shape[0])
            feature_steps = int(input_shape[1])
            second_dim = feature_steps * (feature_steps-1) / 2
            return tuple(batch_size, second_dim)
        else:
            first_shape, second_shape = input_shape
            batch_size = int(first_shape[0])
            first_steps = int(first_shape[1])
            second_steps = int(second_shape[1])
            return tuple(batch_size, first_steps * second_steps)

    def get_config(self):
        return super(SecondOrderIntersection, self).get_config()

class BiInteraction(Layer):

    def __init__(self, keepdims=False, **kwargs):
        self.keepdims = keepdims
        super(BiInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('BiInteraction layer only support input tensor with 3d shape')
        super(BiInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):

        square_part_one = tf.square(
            tf.reduce_sum(inputs, axis=1, keepdims=False)
        )

        square_part_two = tf.reduce_sum(
            tf.square(inputs), axis=1, keepdims=False
        )

        ans = 0.5 * tf.subtract(square_part_one, square_part_two)

        if not self.keepdims:
            return ans
        else:
            return tf.expand_dims(ans, axis=1)

    def compute_output_shape(self, input_shape):
        if not self.keepdims:
            return tuple(int(input_shape[0]), int(input_shape[-1]))
        else:
            return tuple(int(input_shape[0]), 1, int(input_shape[-1]))

    def get_config(self):
        return super(BiInteraction, self).get_config()