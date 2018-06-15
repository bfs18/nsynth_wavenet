# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library of functions that help with causal masking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf
from auxilaries.utils import get_kernel


def shift_right(x):
    """Shift the input over by one and a zero to the front.

    Args:
      x: The [mb, time, channels] tensor input.

    Returns:
      x_sliced: The [mb, time, channels] tensor output.
    """
    shape = x.get_shape().as_list()
    x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
    x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, shape[1], -1]))
    x_sliced.set_shape(shape)
    return x_sliced


def mul_or_none(a, b):
    """Return the element wise multiplicative of the inputs.

    If either input is None, we return None.

    Args:
      a: A tensor input.
      b: Another tensor input with the same type as a.

    Returns:
      None if either input is None. Otherwise returns a * b.
    """
    if a is None or b is None:
        return None
    return a * b


def time_to_batch(x, block_size):
    """Splits time dimension (i.e. dimension 1) of `x` into batches.

    Within each batch element, the `k*block_size` time steps are transposed,
    so that the `k` time steps in each output batch element are offset by
    `block_size` from each other.

    The number of input time steps must be a multiple of `block_size`.

    Args:
      x: Tensor of shape [nb, k*block_size, n] for some natural number k.
      block_size: number of time steps (i.e. size of dimension 1) in the output
        tensor.

    Returns:
      Tensor of shape [nb*block_size, k, n]
    """
    shape = x.get_shape().as_list()
    y = tf.reshape(x, [
        shape[0], shape[1] // block_size, block_size, shape[2]
    ])
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [
        shape[0] * block_size, shape[1] // block_size, shape[2]
    ])
    y.set_shape([
        mul_or_none(shape[0], block_size), mul_or_none(shape[1], 1. / block_size),
        shape[2]
    ])
    return y


def batch_to_time(x, block_size):
    """Inverse of `time_to_batch(x, block_size)`.

    Args:
      x: Tensor of shape [nb*block_size, k, n] for some natural number k.
      block_size: number of time steps (i.e. size of dimension 1) in the output
        tensor.

    Returns:
      Tensor of shape [nb, k*block_size, n].
    """
    shape = x.get_shape().as_list()
    y = tf.reshape(x, [shape[0] // block_size, block_size, shape[1], shape[2]])
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, [shape[0] // block_size, shape[1] * block_size, shape[2]])
    y.set_shape([mul_or_none(shape[0], 1. / block_size),
                 mul_or_none(shape[1], block_size),
                 shape[2]])
    return y


def conv1d(x,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(1.0),
           biases_initializer=tf.constant_initializer(0.0),
           use_weight_norm=False):
    """Fast 1D convolution that supports causal padding and dilation.

    Args:
      x: The [mb, time, channels] float tensor that we convolve.
      num_filters: The number of filter maps in the convolution.
      filter_length: The integer length of the filter.
      name: The name of the scope for the variables.
      dilation: The amount of dilation.
      causal: Whether or not this is a causal convolution.
      kernel_initializer: The kernel initialization function.
      biases_initializer: The biases initialization function.
      use_weight_norm: use weight normalization or not.

    Returns:
      y: The output of the 1D convolution.
    """
    batch_size, length, num_input_channels = x.get_shape().as_list()
    assert length % dilation == 0

    kernel_shape = [1, filter_length, num_input_channels, num_filters]
    strides = [1, 1, 1, 1]
    biases_shape = [num_filters]
    padding = 'VALID' if causal else 'SAME'

    with tf.variable_scope(name):
        weights = get_kernel(
            kernel_shape=kernel_shape, name='W',
            initializer=kernel_initializer, use_weight_norm=use_weight_norm)
        biases = tf.get_variable(
            'biases', shape=biases_shape, initializer=biases_initializer)

    x_ttb = time_to_batch(x, dilation)
    if filter_length > 1 and causal:
        x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

    x_ttb_shape = x_ttb.get_shape().as_list()
    x_4d = tf.reshape(x_ttb, [x_ttb_shape[0], 1,
                              x_ttb_shape[1], num_input_channels])
    y = tf.nn.conv2d(x_4d, weights, strides, padding=padding)
    y = tf.nn.bias_add(y, biases)
    y_shape = y.get_shape().as_list()
    y = tf.reshape(y, [y_shape[0], y_shape[2], num_filters])
    y = batch_to_time(y, dilation)
    y.set_shape([batch_size, length, num_filters])
    return y


def trans_conv1d(x,
                 num_filters,
                 filter_length,
                 stride,
                 name,
                 activation=tf.nn.tanh,
                 kernel_initializer=tf.uniform_unit_scaling_initializer(1.15),
                 biases_initializer=tf.constant_initializer(0.0),
                 use_weight_norm=False):
    batch_size, length, num_input_channels = x.get_shape().as_list()
    x_4d = tf.reshape(x, [batch_size, 1, length, num_input_channels])
    output_length = stride * length

    kernel_shape = (1, filter_length, num_filters, num_input_channels)
    strides = (1, 1, stride, 1)
    output_shape = (batch_size, 1, output_length, num_filters)
    biases_shape = (num_filters,)
    padding = 'SAME'

    with tf.variable_scope(name):
        weights = get_kernel(
            kernel_shape=kernel_shape, name='kernel', deconv=True,
            initializer=kernel_initializer, use_weight_norm=use_weight_norm)
        biases = tf.get_variable(
            'bias', shape=biases_shape, initializer=biases_initializer)

    y = tf.nn.conv2d_transpose(
        x_4d,
        filter=weights,
        output_shape=output_shape,
        strides=strides,
        padding=padding)

    y = tf.nn.bias_add(y, biases)
    if activation is not None:
        y = activation(y)
    y = tf.reshape(y, [batch_size, output_length, num_filters])
    y.set_shape([batch_size, output_length, num_filters])
    return y


def _trans_conv1d(x,
                  num_filters,
                  filter_length,
                  stride,
                  name,
                  activation=tf.nn.tanh,
                  kernel_initializer=tf.uniform_unit_scaling_initializer(1.15),
                  biases_initializer=tf.constant_initializer(0.0)):
    batch_size, length, num_input_channels = x.get_shape().as_list()
    x_4d = tf.reshape(x, [batch_size, 1, length, num_input_channels])
    y = tf.layers.conv2d_transpose(x_4d, num_filters, (1, filter_length), (1, stride),
                                   padding='SAME',
                                   use_bias=True,
                                   activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=biases_initializer,
                                   name=name)
    y = tf.reshape(y, [batch_size, length * stride, num_filters])
    y.set_shape([batch_size, length * stride, num_filters])
    return y


