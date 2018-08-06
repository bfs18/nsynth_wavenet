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
from functools import partial


WN_INIT_SCALE = 1.0


def get_upsample_act(act_str):
    if act_str == 'tanh':
        return tf.nn.tanh
    elif act_str == 'relu':
        return tf.nn.relu
    elif act_str == 'leaky_relu':
        return partial(tf.nn.leaky_relu, alpha=0.4)
    else:
        raise ValueError('Unsupported activation function for upsample layer')


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


def l2_norm(tensor, axis, keep_dims=True):
    s = tf.reduce_sum(tf.pow(tensor, 2.0),
                      axis=axis, keep_dims=keep_dims)
    return tf.sqrt(s)


def get_kernel(kernel_shape, initializer, name,
               use_weight_norm=False, deconv=False):
    if use_weight_norm:
        # if deconv == True, the 2nd dim in kernel_shape is the output size
        # if deconv == False, the 3rd dim in kernel_shape is the output size
        if deconv:
            norm_axis = (0, 1, 3)
            out_dim = kernel_shape[2]
            g_bc_shape = [1, 1, out_dim, 1]
        else:
            norm_axis = (0, 1, 2)
            out_dim = kernel_shape[3]
            g_bc_shape = [1, 1, 1, out_dim]

        V = tf.get_variable(
            '{}_V'.format(name), shape=kernel_shape, initializer=initializer)
        g = tf.get_variable(
            '{}_g'.format(name),
            initializer=l2_norm(V.initialized_value(), axis=norm_axis, keep_dims=False))
        V_norm = tf.nn.l2_normalize(V, axis=norm_axis)
        weights = V_norm * tf.reshape(g, shape=g_bc_shape)
    else:
        weights = tf.get_variable(
            name, shape=kernel_shape, initializer=initializer)
    return weights


def conv1d(x,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.random_normal_initializer(0, 0.05),
           biases_initializer=tf.constant_initializer(0.0),
           use_weight_norm=False,
           init=False,
           dropout_rate=0.0):
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
      init: run data dependent initialization for g and b.
      dropout_rate: ues dropout

    Returns:
      y: The output of the 1D convolution.
    """
    batch_size, length, num_input_channels = x.get_shape().as_list()
    assert length % dilation == 0

    kernel_shape = [1, filter_length, num_input_channels, num_filters]
    strides = [1, 1, 1, 1]
    biases_shape = [num_filters]
    padding = 'VALID' if causal else 'SAME'

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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

    if use_weight_norm and init:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            g = tf.get_variable('W_g')
            b = tf.get_variable('biases')
            m_init, v_init = tf.nn.moments(y, [0, 1, 2])
            scale_init = WN_INIT_SCALE / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies(
                    [g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # y = tf.identity(y)
                # the conv outputs should be recalculated after init for g and b
                weights = get_kernel(kernel_shape=kernel_shape, name='W',
                                     initializer=None, use_weight_norm=use_weight_norm)
                y = tf.nn.conv2d(x_4d, weights, strides, padding=padding)
                y = tf.nn.bias_add(y, biases)

    y_shape = y.get_shape().as_list()
    y = tf.reshape(y, [y_shape[0], y_shape[2], num_filters])
    y = batch_to_time(y, dilation)
    y.set_shape([batch_size, length, num_filters])

    if dropout_rate > 0.0:
        y = tf.layers.dropout(y, rate=dropout_rate, training=True)

    return y


def trans_conv1d(x,
                 num_filters,
                 filter_length,
                 stride,
                 name,
                 activation,
                 kernel_initializer=tf.random_normal_initializer(0, 0.05),
                 biases_initializer=tf.constant_initializer(0.0),
                 use_weight_norm=False,
                 init=False):
    batch_size, length, num_input_channels = x.get_shape().as_list()
    x_4d = tf.reshape(x, [batch_size, 1, length, num_input_channels])
    output_length = stride * length

    kernel_shape = (1, filter_length, num_filters, num_input_channels)
    strides = (1, 1, stride, 1)
    output_shape = (batch_size, 1, output_length, num_filters)
    biases_shape = (num_filters,)
    padding = 'SAME'

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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

    if use_weight_norm and init:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            g = tf.get_variable('kernel_g')
            b = tf.get_variable('bias')
            m_init, v_init = tf.nn.moments(y, [0, 1, 2])
            scale_init = WN_INIT_SCALE / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies(
                    [g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # y = tf.identity(y)
                weights = get_kernel(
                    kernel_shape=kernel_shape, name='kernel', deconv=True,
                    initializer=None, use_weight_norm=use_weight_norm)
                y = tf.nn.conv2d_transpose(
                    x_4d, filter=weights, output_shape=output_shape, strides=strides,
                    padding=padding)
                y = tf.nn.bias_add(y, biases)

    if activation is not None:
        y = activation(y)
    y = tf.reshape(y, [batch_size, output_length, num_filters])
    y.set_shape([batch_size, output_length, num_filters])
    return y


def resize_conv1d(x,
                  num_filters,
                  filter_length,
                  stride,
                  name,
                  activation,
                  kernel_initializer=tf.random_normal_initializer(0, 0.05),
                  bias_initializer=tf.constant_initializer(0.0),
                  use_weight_norm=False,
                  init=False):
    x_4d = tf.expand_dims(x, axis=1)
    in_length = x_4d.get_shape().as_list()[2]
    ups_x_4d = tf.image.resize_nearest_neighbor(
        x_4d, [1, in_length * stride], name='{}/resize'.format(name))

    x = tf.squeeze(ups_x_4d, axis=1)
    y = conv1d(x,
               num_filters=num_filters,
               filter_length=filter_length,
               name=name,
               causal=False,
               dilation=1,
               kernel_initializer=kernel_initializer,
               biases_initializer=bias_initializer,
               use_weight_norm=use_weight_norm,
               init=init)
    if activation:
        y = activation(y)
    return y


# ---------------------------------------------------
# Inference functions
# ---------------------------------------------------
def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate,
                  batch_size, use_weight_norm=False):
    """Applies dilated convolution using queues.

    Assumes a filter_length of 3.

    Args:
      x: The [mb, time, channels] tensor input.
      n_inputs: The input number of channels.
      n_outputs: The output number of channels.
      name: The variable scope to provide to W and biases.
      filter_length: The length of the convolution, assumed to be 3.
      rate: The rate or dilation
      batch_size: Non-symbolic value for batch_size.
      use_weight_norm: use weight normalization or not

    Returns:
      y: The output of the operation
      (init_1, init_2): Initialization operations for the queues
      (push_1, push_2): Push operations for the queues
    """
    assert filter_length == 3

    # create queue
    q_1 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, 1, n_inputs))
    q_2 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, 1, n_inputs))
    init_1 = q_1.enqueue_many(tf.zeros((rate, batch_size, 1, n_inputs)))
    init_2 = q_2.enqueue_many(tf.zeros((rate, batch_size, 1, n_inputs)))
    state_1 = q_1.dequeue()
    push_1 = q_1.enqueue(x)
    state_2 = q_2.dequeue()
    push_2 = q_2.enqueue(state_1)

    # get pretrained weights
    with tf.variable_scope(name):
        w = get_kernel(
            kernel_shape=[1, filter_length, n_inputs, n_outputs],
            name='W', initializer=None, use_weight_norm=use_weight_norm)
        b = tf.get_variable(
            name="biases", shape=[n_outputs], dtype=tf.float32)

    w_q_2 = tf.slice(w, [0, 0, 0, 0], [-1, 1, -1, -1])
    w_q_1 = tf.slice(w, [0, 1, 0, 0], [-1, 1, -1, -1])
    w_x = tf.slice(w, [0, 2, 0, 0], [-1, 1, -1, -1])

    # perform op w/ cached states
    y = tf.nn.bias_add(
        tf.matmul(state_2[:, 0, :], w_q_2[0][0]) + tf.matmul(
            state_1[:, 0, :], w_q_1[0][0]) + tf.matmul(x[:, 0, :], w_x[0][0]), b)

    y = tf.expand_dims(y, 1)

    return y, (init_1, init_2), (push_1, push_2)


def linear(x, n_inputs, n_outputs, name, use_weight_norm=False):
    """Simple linear layer.

    Args:
      x: The [mb, time, channels] tensor input.
      n_inputs: The input number of channels.
      n_outputs: The output number of channels.
      name: The variable scope to provide to W and biases.
      use_weight_norm: use weight normalization or not.

    Returns:
      y: The output of the operation.
    """
    with tf.variable_scope(name):
        w = get_kernel(
            kernel_shape=[1, 1, n_inputs, n_outputs],
            name='W', initializer=None, use_weight_norm=use_weight_norm)
        b = tf.get_variable(
            name="biases", shape=[n_outputs], dtype=tf.float32)
    y = tf.nn.bias_add(tf.matmul(x[:, 0, :], w[0][0]), b)
    y = tf.expand_dims(y, 1)

    return y


# ####################################################################
# deprecated functions.
# ####################################################################
def _trans_conv1d(x,
                  num_filters,
                  filter_length,
                  stride,
                  name,
                  activation,
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


