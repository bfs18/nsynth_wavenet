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
"""Utility functions for NSynth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os

# internal imports
import librosa
import numpy as np
import tensorflow as tf

from wavenet.masked import get_kernel

slim = tf.contrib.slim


def shell_path(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


# ===============================================================================
# WaveNet Functions
# ===============================================================================
def get_module(module_path):
    """Imports module from NSynth directory.

    Args:
      module_path: Path to module separated by dots.
        -> "configs.linear"

    Returns:
      module: Imported module.
    """
    import_path = "magenta.models.nsynth."
    module = importlib.import_module(import_path + module_path)
    return module


def load_audio(path, sample_length=64000, sr=16000):
    """Loading of a wave file.

    Args:
      path: Location of a wave file to load.
      sample_length: The truncated total length of the final wave file.
      sr: Samples per a second.

    Returns:
      out: The audio in samples from -1.0 to 1.0
    """
    audio, _ = librosa.load(path, sr=sr)
    if sample_length > 0:
        audio = audio[:sample_length]
    return audio


def mu_law(x, mu=255, int8=False):
    """A TF implementation of Mu-Law encoding.

    Args:
      x: The audio samples to encode.
      mu: The Mu to use in our Mu-Law.
      int8: Use int8 encoding.

    Returns:
      out: The Mu-Law encoded int8 data.
    """
    out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
    out = tf.floor(out * 128)
    if int8:
        out = tf.cast(out, tf.int8)
    return out


def mu_law_numpy(x, mu=255, int8=False):
    """A TF implementation of Mu-Law encoding.

    Args:
      x: The audio samples to encode.
      mu: The Mu to use in our Mu-Law.
      int8: Use int8 encoding.

    Returns:
      out: The Mu-Law encoded int8 data.
    """
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    if int8:
        out = out.astype(np.int8)
    return out


def inv_mu_law(x, mu=255):
    """A TF implementation of inverse Mu-Law.

    Args:
      x: The Mu-Law samples to decode.
      mu: The Mu we used to encode these samples.

    Returns:
      out: The decoded data.
    """
    x = tf.cast(x, tf.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = tf.sign(out) / mu * ((1 + mu) ** tf.abs(out) - 1)
    out = tf.where(tf.equal(x, 0), x, out)
    return out


def inv_mu_law_numpy(x, mu=255.0):
    """A numpy implementation of inverse Mu-Law.

    Args:
      x: The Mu-Law samples to decode.
      mu: The Mu we used to encode these samples.

    Returns:
      out: The decoded data.
    """
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu) ** np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


def cast_quantize(x, quant_chann):
    """ Cast real audio signal in range [-1, 1) to
        int values in range [-quant_chann / 2, quant_chann / 2).

    Args:
      x: real audio signal.
      quant_chann: number of quantization channels.

    Returns:
      out: quantized signal.
    """
    x_quantized = x * quant_chann / 2
    return tf.cast(tf.floor(x_quantized), tf.int32)


def inv_cast_quantize(x_quantized, quant_chann):
    x = tf.cast(x_quantized, tf.float32) / tf.cast(quant_chann / 2, tf.float32)
    return x


def cast_quantize_numpy(x, quant_chann):
    x_quantized = x * quant_chann / 2
    return x_quantized.astype(np.int32)


def inv_cast_quantize_numpy(x_quantized, quant_chann):
    x = x_quantized.astype(np.float32) / (quant_chann / 2)
    return x


# ===============================================================================
# Basic Functions
# ===============================================================================
def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    orig_shape = tensor.get_shape().as_list()
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    _prod = lambda s, r: None if s is None else s * r
    new_shape = [_prod(s, r) for s, r in zip(orig_shape, repeats)]
    repeated_tesnor.set_shape(new_shape)
    return repeated_tesnor


# ===============================================================================
# Keras functions
# ===============================================================================
def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))
