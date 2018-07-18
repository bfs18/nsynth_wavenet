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
"""Module to load the Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import os
import tensorflow as tf
import numpy as np
import random

from auxilaries import mel_extractor

# tf_melspectrogram extracts normalized log mel spectrogram
# tf_melspectrogram2 extracts mel spectrogram
USE_NEW_MEL_EXTRACTOR = False

if USE_NEW_MEL_EXTRACTOR:
    MELSPECTROGRAM = mel_extractor.tf_melspectrogram2
    BATCH_MELSPECTROGRAM = mel_extractor.batch_melspectrogram2
else:
    MELSPECTROGRAM = mel_extractor.tf_melspectrogram
    BATCH_MELSPECTROGRAM = mel_extractor.batch_melspectrogram

FEATURES = {
    'audio_id': tf.FixedLenFeature([], dtype=tf.string),
    'audio': tf.FixedLenFeature([], dtype=tf.string),
    'length': tf.FixedLenFeature([], dtype=tf.int64)}


def _tf_instance_log_mean_norm(tf_mel):
    log_mel = tf.log(tf.maximum(tf_mel, 1e-5))
    log_mel_mean = tf.reduce_mean(log_mel, axis=1, keepdims=True)
    norm_log_mel = log_mel - log_mel_mean
    norm_mel = tf.exp(norm_log_mel)
    return norm_mel


def _np_instance_log_mean_norm(np_mel):
    log_mel = np.log(np.maximum(np_mel, 1e-5))
    log_mel_mean = np.mean(log_mel, axis=1, keepdims=True)
    norm_log_mel = log_mel - log_mel_mean
    norm_mel = np.exp(norm_log_mel)
    return norm_mel


class Dataset(object):
    def __init__(self, tfrecord_path, is_training=True):
        self.is_training = is_training
        self.record_path = tfrecord_path

    def get_example(self, batch_size):
        reader = tf.TFRecordReader()
        num_epochs = None if self.is_training else 1
        capacity = batch_size
        path_queue = tf.train.string_input_producer(
            [self.record_path],
            num_epochs=num_epochs,
            shuffle=self.is_training,
            capacity=capacity,
            name='reader/input_producer')
        unused_eky, serialized_example = reader.read(path_queue)
        example = tf.parse_single_example(serialized_example, features=FEATURES)
        example['length'] = tf.cast(example['length'], tf.int32)
        audio = tf.decode_raw(example['audio'], tf.float32)
        example['audio'] = audio
        return example

    def get_batch(self, batch_size, length=15360):
        example = self.get_example(batch_size)
        wav = example['audio']
        key = example['audio_id']
        if self.is_training:
            crop = tf.random_crop(wav, [length])
            crop = tf.reshape(crop, [length])
            mel = MELSPECTROGRAM(crop)
            key, crop, mel = tf.train.shuffle_batch(
                [key, crop, mel],
                batch_size,
                num_threads=4,
                capacity=500 * batch_size,
                min_after_dequeue=200 * batch_size,
                name='reader/shuffle_batch')
        else:
            crop = tf.slice(wav, [0], [length])
            crop = tf.reshape(crop, [length])
            mel = MELSPECTROGRAM(crop)
            key, crop, mel = tf.train.batch(
                [key, crop, mel],
                batch_size,
                name='reader/fifo_batch')
        return {'key': key, 'wav': crop, 'mel': mel}


def np_random_crop(vector, crop_len):
    total_len = vector.shape[0]
    last_idx = total_len - crop_len
    start_idx = np.random.randint(low=0, high=last_idx + 1)
    cropped = vector[start_idx: start_idx + crop_len]
    return cropped


def get_init_batch(train_path, batch_size, seq_len=7680, first_n=1000):
    train_path = os.path.abspath(os.path.expanduser(train_path))

    first_n_serialized_example = []
    serialized_examples = tf.python_io.tf_record_iterator(train_path)
    for i in range(first_n):
        first_n_serialized_example.append(serialized_examples.__next__())
    random.shuffle(first_n_serialized_example)

    batch_waves = []
    for i in range(batch_size):
        serialized_example = first_n_serialized_example[i]
        example_proto = tf.train.Example()
        example_proto.ParseFromString(serialized_example)
        bytes = example_proto.features.feature["audio"].bytes_list.value[0]
        wave = np.frombuffer(bytes, dtype=np.float32)
        batch_waves.append(np_random_crop(wave, crop_len=seq_len))
    batch_waves = np.stack(batch_waves)
    batch_mels = BATCH_MELSPECTROGRAM(batch_waves)

    return {'wav': batch_waves, 'mel': batch_mels}


def spec_feat_mean_std(train_path, feat_fn=lambda x: tf.pow(tf.abs(x), 2.0)):
    local_graph = tf.Graph()
    with local_graph.as_default():
        input_vals = get_init_batch(
            train_path, batch_size=4096, seq_len=7680, first_n=10000)['wav']
        ph = tf.placeholder(dtype=np.float32, shape=[4096, 7680])
        feat = feat_fn(mel_extractor._tf_stft(ph))

    tf.logging.info('Calculating mean and std for stft feat.')
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config, graph=local_graph)
    feat_val = sess.run(feat, feed_dict={ph: input_vals})
    mean_val = np.mean(feat_val, axis=(0, 1))
    std_val = np.std(feat_val, axis=(0, 1))
    tf.logging.info('Done calculating mean and std for stft feat.')

    return mean_val, std_val

