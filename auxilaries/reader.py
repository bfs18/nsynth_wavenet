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
import tensorflow as tf

from auxilaries import mel_extractor


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
        features = {
            'audio_id': tf.FixedLenFeature([], dtype=tf.string),
            'audio': tf.FixedLenFeature([], dtype=tf.string),
            'length': tf.FixedLenFeature([], dtype=tf.int64)}
        example = tf.parse_single_example(serialized_example, features)
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
            mel = mel_extractor.tf_melspectrogram(crop)
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
            mel = mel_extractor.tf_melspectrogram(crop)
            key, crop, mel = tf.train.batch(
                [key, crop, mel],
                batch_size,
                name='reader/fifo_batch')
        return {'key': key, 'wav': crop, 'mel': mel}

