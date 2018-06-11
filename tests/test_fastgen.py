import os
import sys
import librosa
import json
import numpy as np
import tensorflow as tf
from argparse import Namespace
sys.path.append(os.path.dirname(os.getcwd()))
from wavenet.wavenet import Fastgen

os.environ['CUDA_VISIBLE_DEVICES'] = ''

with open('../config_jsons/wavenet_mol.json', 'rt') as F:
    configs = json.load(F)
hparams = Namespace(**configs)

batch_size = 4
fastgen = Fastgen(hparams, batch_size)

wav_shape = [batch_size, 1]
mel_en_shape = [batch_size, hparams.deconv_width]

wav_ph = tf.placeholder(tf.float32, wav_shape)
mel_en_ph = tf.placeholder(tf.float32, mel_en_shape)
inputs = {'wav': wav_ph,
          'encoding': mel_en_ph}
tf.set_random_seed(12345)
fg_dict = fastgen.sample(inputs)

np.random.seed(12345)
wav_val = np.random.uniform(-1, 1, wav_shape)
mel_en_val = np.random.uniform(-1, 1, mel_en_shape)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(fg_dict['init_ops'])

s, _ = sess.run([fg_dict['sample'], fg_dict['push_ops']],
                feed_dict={wav_ph: wav_val, mel_en_ph: mel_en_val})


print(s)

