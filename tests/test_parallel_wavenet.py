import os
import sys
import librosa
import json
import numpy as np
import tensorflow as tf
from argparse import Namespace
sys.path.append(os.path.dirname(os.getcwd()))
from wavenet.parallel_wavenet import ParallelWavenet
from wavenet.wavenet import Wavenet
from auxilaries import mel_extractor

os.environ['CUDA_VISIBLE_DEVICES'] = ''
with open('../config_jsons/wavenet_mol.json', 'rt') as F:
    te_configs = json.load(F)
te_hparams = Namespace(**te_configs)
teacher_wavenet = Wavenet(te_hparams)

with open('../config_jsons/parallel_wavenet.json', 'rt') as F:
    configs = json.load(F)
hparams = Namespace(**configs)
parallel_wavenet = ParallelWavenet(
    hparams, teacher=teacher_wavenet)

seq_len = 7680
wav_val, _ = librosa.load('test_data/test.wav', sr=16000)
batch_size = 4
wav_val = wav_val[:batch_size * seq_len].reshape([batch_size, seq_len])

wav_shape = [batch_size, seq_len]
mel_shape = [batch_size, 39, 80]
mel_val = np.zeros(mel_shape)
for i in range(batch_size):
    mel_val[i] = mel_extractor.melspectrogram(wav_val[i])

mel_ph = tf.placeholder(tf.float32, mel_shape)
wav_ph = tf.placeholder(tf.float32, wav_shape)
inputs = {'mel': mel_ph,
          'wav': wav_ph}

tf.set_random_seed(12345)
pff_dict = parallel_wavenet.feed_forward(inputs)
pff_dict.update(inputs)
loss_dict = parallel_wavenet.calculate_loss(pff_dict)

# test differentiable
st_vars = [var for var in tf.trainable_variables() if 'iaf' in var.name]
grads = tf.gradients(loss_dict['loss'], st_vars)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

# pff_vals = sess.run(pff_dict, feed_dict={mel_ph: mel_val})
# x = pff_vals['x']
# mean = pff_vals['mean_tot']
# scale = pff_vals['scale_tot']
# rl = pff_vals['rand_input']
# print(np.allclose(x, rl * scale + mean))

loss_vals, grad_vals = sess.run(
    [loss_dict, grads], feed_dict={mel_ph: mel_val, wav_ph: wav_val})
print(loss_vals)
print(sum([np.sum(np.isnan(gv)) for gv in grad_vals]))
