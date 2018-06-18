import os
import sys
import librosa
import json
import numpy as np
import tensorflow as tf
from argparse import Namespace
sys.path.append(os.path.dirname(os.getcwd()))
from wavenet.wavenet import Wavenet
from auxilaries import mel_extractor

os.environ['CUDA_VISIBLE_DEVICES'] = ''

with open('../config_jsons/wavenet_mol.json', 'rt') as F:
    configs = json.load(F)
hparams = Namespace(**configs)
wavenet = Wavenet(hparams)

seq_len = 7680
wav_val, _ = librosa.load('test_data/test.wav', sr=16000)
# batch_size = len(wav_val) // seq_len
batch_size = 4
wav_val = wav_val[:batch_size * seq_len].reshape([batch_size, seq_len])

wav_shape = [batch_size, seq_len]
mel_shape = [batch_size, 39, 80]

mel_val = np.zeros(mel_shape)
for i in range(batch_size):
    mel_val[i] = mel_extractor.melspectrogram(wav_val[i])

wav_ph = tf.placeholder(tf.float32, wav_shape)
mel_ph = tf.placeholder(tf.float32, mel_shape)
inputs = {'wav': wav_ph,
          'mel': mel_ph}

tf.set_random_seed(12345)
encode_dict = wavenet.encode_signal(inputs)
inputs.update(encode_dict)
ff_dict = wavenet.feed_forward(inputs, init=False)
ff_dict.update(encode_dict)
loss_dict = wavenet.calculate_loss(ff_dict)


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
wavenet_vals = sess.run([ff_dict['real_targets'], ff_dict['cate_targets'],
                         ff_dict['out_params'], ff_dict['encoding'],
                         loss_dict['loss']],
                        feed_dict={wav_ph: wav_val,
                                   mel_ph: mel_val})

rt, ct, out, mel_en, loss_val = wavenet_vals
logit, mean, log_scale = np.split(out, indices_or_sections=[10, 20], axis=2)
print(logit.mean())
print(mean.mean())
print(log_scale.mean())
print(np.exp(log_scale.mean()))

print(len(set(rt.flatten().tolist())))
print(len(set(ct.flatten().tolist())))
print(loss_val)

# check random initialization probability and random assignment probability.
prob = 1. / np.exp(loss_val)
print(prob)
print(1. / wavenet.quant_chann)

