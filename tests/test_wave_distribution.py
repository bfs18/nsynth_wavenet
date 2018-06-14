import os
import librosa
import glob
import numpy as np
import matplotlib.pyplot as plt
from auxilaries import utils

use_mu_law = False
test_norm_data = True

if test_norm_data:
    wav_dir = '/data/corpus/LJSpeech-1.1/wavs_16k_norm'
else:
    wav_dir = '/data/corpus/LJSpeech-1.1/wavs_16k'

wavs = glob.glob(os.path.join(wav_dir, '*.wav'))

wav_data = []
for wav in wavs:
    wd, _ = librosa.load(wav, sr=16000)
    if use_mu_law:
        wd = utils.inv_cast_quantize_numpy(utils.mu_law_numpy(wd), 256)
    wav_data.append(wd)
wav_data = np.concatenate(wav_data)

if test_norm_data:
    normal_scale = 0.8 if use_mu_law else 0.18
    logistic_scale = 0.4 if use_mu_law else 0.09
    max_height = 2.5 if use_mu_law else 10
    text_height = 1.5
else:
    normal_scale = 0.6 if use_mu_law else 0.1
    logistic_scale = 0.3 if use_mu_law else 0.05
    max_height = 5. if use_mu_law else 20
    text_height = 4.

data_shape = wav_data.shape

wav_mean = np.mean(wav_data)
wav_std = np.std(wav_data)

plt.figure(1)
plt.subplot(3, 1, 1)
plt.hist(wav_data, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Wave values')
plt.ylabel('Probability')
plt.text(-0.75, text_height, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(wav_mean, wav_std))
plt.axis([-1., 1., 0., max_height])
plt.title('Histogram of Wave values')
plt.grid(True)

plt.figure(3)
plt.subplot(2, 1, 1)
plt.hist(wav_data, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Wave values')
plt.ylabel('Probability')
plt.text(-0.75, text_height, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(wav_mean, wav_std))
plt.axis([-1., 1., 0., max_height])
plt.title('Histogram of Wave values')
plt.grid(True)

wav_data = np.abs(wav_data)
wav_mean = np.mean(wav_data)
wav_std = np.std(wav_data)

plt.figure(2)
plt.subplot(3, 1, 1)
plt.hist(wav_data, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Abs Wave values')
plt.ylabel('Probability')
plt.text(0.6, text_height, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(wav_mean, wav_std))
plt.axis([0., 1., 0., max_height])
plt.title('Histogram of Abs Wave values')
plt.grid(True)

plt.figure(3)
plt.subplot(2, 1, 2)
plt.hist(wav_data, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Abs Wave values')
plt.ylabel('Probability')
plt.text(0.6, text_height, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(wav_mean, wav_std))
plt.axis([0., 1., 0., max_height])
plt.title('Histogram of Abs Wave values')
plt.grid(True)

del wav_data

###
# Logistic random variables
###
rl = np.random.logistic(loc=0.0, scale=logistic_scale, size=data_shape)
rl[rl > 1.0] = 1.0
rl[rl < -1.0] = -1.0

plt.figure(1)
plt.subplot(3, 1, 2)
plt.hist(rl, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Logistic random variables')
plt.ylabel('Probability')
plt.text(-0.75, text_height, r'$\mu=0.0,\ \sigma={}$'.format(logistic_scale))
plt.axis([-1., 1., 0., max_height])
plt.title('Histogram of Logistic random variables')
plt.grid(True)

rl = np.abs(rl)
rl_mean = np.mean(rl)
rl_std = np.std(rl)

plt.figure(2)
plt.subplot(3, 1, 2)
plt.hist(rl, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Abs Logistic random variables')
plt.ylabel('Probability')
plt.text(0.6, text_height, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(rl_mean, rl_std))
plt.axis([0., 1., 0., max_height])
plt.title('Histogram of Abs Logistic random variables')
plt.grid(True)

del rl

###
# Normal random variables
###
rn = np.random.normal(loc=0., scale=normal_scale, size=data_shape)
rn[rn > 1.0] = 1.0
rn[rn < -1.0] = -1.0

plt.figure(1)
plt.subplot(3, 1, 3)
plt.hist(rn, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Normal random variables')
plt.ylabel('Probability')
plt.text(-0.75, text_height, r'$\mu=0.0,\ \sigma={}$'.format(normal_scale))
plt.axis([-1., 1., 0., max_height])
plt.title('Histogram of Normal random variables')
plt.grid(True)

rn = np.abs(rn)
rn_mean = np.mean(rn)
rn_std = np.std(rn)

plt.figure(2)
plt.subplot(3, 1, 3)
plt.hist(rn, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Abs Normal random variables')
plt.ylabel('Probability')
plt.text(0.6, text_height, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(rn_mean, rn_std))
plt.axis([0., 1., 0., max_height])
plt.title('Histogram of Abs Normal random variables')
plt.grid(True)

del rn

plt.figure(1)
fig1 = plt.gcf()
plt.figure(2)
fig2 = plt.gcf()
plt.figure(3)
fig3 = plt.gcf()

plt.show()

prefix = 'norm-' if test_norm_data else ''
surfix = '-mu_law' if use_mu_law else ''
fig1.savefig('figures/{}dist{}.png'.format(prefix, surfix),
             format='png', dpi=200)
fig2.savefig('figures/{}dist_abs{}.png'.format(prefix, surfix),
             format='png', dpi=200)
fig3.savefig('figures/{}x_x_abs-stat{}.png'.format(prefix, surfix),
             format='png', dpi=100)



