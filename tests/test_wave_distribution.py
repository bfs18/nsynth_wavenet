import os
import librosa
import glob
import numpy as np
import matplotlib.pyplot as plt


wav_dir = '/data/corpus/LJSpeech-1.1/wavs_16k'
wavs = glob.glob(os.path.join(wav_dir, '*.wav'))

wav_data = []
for wav in wavs:
    wd, _ = librosa.load(wav, sr=16000)
    wav_data.append(wd)
wav_data = np.concatenate(wav_data)

normal_scale = 0.1
logistic_scale = 0.05

data_shape = wav_data.shape

wav_mean = np.mean(wav_data)
wav_std = np.std(wav_data)
plt.subplot(3, 1, 1)
plt.hist(wav_data, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Wave values')
plt.ylabel('Probability')
plt.text(-0.75, 4., r'$\mu={:.3f},\ \sigma={:.3f}$'.format(wav_mean, wav_std))
plt.axis([-1., 1., 0., 20.])
plt.title('Histogram of Wave values')
plt.grid(True)
del wav_data

###
# Logistic random variables
###
rl = np.random.logistic(loc=0.0, scale=logistic_scale, size=data_shape)
rl[rl > 1.0] = 1.0
rl[rl < -1.0] = -1.0

plt.subplot(3, 1, 2)
plt.hist(rl, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Logistic random variables')
plt.ylabel('Probability')
plt.text(-0.75, 4., r'$\mu=0.0,\ \sigma={}$'.format(logistic_scale))
plt.axis([-1., 1., 0., 20.])
plt.title('Histogram of Logistic random variables')
plt.grid(True)
del rl

###
# Normal random variables
###
rn = np.random.normal(loc=0., scale=normal_scale, size=data_shape)
rn[rn > 1.0] = 1.0
rn[rn < -1.0] = -1.0

plt.subplot(3, 1, 3)
plt.hist(rn, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Normal random variables')
plt.ylabel('Probability')
plt.text(-0.75, 4., r'$\mu=0.0,\ \sigma={}$'.format(normal_scale))
plt.axis([-1., 1., 0., 20.])
plt.title('Histogram of Normal random variables')
plt.grid(True)
del rn

fig = plt.gcf()
plt.show()
# fig.savefig('dist.eps', format='eps')
fig.savefig('dist2.png', format='png', dpi=100)

