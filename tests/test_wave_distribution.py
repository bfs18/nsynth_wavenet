import os
import librosa
import glob
import numpy as np
import matplotlib.pyplot as plt


wav_dir = '/data/corpus/LJSpeech-1.1/wavs_16k'
wavs = glob.glob(os.path.join(wav_dir, '*.wav'))[:2000]

wav_data = np.asarray([], dtype=np.float32)
for wav in wavs:
    wd, _ = librosa.load(wav, sr=16000)
    wav_data = np.concatenate([wav_data, wd])

normal_scale = 0.1
logistic_scale = 0.05

rn = np.random.normal(loc=0., scale=normal_scale, size=wav_data.shape)
rn[rn > 1.0] = 1.0
rn[rn < -1.0] = -1.0

ru = np.random.uniform(low=1e-5, high=1.0 - 1e-5, size=wav_data.shape)
rl = (np.log(ru) - np.log(1 - ru)) * logistic_scale
rl[rl > 1.0] = 1.0
rl[rl < -1.0] = -1.0

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

plt.subplot(3, 1, 2)
plt.hist(rl, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Logistic random variables')
plt.ylabel('Probability')
plt.text(-0.75, 4., r'$\mu=0.0,\ \sigma={}$'.format(logistic_scale))
plt.axis([-1., 1., 0., 20.])
plt.title('Histogram of Logistic random variables')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.hist(rn, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Normal random variables')
plt.ylabel('Probability')
plt.text(-0.75, 4., r'$\mu=0.0,\ \sigma={}$'.format(normal_scale))
plt.axis([-1., 1., 0., 20.])
plt.title('Histogram of Normal random variables')
plt.grid(True)

fig = plt.gcf()
plt.show()
# fig.savefig('dist.eps', format='eps')
fig.savefig('dist.png', format='png', dpi=100)

