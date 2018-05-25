import os
import sys
import numpy as np
import tensorflow as tf
import librosa
sys.path.append(os.path.dirname(os.getcwd()))
from auxilaries import mel_extractor


test_wave = 'test_data/test.wav'
waveform, _ = librosa.load(test_wave, sr=16000)
mel_spec_np = mel_extractor.melspectrogram(waveform)

y_ph = tf.placeholder(dtype=tf.float32, shape=[waveform.shape[0]])
ms = mel_extractor.tf_melspectrogram(y_ph)
sess = tf.Session()
mel_spec_tf = sess.run(ms, feed_dict={y_ph: waveform})

print(np.allclose(mel_spec_np, mel_spec_tf))


waveform = waveform[: (waveform.shape[0] // 7680) * 7680]
waveform_batch = waveform.reshape([-1, 7680])
batch_mel_spec_np = mel_extractor.batch_melspectrogram(waveform_batch)

by_ph = tf.placeholder(dtype=tf.float32, shape=[waveform.shape[0] // 7680, 7680])
bms = mel_extractor.tf_batch_melspectrogram(by_ph)
batch_mel_spec_tf = sess.run(bms, feed_dict={by_ph: waveform_batch})

print(np.allclose(batch_mel_spec_np, batch_mel_spec_tf))
