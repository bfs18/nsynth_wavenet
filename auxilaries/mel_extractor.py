import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
import math

from scipy import signal


_mel_basis = None


mel_params = tf.contrib.training.HParams(
    sample_rate=16000,
    num_freq=1025,
    num_mel=80,
    frame_shift_ms=12.5,
    frame_length_ms=50,
    preemphasis=0.97,
    min_level_db=-100,
    mel_fmin=125,
    mel_fmax=7600)

FRAME_SHIFT = int(mel_params.frame_shift_ms * mel_params.sample_rate / 1000.)


def melspectrogram(y):
    D = _stft(y, mel_params)
    S = _amp_to_db(_linear_to_mel(np.abs(D), mel_params))
    NS = _normalize(S, mel_params)
    return NS.T.astype(np.float32)


def batch_melspectrogram(y):
    assert len(y.shape) == 2
    batch_size = y.shape[0]
    res = []
    for b in range(batch_size):
        res.append(melspectrogram(y[b]))
    return np.array(res)


def tf_melspectrogram(y):
    l, = y.get_shape().as_list()
    num_frame = int(math.ceil(l / float(FRAME_SHIFT)))
    num_mel = mel_params.num_mel
    mel = tf.py_func(melspectrogram, [y], tf.float32)
    mel.set_shape([num_frame, num_mel])
    return mel


def tf_batch_melspectrogram(y):
    b, l = y.get_shape().as_list()
    num_frame = int(math.ceil(l / float(FRAME_SHIFT)))
    num_mel = mel_params.num_mel
    mel = tf.py_func(batch_melspectrogram, [y], tf.float32)
    mel.set_shape([b, num_frame, num_mel])
    return mel


def _stft(y, mel_params):
    n_fft = (mel_params.num_freq - 1) * 2
    hop_length = int(mel_params.frame_shift_ms / 1000.0 * mel_params.sample_rate)
    win_length = int(mel_params.frame_length_ms / 1000.0 * mel_params.sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _linear_to_mel(spectrogram, mel_params):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(mel_params)
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis(mel_params):
    n_fft = (mel_params.num_freq - 1) * 2
    return librosa.filters.mel(mel_params.sample_rate, n_fft, n_mels=mel_params.num_mel,
                               fmin=mel_params.mel_fmin, fmax=mel_params.mel_fmax)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _preemphasis(x, mel_params):
    return signal.lfilter([1, -mel_params.preemphasis], [1], x)


def _normalize(S, mel_params):
    return np.clip((S - mel_params.min_level_db) / -mel_params.min_level_db, 0, 1)
