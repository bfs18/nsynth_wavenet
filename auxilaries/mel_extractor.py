import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
import math
from auxilaries import reader, utils
from functools import lru_cache

from scipy import signal


_mel_basis = None


mel_params = tf.contrib.training.HParams(
    sample_rate=16000,
    num_freq=1025,
    num_mel=80,
    frame_shift_ms=12.5,
    frame_length_ms=50,
    preemphasis=0.97,
    mel_min_level_db=-100,
    spec_min_level_db=-140,
    spec_ref_level_db=40,
    mel_fmin=125,
    mel_fmax=7600)

PRIORITY_FREQ = int(3000 / (mel_params.sample_rate * 0.5) * mel_params.num_freq)
FRAME_SHIFT = int(mel_params.frame_shift_ms * mel_params.sample_rate / 1000.)


def melspectrogram(y):
    D = _stft(y, mel_params)
    S = _amp_to_db(_linear_to_mel(np.abs(D), mel_params))
    NS = _normalize(S, mel_params.mel_min_level_db)
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


def _normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


######################
# extend for power loss
######################
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def _tf_amp_to_db(x):
    return 20 * log10(tf.maximum(1e-5, x))


def _tf_normalize(S, min_level_db):
    min_level_db = tf.cast(min_level_db, np.float32)
    return tf.clip_by_value((S - min_level_db) / -min_level_db, 0., 1.)


def _tf_stft(y):
    frame_shift = int(mel_params.frame_shift_ms * mel_params.sample_rate / 1000)
    frame_length = int(mel_params.frame_length_ms * mel_params.sample_rate / 1000)
    fft_length = int(2 * (mel_params.num_freq - 1))
    tf_stft = tf.contrib.signal.stft(
        y,
        frame_length=frame_length,
        frame_step=frame_shift,
        fft_length=fft_length,
        pad_end=True)
    return tf_stft


def tf_spectrogram(y):
    D = _tf_stft(y)
    S = _tf_amp_to_db(tf.abs(D)) - mel_params.spec_ref_level_db
    NS = _tf_normalize(S, mel_params.spec_min_level_db)
    return NS


def spec_feat_mean_std(train_path, feat_fn=lambda x: tf.pow(tf.abs(x), 2.0)):
    local_graph = tf.Graph()
    with local_graph.as_default():
        input_vals = reader.get_init_batch(
            train_path, batch_size=4096, seq_len=7680, first_n=10000)['wav']
        ph = tf.placeholder(dtype=np.float32, shape=[4096, 7680])
        feat = feat_fn(_tf_stft(ph))

    tf.logging.info('Calculating mean and std for stft feat.')
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config, graph=local_graph)
    feat_val = sess.run(feat, feed_dict={ph: input_vals})
    mean_val = np.mean(feat_val, axis=(0, 1))
    std_val = np.std(feat_val, axis=(0, 1))
    tf.logging.info('Done calculating mean and std for stft feat.')

    return mean_val, std_val


@lru_cache(maxsize=1)
def _tf_build_mel_basis(graph):
    # graph is only used to index cache
    tf.logging.info('Initiate tensorflow mel weight matrix for {}'.format(graph))
    return tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_params.num_mel,
        num_spectrogram_bins=mel_params.num_freq,
        sample_rate=mel_params.sample_rate,
        lower_edge_hertz=mel_params.mel_fmin,
        upper_edge_hertz=mel_params.mel_fmax)


def tf_melspectrogram2(spec):
    graph = tf.get_default_graph()
    mel_basis = _tf_build_mel_basis(graph)
    mel_spec = tf.tensordot(spec, mel_basis, 1)
    mel_spec.set_shape(spec.shape[:-1].concatenate(mel_basis.shape[-1:]))
    return mel_spec
