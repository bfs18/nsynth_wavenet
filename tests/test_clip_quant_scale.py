import tensorflow as tf
import librosa

from auxilaries import utils


def _clip_quant_scale(x, quant_chann, use_mu_law):
    x = tf.clip_by_value(x, -1.0, 1.0 - 2.0 / quant_chann)
    # Remove the values unseen in data.
    if use_mu_law:
        # suppose x is mu_law encoded audio signal in [-1, 1)
        x_quantized = utils.cast_quantize(x, quant_chann)
        x_scaled = utils.inv_mu_law(x_quantized)
    else:
        # suppose x is real audio signal in [-1, 1)
        x_quantized = utils.cast_quantize(x, quant_chann)
        x_scaled = utils.inv_cast_quantize(x_quantized, quant_chann)
    return x_scaled


audio, _ = librosa.load('test_data/test.wav', sr=16000)
encoded_audio = utils.mu_law_numpy(audio)
real_encoded_audio = utils.inv_cast_quantize_numpy(encoded_audio, 2 ** 8)

ph = tf.placeholder(dtype=tf.float32, shape=[audio.shape[0]])
cqs_audio_mu_law = _clip_quant_scale(ph, 2 ** 8, True)
cqs_audio_real = _clip_quant_scale(ph, 2 ** 16, False)

sess = tf.Session()

inv_mu_law_audio = sess.run(cqs_audio_mu_law, feed_dict={ph: real_encoded_audio})
inv_quant_audio = sess.run(cqs_audio_real, feed_dict={ph: audio})

inv_mu_law_path = 'test_data/test-inv_mu_law.wav'
inv_quant_path = 'test_data/test-inv_quant.wav'
librosa.output.write_wav(inv_mu_law_path, inv_mu_law_audio, sr=16000)
librosa.output.write_wav(inv_quant_path, inv_quant_audio, sr=16000)


audio, _ = librosa.load('test_data/test.wav', sr=16000)
audio_int = utils.cast_quantize_numpy(audio, 2 ** 8)
audio_ = utils.inv_mu_law_numpy(audio_int)
librosa.output.write_wav('test_data/test_inv.wav', audio_, sr=16000)
