import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from scipy.io import wavfile
from auxilaries import mel_extractor, utils
from wavenet import wavenet


# ExponentialMovingAverage shadow var dict
def get_ema_shadow_dict(tf_vars):
    return {'{}/ExponentialMovingAverage'.format(var.name[:-2]): var
            for var in tf_vars}


def load_batch(files, sample_length=64000):
    """Load a batch of data from either .wav or .npy files.

    Args:
      files: A list of filepaths to .wav or .npy files
      sample_length: Maximum sample length

    Returns:
      batch_data: A padded array of audio or embeddings [batch, length, (dims)]
    """
    batch_data = []
    max_length = 0
    is_npy = (os.path.splitext(files[0])[1] == ".npy")
    # Load the data
    for f in files:
        if is_npy:
            data = np.load(f)
            batch_data.append(data)
        else:
            data = utils.load_audio(f, sample_length, sr=16000)
            batch_data.append(data)
        if data.shape[0] > max_length:
            max_length = data.shape[0]
    # Add padding
    for i, data in enumerate(batch_data):
        if data.shape[0] < max_length:
            if is_npy:
                padded = np.zeros([max_length, +data.shape[1]])
                padded[:data.shape[0], :] = data
            else:
                padded = np.zeros([max_length])
                padded[:data.shape[0]] = data
            batch_data[i] = padded
    # Return arrays
    batch_data = np.vstack(batch_data)
    return batch_data


def save_batch(batch_audio, batch_save_paths):
    for audio, name in zip(batch_audio, batch_save_paths):
        tf.logging.info("Saving: %s" % name)
        wavfile.write(name, 16000, audio)


def load_deconv_stack(hparams, batch_size=1, mel_length=320, num_mel=80):
    wn = wavenet.Wavenet(hparams)
    mel_ph = tf.placeholder(tf.float32, shape=[batch_size, mel_length, num_mel])
    ds_dict = wn.deconv_stack({'mel': mel_ph})
    ds_dict.update({'mel_in': mel_ph})
    return ds_dict


def encode(hparams, wav_data, checkpoint_path):
    if wav_data.ndim == 1:
        wav_data = np.expand_dims(wav_data, 0)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        mel_val = mel_extractor.batch_melspectrogram(wav_data)
        batch_size, mel_length, num_mel = mel_val.shape
        ds_dict = load_deconv_stack(hparams, batch_size, mel_length, num_mel)

        # model use ExponentialMovingAverage
        tf_vars = tf.trainable_variables()
        shadow_var_dict = get_ema_shadow_dict(tf_vars)
        saver = tf.train.Saver(shadow_var_dict)
        saver.restore(sess, checkpoint_path)

        encoding = sess.run(
            ds_dict['encoding'], feed_dict={ds_dict['mel_in']: mel_val})
    return encoding


def load_fastgen(hparams, batch_size=1):
    deconv_width = hparams.deconv_width
    fg = wavenet.Fastgen(hparams, batch_size)
    wav_ph = tf.placeholder(tf.float32, shape=[batch_size, 1])
    mel_en_ph = tf.placeholder(tf.float32, shape=[batch_size, deconv_width])
    fg_dict = fg.sample({'wav': wav_ph, 'encoding': mel_en_ph})
    fg_dict.update({'wav_in': wav_ph, 'encoding_in': mel_en_ph})
    return fg_dict


def synthesis(hparams, mel_encoding, save_paths, checkpoint_path):
    use_mu_law = hparams.use_mu_law
    if use_mu_law:
        quant_chann = 2 ** 8
    else:
        quant_chann = 2 ** 16

    batch_size = mel_encoding.shape[0]
    length = mel_encoding.shape[1]

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        fg_dict = load_fastgen(hparams, batch_size)

        # model use ExponentialMovingAverage
        tf_vars = tf.trainable_variables()
        shadow_var_dict = get_ema_shadow_dict(tf_vars)
        saver = tf.train.Saver(shadow_var_dict)
        saver.restore(sess, checkpoint_path)

        # initialize queues w/ 0s
        sess.run(fg_dict['init_ops'])

        # Regenerate the audio file sample by sample
        audio_batch = np.zeros((batch_size, length), dtype=np.float32)
        audio = np.zeros([batch_size, 1])

        for sample_i in tqdm(range(length)):
            mel_en = mel_encoding[:, sample_i, :]
            int_audio, _ = sess.run(
                [fg_dict['sample'], fg_dict['push_ops']],
                feed_dict={fg_dict['wav_in']: audio,
                           fg_dict['encoding_in']: mel_en})

            # int_audio in [-qc / 2, qc / 2)
            if use_mu_law:
                audio = utils.inv_mu_law_numpy(int_audio)
            else:
                audio = utils.inv_cast_quantize_numpy(int_audio, quant_chann)
            audio_batch[:, sample_i] = audio[:, 0]
    save_batch(audio_batch, save_paths)
