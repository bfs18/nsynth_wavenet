import tensorflow as tf

from wavenet import parallel_wavenet, fastgen


def load_parallelgen(hparams, batch_size=1, length=7680, num_mel=80):
    pw = parallel_wavenet.ParallelWavenet(hparams)
    quant_chann = pw.quant_chann
    use_mu_law = pw.use_mu_law
    mel_ph = tf.placeholder(tf.float32, shape=[batch_size, length, num_mel])
    fg_dict = pw.feed_forward({'mel': mel_ph})
    fg_dict['x'] = pw._clip_quant_scale(fg_dict['x'], quant_chann, use_mu_law)
    fg_dict.update({'mel_in': mel_ph})
    return fg_dict


def synthesis(hparams, mel, save_paths, checkpoint_path):
    batch_size, length, num_mel = mel.shape
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        fg_dict = load_parallelgen(hparams, batch_size, length, num_mel)

        # model use ExponentialMovingAverage
        tf_vars = tf.trainable_variables()
        shadow_var_dict = fastgen.get_ema_shadow_dict(tf_vars)
        saver = tf.train.Saver(shadow_var_dict, reshape=True)
        saver.restore(sess, checkpoint_path)

        audio = sess.run(fg_dict['x'],
                         feed_dict={fg_dict['mel_in']: mel})
    fastgen.save_batch(audio, save_paths)
