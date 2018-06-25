import tensorflow as tf
import numpy as np
from functools import reduce
from wavenet.masked import conv1d


def test_moments():
    batch_size = 8
    filter_len = 3
    in_channels = 256
    out_channels = 256
    seq_len = 1024
    stride = 1

    ph = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1, seq_len, in_channels))
    c = tf.layers.conv2d(ph,
                         filters=out_channels,
                         kernel_size=(1, filter_len),
                         padding='SAME',
                         kernel_initializer=tf.random_normal_initializer(0, 0.05),
                         bias_initializer=tf.constant_initializer(0.))
    m, v = tf.nn.moments(c, [0, 1, 2])
    scale = 1.0 / tf.sqrt(v + 1e-10)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # data = np.random.uniform(-0.5, 0.5, size=(batch_size, 1, seq_len, in_channels))
    # data = np.random.logistic(0., 0.09, size=(batch_size, 1, seq_len, in_channels))
    data = np.random.normal(0., 0.45, size=(batch_size, 1, seq_len, in_channels))

    c_val, m_val, v_val, scale_val = sess.run([c, m, v, scale], feed_dict={ph: data})

    print(scale_val.max())
    print(scale_val.min())
    print(scale_val.mean())


######################################
# parallel wavenet use_log_scale=False
# INFO:tensorflow:initial new_x.m -0.02581, new_x.std 2.31571, new_x.min -36.26904, new_x.max 27.08098
# INFO:tensorflow:initial mean.m -0.02271, mean.std 1.90750, mean.min -19.27777, mean.max 17.76148
# INFO:tensorflow:initial scale.m 0.38296, scale.std 0.61160, scale.min 0.00130, scale.max 9.61347
#
# parallel wavenet use_log_scale=True
# INFO:tensorflow:initial new_x.m 0.72527, new_x.std 78.91548, new_x.min -2095.91919, new_x.max 6462.81934
# INFO:tensorflow:initial mean.m 0.71400, mean.std 38.13536, mean.min -616.65656, mean.max 4382.88379
# INFO:tensorflow:initial scale.m 6.98023, scale.std 35.60387, scale.min 0.00025, scale.max 1096.63318
#
# wavenet use_log_scale=True
# INFO:tensorflow:initial mean.m -0.00000, mean.std 1.00000, mean.min -4.55415, mean.max 4.90998
# INFO:tensorflow:initial scale.m 1.64126, scale.std 2.10046, scale.min 0.00920, scale.max 58.31658
#
# After data dependent initialization for weight normalization, the output of conv/deconv is approximately
# normally distributed with mean 0.0 and std 1.0.
# This explains why log_scale is feasible for wavenet but is not feasible for parallel wavenet.
# if use log_scale, scale_tot for parallel wavenet is too large and needs much more deliberated
# initialization weights.
######################################

def init_logging(array, array_name):
    print('initial {0}.m {1:.5f}, {0}.std {2:.5f}, '
          '{0}.min {3:.5f}, {0}.max {4:.5f}'.format(
            array_name, array.mean(), array.std(),
            array.min(), array.max()))


def softplus(array):
    return np.log(1.0 + np.exp(array))


def get_scale(scale_params, use_log_scale=False):
    if use_log_scale:
        log_scale = np.clip(scale_params, -9.0, 7.0)
        scale = np.exp(log_scale)
    else:
        scale_params = softplus(scale_params)
        scale = np.clip(scale_params, np.exp(-9.0), np.exp(7.0))
    return scale


def test_wavenet_scale():
    shape = (7680 * 10)
    data = np.random.normal(0., 1., size=shape)
    print('=====wavenet scale use_log_scale=True=====')
    scale = get_scale(data, use_log_scale=True)
    init_logging(scale, 'scale')
    print('INFO:tensorflow:initial scale.m 1.64126, scale.std 2.10046, scale.min 0.00920, scale.max 58.31658')
    print('=====wavenet scale use_log_scaleFalse======')
    scale = get_scale(data, use_log_scale=False)
    init_logging(scale, 'scale')
    print('Not used in experiment')


def test_parallel_wavenet_scale():
    num_iafs = 4
    shape = (7680 * 10)
    datas = [np.random.normal(0., 1., size=shape) for _ in range(num_iafs)]
    print('=====parallel wavenet scale use_log_scale=True=====')
    scales = [get_scale(data, use_log_scale=True) for data in datas]
    scale_tot = reduce(np.multiply, scales)
    init_logging(scale_tot, 'scale')
    print('INFO:tensorflow:initial scale.m 6.98023, scale.std 35.60387, scale.min 0.00025, scale.max 1096.63318')
    print('=====parallel wavenet scale use_log_scale=False=====')
    scales = [get_scale(data, use_log_scale=False) for data in datas]
    scale_tot = reduce(np.multiply, scales)
    init_logging(scale_tot, 'scale')
    print('INFO:tensorflow:initial scale.m 0.38296, scale.std 0.61160, scale.min 0.00130, scale.max 9.61347')


def test_normal_scale():
    batch_size = 8
    filter_len = 3
    in_channels = 64
    out_channels = 10
    seq_len = 7680
    num_iafs = 4

    ph = tf.placeholder(dtype=tf.float32, shape=(batch_size, seq_len, in_channels))
    mean = conv1d(tf.nn.relu(ph),
                  num_filters=out_channels,
                  filter_length=filter_len,
                  name='mean_conv',
                  # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
                  biases_initializer=tf.constant_initializer(0.0),
                  use_weight_norm=True)
    scale_param = conv1d(tf.nn.relu(ph),
                         num_filters=out_channels,
                         filter_length=filter_len,
                         name='scale_conv',
                         # kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
                         biases_initializer=tf.constant_initializer(-0.8),
                         use_weight_norm=True)

    mean_tot = 0.
    scale_tot = 1.
    for i in range(num_iafs):
        data = np.random.normal(0., 1., size=(batch_size, seq_len, in_channels))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        mean_val, scale_param_val = sess.run([mean, scale_param], feed_dict={ph: data})
        scale_val = get_scale(scale_param_val, use_log_scale=True)
        mean_tot = mean_val + scale_val * mean_tot
        scale_tot *= scale_val
    init_logging(mean_tot, 'mean')
    init_logging(scale_tot, 'scale')


if __name__ == '__main__':
    test_normal_scale()
