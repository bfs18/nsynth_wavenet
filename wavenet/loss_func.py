import tensorflow as tf
from auxilaries import utils


def _log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))


def _log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def mol_log_probs(mol_params, targets, quant_chann, use_log_scales=True):
    """
    log-likelihood for mixture of discretized logistics,
    assumes the data has been rescaled to [-1,1) interval
    (consistent with real audio signal, int signal is in range [0, 65536))
    """
    logit_probs, means, scale_params = tf.split(
        mol_params, num_or_size_splits=3, axis=2)
    if use_log_scales:
        log_scales = tf.maximum(scale_params, -7.0)
        inv_stdv = tf.exp(-log_scales)
    else:
        scales = tf.maximum(scale_params, tf.exp(-7.0))
        inv_stdv = 1. / scales

    nr_mix = mol_params.get_shape().as_list()[-1] // 3
    targets = tf.expand_dims(targets, axis=2) + tf.zeros([1, 1, nr_mix])

    centered_x = targets - means
    plus_in = inv_stdv * (centered_x + 1. / quant_chann)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / quant_chann)
    cdf_min = tf.nn.sigmoid(min_in)

    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)

    cdf_delta = cdf_plus - cdf_min

    # int_x / (quant_chann / 2) - 1. to cast original data to [-1, 1)
    max_val = float(quant_chann - 1)
    max_thres = (max_val - 0.5) / (quant_chann / 2.) - 1.0
    min_thres = 0.5 / (quant_chann / 2.) - 1.0
    log_probs = tf.where(targets < min_thres,
                         log_cdf_plus,
                         tf.where(targets > max_thres,
                                  log_one_minus_cdf_min,
                                  tf.log(tf.maximum(cdf_delta, 1e-12))))

    log_probs = log_probs + _log_prob_from_logits(logit_probs)
    log_sum_probs = _log_sum_exp(log_probs)
    return log_sum_probs


def mol_loss(mol_params, targets, quant_chann):
    log_sum_probs = mol_log_probs(mol_params, targets, quant_chann)
    return -tf.reduce_mean(log_sum_probs)


def ce_loss(logits, targets):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets, name='nll'),
        name='loss')
    return loss


def ce_sample(logits, quant_chann):
    """
    Args:
        logits: [batch_size, number of categories]
    Returns:
        s: [batch_size, 1]
           s is in [-quant_chann / 2, quant_chann / 2)
    """
    dist = tf.distributions.Categorical(logits=logits)
    s = dist.sample(1) - tf.cast(quant_chann / 2, tf.int32)
    s = tf.expand_dims(s[0], axis=1)
    return s


def mol_sample(mol_params, quant_chann, use_log_scales=True):
    """
    Args:
        mol_params: [batch_size, number of mixture * 3]
        quant_chann: quantization channels (2 ** 8 or 2 ** 16)
        use_log_scales: scale parameters is in log scale or linear scale.
    Returns:
        x_quantized: [batch_size, 1],
                     x_quantized is casted to [-quant_chann / 2, quant_chann / 2)
    """
    logit_probs, means, scale_params = tf.split(
        mol_params, num_or_size_splits=3, axis=1)
    nr_mix = mol_params.get_shape().as_list()[1] // 3

    ru = tf.random_uniform(tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5)
    sel = tf.one_hot(
        tf.argmax(logit_probs - tf.log(-tf.log(ru)), axis=1),
        depth=nr_mix, dtype=tf.float32)
    means = tf.reduce_sum(means * sel, axis=1)

    if use_log_scales:
        log_scales = tf.clip_by_value(
            tf.reduce_sum(scale_params * sel, axis=1), -7.0, 7.0)
        scales = tf.exp(log_scales)
    else:
        scales = tf.clip_by_value(
            tf.reduce_sum(scale_params * sel, axis=1), tf.exp(-7.0), tf.exp(7.0))

    ru2 = tf.random_uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + scales * (tf.log(ru2) - tf.log(1. - ru2))
    x = tf.clip_by_value(x, -1., 1. - 2. / quant_chann)
    x_quantized = utils.cast_quantize(x, quant_chann)
    x_quantized = tf.expand_dims(x_quantized, axis=1)
    return x_quantized
