import tensorflow as tf
import numpy as np

from functools import partial, lru_cache
from wavenet import wavenet, masked, loss_func
from auxilaries import utils, mel_extractor

# log switch for debugging scale value range
DETAIL_LOG = True
MANUAL_FINAL_INIT = True
CLIP = True
LOG_SPEC = False


class ParallelWavenet(object):
    def __init__(self, hparams, teacher=None, train_path=None):
        self.hparams = hparams
        self.num_iters = hparams.num_iters
        self.learning_rate_schedule = dict(
            getattr(hparams, 'lr_schedule', wavenet.DEFAULT_LR_SCHEDULE))
        self.train_path = train_path
        self.use_mu_law = self.hparams.use_mu_law
        self.wave_length = self.hparams.wave_length
        self.use_log_scale = getattr(self.hparams, 'use_log_scale', False)
        self.use_weight_norm = getattr(self.hparams, 'use_weight_norm', False)

        if self.use_mu_law:
            self.quant_chann = 2 ** 8
        else:
            self.quant_chann = 2 ** 16
        self.out_width = 2  # mean, scale

        # generation needs no teacher
        if teacher is not None:
            self.teacher = teacher
            assert teacher.loss_type == 'mol'
            # when using st._clip_quant_scale and te.encode_signal,
            # There is no need for te.use_mu_law and st.use_mu_law to be consistent.
            assert teacher.use_mu_law == self.use_mu_law
            _ = self.stft_mag_mean_std()  # calculated the cached value

            if LOG_SPEC:
                self.stft_feat_fn = lambda x: tf.pow(tf.abs(x), 2.0)
            else:
                self.stft_feat_fn = lambda x: tf.log(tf.maximum(tf.abs(x), 1e-5))

    def get_batch(self, batch_size):
        train_path = self.train_path
        wave_length = self.wave_length
        return wavenet._get_batch(train_path, batch_size, wave_length)

    @staticmethod
    def _logistic_0_1(batch_size, length):
        # logistic(0, 1) random variables
        ru = tf.random_uniform(
            [batch_size, length], minval=1e-5, maxval=1. - 1e-5)
        rl = tf.log(ru) - tf.log(1. - ru)
        return rl

    @property
    def final_kernel_mean(self):
        # The final kernel_initializer keeps the scale in a reasonable small range.
        # Tuned for LJSpeech
        if self.use_mu_law:
            normal_mean = -0.01 if self.use_log_scale else -0.001
        else:
            normal_mean = -0.03 if self.use_log_scale else -0.01
        return normal_mean

    @property
    def final_bias(self):
        if self.use_log_scale:
            return -0.8
        else:
            return -0.3

    def _create_iaf(self, inputs, iaf_idx, init):
        num_stages = self.hparams.num_stages
        num_layers = self.hparams.num_iaf_layers[iaf_idx]
        filter_length = self.hparams.filter_length
        width = self.hparams.width
        out_width = self.out_width
        deconv_width = self.hparams.deconv_width
        deconv_config = self.hparams.deconv_config  # [[l1, s1], [l2, s2]]
        use_log_scale = self.use_log_scale
        use_weight_norm = self.use_weight_norm
        fk_mean = self.final_kernel_mean
        # in parallel wavenet paper, gate width is the same with residual width
        # not double of that.
        # gate_width = 2 * width
        gate_width = width

        if MANUAL_FINAL_INIT:
            final_init = False
            final_bias = self.final_bias
            if init:
                tf.logging.info('manually initialize the weights for ' 
                                'the final layer of flow_{}.'.format(iaf_idx))
        else:
            final_init = init
            final_bias = 0.0

        mel = inputs['mel']
        x = inputs['x']

        iaf_name = 'iaf_{:d}'.format(iaf_idx + 1)

        mel_en = wavenet._deconv_stack(
            mel, deconv_width, deconv_config, name=iaf_name,
            use_weight_norm=use_weight_norm, init=init)

        l = masked.shift_right(x)
        l = masked.conv1d(l, num_filters=width, filter_length=filter_length,
                          name='{}/start_conv'.format(iaf_name),
                          use_weight_norm=use_weight_norm, init=init)

        for i in range(num_layers):
            dilation = 2 ** (i % num_stages)
            d = masked.conv1d(
                l,
                num_filters=gate_width,
                filter_length=filter_length,
                dilation=dilation,
                name='{}/dilated_conv_{:d}'.format(iaf_name, i + 1),
                use_weight_norm=use_weight_norm,
                init=init)
            c = masked.conv1d(
                mel_en,
                num_filters=gate_width,
                filter_length=1,
                name='{}/mel_cond_{:d}'.format(iaf_name, i + 1),
                use_weight_norm=use_weight_norm,
                init=init)
            d = wavenet._condition(d, c)

            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += masked.conv1d(d, num_filters=width, filter_length=1,
                               name='{}/res_{:d}'.format(iaf_name, i + 1),
                               use_weight_norm=use_weight_norm, init=init)

        l = tf.nn.relu(l)
        l = masked.conv1d(l, num_filters=width, filter_length=1,
                          name='{}/out1'.format(iaf_name),
                          use_weight_norm=use_weight_norm, init=init)
        c = masked.conv1d(mel_en, num_filters=width, filter_length=1,
                          name='{}/mel_cond_out1'.format(iaf_name),
                          use_weight_norm=use_weight_norm, init=init)
        l = wavenet._condition(l, c)
        l = tf.nn.relu(l)

        mean = masked.conv1d(
            l, num_filters=out_width // 2, filter_length=1,
            name='{}/out2_mean'.format(iaf_name),
            use_weight_norm=use_weight_norm, init=final_init)
        scale_params = masked.conv1d(
            l, num_filters=out_width // 2, filter_length=1,
            name='{}/out2_scale'.format(iaf_name),
            use_weight_norm=use_weight_norm, init=final_init,
            biases_initializer=tf.constant_initializer(final_bias))

        if use_log_scale:
            log_scale = tf.clip_by_value(scale_params, -9.0, 7.0)
            scale = tf.exp(log_scale)
        else:
            scale_params = tf.nn.softplus(scale_params)
            scale = tf.clip_by_value(scale_params, tf.exp(-9.0), tf.exp(7.0))
            log_scale = tf.log(scale)
        new_x = x * scale + mean

        if DETAIL_LOG:
            tf.summary.scalar('scale_{}'.format(iaf_idx), tf.reduce_mean(scale))
            tf.summary.scalar('log_scale_{}'.format(iaf_idx), tf.reduce_mean(log_scale))
            tf.summary.scalar('mean_{}'.format(iaf_idx), tf.reduce_mean(mean))

        return {'x': new_x,
                'mean': mean,
                'scale': scale,
                'log_scale': log_scale}

    def feed_forward(self, inputs, init=False):
        num_stages = self.hparams.num_stages
        num_iafs = len(self.hparams.num_iaf_layers)
        deconv_config = self.hparams.deconv_config  # [[l1, s1], [l2, s2]]
        frame_shift = int(np.prod([dc[1] for dc in deconv_config]))
        max_dilation = 2 ** (num_stages - 1)

        mel = inputs['mel']

        batch_size, num_frames, _ = mel.get_shape().as_list()
        # length must be a multiple of dilation length
        length = (num_frames * frame_shift // max_dilation) * max_dilation
        x = self._logistic_0_1(batch_size, length)

        iaf_x = tf.expand_dims(x, axis=2)
        mean_tot, scale_tot, log_scale_tot = 0., 1., 0.
        for iaf_idx in range(num_iafs):
            iaf_dict = self._create_iaf({'mel': mel, 'x': iaf_x}, iaf_idx, init=init)
            iaf_x = iaf_dict['x']
            scale = iaf_dict['scale']
            log_scale = iaf_dict['log_scale']
            mean_tot = iaf_dict['mean'] + mean_tot * scale
            scale_tot *= scale
            log_scale_tot += log_scale

        mean_tot = tf.squeeze(mean_tot, axis=2)
        scale_tot = tf.squeeze(tf.minimum(scale_tot, tf.exp(7.0)), axis=2)
        log_scale_tot = tf.squeeze(tf.minimum(log_scale_tot, 7.0), axis=2)
        # new_x = tf.squeeze(iaf_x, axis=2)
        new_x = x * scale_tot + mean_tot

        if DETAIL_LOG:
            tf.summary.scalar('new_x', tf.reduce_mean(new_x))
            tf.summary.scalar('new_x_std', utils.reduce_std(new_x))
            tf.summary.scalar('new_x_abs', tf.reduce_mean(tf.abs(new_x)))
            tf.summary.scalar('new_x_abs_std', utils.reduce_std(tf.abs(new_x)))
            tf.summary.scalar('mean_tot', tf.reduce_mean(mean_tot))
            tf.summary.scalar('scale_tot', tf.reduce_mean(scale_tot))
            tf.summary.scalar('log_scale_tot', tf.reduce_mean(log_scale_tot))

        return {'x': new_x,
                'mean_tot': mean_tot,
                'scale_tot': scale_tot,
                'log_scale_tot': log_scale_tot,
                'rand_input': x}

    @staticmethod
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

    def kl_loss(self, ff_dict, num_samples=100):
        teacher = self.teacher
        quant_chann = self.quant_chann
        use_mu_law = self.use_mu_law

        mel = ff_dict['mel']
        x = ff_dict['x']
        mean = ff_dict['mean_tot']
        scale = ff_dict['scale_tot']
        log_scale = ff_dict['log_scale_tot']

        batch_size, length = x.get_shape().as_list()

        rl = self._logistic_0_1(batch_size * num_samples, length)
        mean = utils.tf_repeat(mean, [num_samples, 1])
        scale = utils.tf_repeat(scale, [num_samples, 1])
        # (x_i|x_<i), x given x_previous from student
        x_xp = rl * scale + mean

        if CLIP:
            # clip x and x_xp to real audio range [-1.0, 1.0)
            # if use_mu_law = True,
            # take iaf output as mu_law encoded real audio signal.
            x_cqs = self._clip_quant_scale(x, quant_chann, use_mu_law)
            x_xp_cqs = self._clip_quant_scale(x_xp, quant_chann, use_mu_law)
            x_scaled = teacher.encode_signal({'wav': x_cqs})['wav_scaled']
            x_xp_scaled = teacher.encode_signal({'wav': x_xp_cqs})['wav_scaled']
        else:
            # remove clip_quant_scale
            x_scaled = x
            x_xp_scaled = x_xp

        wn_ff_dict = teacher.feed_forward({'wav_scaled': x_scaled,
                                           'mel': mel})
        te_mol = wn_ff_dict['out_params']
        te_mol = utils.tf_repeat(te_mol, [num_samples, 1, 1])

        # teacher always use log_scale, so use_log_scale of
        # loss_func.mol_log_probs is set to default value True.
        log_te_probs = loss_func.mol_log_probs(
            te_mol, x_xp_scaled, quant_chann)
        # H_Ps_Pt for batch * length
        H_Ps_Pt_bl = -tf.reduce_mean(
            tf.reshape(log_te_probs, [batch_size, num_samples, length]),
            axis=1)

        H_Ps = tf.reduce_mean(log_scale) + 2
        H_Ps_Pt = tf.reduce_mean(H_Ps_Pt_bl)
        kl_loss = H_Ps_Pt - H_Ps

        return {'kl_loss': kl_loss,
                'H_Ps': H_Ps,
                'H_Ps_Pt': H_Ps_Pt}

    @staticmethod
    def _trim(x, trim_len):
        x_len = x.get_shape().as_list()[1]
        left_tl = int(trim_len // 2)
        trim_wav = tf.slice(x, [0, left_tl], [-1, x_len - trim_len])
        return trim_wav

    @lru_cache(maxsize=1)
    def stft_mag_mean_std(self):
        return mel_extractor.spec_mag_mean_std(
            self.train_path, feat_fn=self.stft_feat_fn)

    def power_loss(self, wav_dict):
        mean, std = self.stft_mag_mean_std()
        feat_fn = self.stft_feat_fn

        pred_wav = wav_dict['x']
        orig_wav = wav_dict['wav']
        pred_len = pred_wav.get_shape().as_list()[1]
        orig_len = orig_wav.get_shape().as_list()[1]
        # crop longer wave
        if pred_len > orig_len:
            pred_wav = self._trim(pred_wav, pred_len - orig_len)
        elif pred_len < orig_len:
            orig_wav = self._trim(orig_wav, orig_len - pred_len)

        orig_stft = mel_extractor._tf_stft(orig_wav)
        pred_stft = mel_extractor._tf_stft(pred_wav)

        def _norm(feat):
            return (feat - tf.cast(mean, tf.float32)) / tf.cast(std, tf.float32)

        orig_feat = _norm(feat_fn(orig_stft))
        pred_feat = _norm(feat_fn(pred_stft))

        power_loss = 0.5 * tf.reduce_mean(
            tf.squared_difference(orig_feat, pred_feat))
        return {'power_loss': power_loss}

    def calculate_loss(self, ff_dict):
        plf = self.hparams.power_loss_factor
        num_samples = self.hparams.num_samples
        loss_dict = self.kl_loss(ff_dict, num_samples)
        loss = loss_dict['kl_loss']
        if plf > 0.0:
            pl_dict = self.power_loss(ff_dict)
            loss += plf * pl_dict['power_loss']
            loss_dict.update(pl_dict)
        loss_dict.update({'loss': loss})
        return loss_dict
