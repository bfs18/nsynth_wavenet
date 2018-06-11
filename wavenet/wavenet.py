import tensorflow as tf
import numpy as np
from auxilaries import reader, utils
from wavenet import masked, loss_func


DEFAULT_LR_SCHEDULE = {
    0: 2e-4,
    90000: 4e-4 / 3,
    120000: 6e-5,
    150000: 4e-5,
    180000: 2e-5,
    210000: 6e-6,
    240000: 2e-6}


def _deconv_stack(inputs, width, config, name=''):
    b, l, _ = inputs.get_shape().as_list()
    frame_shift = int(np.prod([c[1] for c in config]))
    mel_en = inputs
    for i in range(len(config)):
        fl, s = config[i]
        if name:
            tc_name = '{}/trans_conv_{:d}'.format(name, i + 1)
        else:
            tc_name = 'trans_conv_{:d}'.format(i + 1)
        mel_en = masked.trans_conv1d(
            mel_en,
            num_filters=width,
            filter_length=fl,
            stride=s,
            name=tc_name)
    mel_en.set_shape([b, l * frame_shift, width])
    return mel_en


def _condition(x, cond):
    x_len = x.get_shape().as_list()[1]
    cond_len = cond.get_shape().as_list()[1]
    assert cond_len >= x_len
    trim_len = cond_len - x_len
    left_tl = int(trim_len // 2)
    cond = tf.slice(cond, [0, left_tl, 0], [-1, cond_len - trim_len, -1])
    s = x + cond
    s.set_shape(x.get_shape().as_list())
    return s


def _get_batch(train_path, batch_size, wave_length):
    assert train_path is not None
    data_train = reader.Dataset(train_path, is_training=True)
    return data_train.get_batch(batch_size, length=wave_length)


class Wavenet(object):
    """Wavenet object that helps manage the graph."""

    def __init__(self, hparams, train_path=None):
        self.hparams = hparams
        self.num_iters = hparams.num_iters
        self.learning_rate_schedule = dict(
            getattr(hparams, 'lr_schedule', DEFAULT_LR_SCHEDULE))
        self.train_path = train_path

        # only take the variables used both by
        # feed_forward and calculate_loss as class property.
        self.use_mu_law = self.hparams.use_mu_law
        self.loss_type = self.hparams.loss_type
        if self.use_mu_law:
            self.quant_chann = 2 ** 8
        else:
            self.quant_chann = 2 ** 16
        if self.loss_type == 'ce':
            self.out_width = self.quant_chann
        elif self.loss_type == 'mol':
            mol_mix = self.hparams.mol_mix
            self.out_width = mol_mix * 3
        else:
            raise ValueError('[{}] loss is not supported')

    def get_batch(self, batch_size):
        train_path = self.train_path
        wave_length = self.hparams.wave_length
        return _get_batch(train_path, batch_size, wave_length)

    def deconv_stack(self, mel_inputs):
        mel = mel_inputs['mel']
        deconv_width = self.hparams.deconv_width
        deconv_config = self.hparams.deconv_config  # [[l1, s1], [l2, s2]]

        mel_en = _deconv_stack(mel, deconv_width, deconv_config)
        return {'encoding': mel_en}

    def feed_forward(self, inputs):
        """Build the graph for this configuration.

        Args:
          inputs: A dict of inputs. For training, should contain 'wav'.

        Returns:
          A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
          the 'quantized_input', and whatever metrics we want to track for eval.
        """
        num_stages = self.hparams.num_stages
        num_layers = self.hparams.num_layers
        filter_length = self.hparams.filter_length
        width = self.hparams.width
        skip_width = self.hparams.skip_width
        use_mu_law = self.use_mu_law
        quant_chann = self.quant_chann
        out_width = self.out_width

        ###
        # The Transpose Convolution Stack for mel feature.
        ###
        # wavenet inputs <- trans_conv (l2, s2) <- trans_conv (l1, s1) <- mel_ceps
        # win_len: l1 * s2 + (l2 - s2); win_shift: s1 * s2
        # (l1, s1) = (40, 10), (l2, s2) = (80, 20) is a proper configuration.
        # it is almost consistent with mel analysis frame shift (200) and frame length (800).
        mel = inputs['mel']
        ds_dict = self.deconv_stack({'mel': mel})
        mel_en = ds_dict['encoding']

        ###
        # Encode the source with 8-bit Mu-Law or just use 16-bit signal.
        ###
        x = inputs['wav']
        if use_mu_law:
            x_quantized = utils.mu_law(x)
            x_scaled = tf.cast(x_quantized, tf.float32) / (quant_chann / 2.)
            real_targets = x_scaled
            cate_targets = tf.cast(x_quantized, tf.int32) + tf.cast(quant_chann / 2., tf.int32)
        else:
            x_quantized = utils.cast_quantize(x, quant_chann)
            x_scaled = x
            real_targets = x
            cate_targets = tf.cast(x_quantized, tf.int32) + tf.cast(quant_chann / 2., tf.int32)
        x_scaled = tf.expand_dims(x_scaled, 2)

        ###
        # The WaveNet Decoder.
        ###
        l = masked.shift_right(x_scaled)
        l = masked.conv1d(
            l, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = masked.conv1d(
            l, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2 ** (i % num_stages)
            d = masked.conv1d(
                l,
                num_filters=2 * width,
                filter_length=filter_length,
                dilation=dilation,
                name='dilated_conv_%d' % (i + 1))
            c = masked.conv1d(
                mel_en,
                num_filters=2 * width,
                filter_length=1,
                name='mel_cond_%d' % (i + 1))
            d = _condition(d, c)

            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += masked.conv1d(
                d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
            s += masked.conv1d(
                d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
        c = masked.conv1d(mel_en, num_filters=skip_width, filter_length=1, name='mel_cond_out1')
        s = _condition(s, c)
        s = tf.nn.relu(s)
        # when using mol loss, the model always predicts log_scale, the initializer makes
        # the log_scale in a reasonable small range to speed up convergence.
        final_kernel_init = (tf.truncated_normal_initializer(0.0, 0.01) if self.loss_type == 'mol'
                             else tf.uniform_unit_scaling_initializer(1.0))
        out = masked.conv1d(s, num_filters=out_width, filter_length=1, name='out2',
                            kernel_initializer=final_kernel_init)

        return {'real_targets': real_targets,
                'cate_targets': cate_targets,
                'encoding': mel_en,
                'out_params': out}

    def calculate_loss(self, ff_dict):
        ###
        # Compute the loss.
        ###
        real_targets = ff_dict['real_targets']
        cate_targets = ff_dict['cate_targets']
        out = ff_dict['out_params']
        loss_type = self.loss_type
        if loss_type == 'ce':
            loss = loss_func.ce_loss(out, cate_targets)
        elif loss_type == 'mol':
            quant_chann = self.quant_chann
            loss = loss_func.mol_loss(out, real_targets, quant_chann)
        else:
            raise ValueError('[{}] loss is not supported.'.format(loss_type))
        return {'loss': loss}


class Fastgen(object):
    """Configuration object that helps manage the graph."""

    def __init__(self, hparams, batch_size=2):
        """."""
        self.batch_size = batch_size
        self.hparams = hparams

        self.use_mu_law = self.hparams.use_mu_law
        self.loss_type = self.hparams.loss_type
        if self.use_mu_law:
            self.quant_chann = 2 ** 8
        else:
            self.quant_chann = 2 ** 16
        if self.loss_type == 'ce':
            self.out_width = self.quant_chann
        elif self.loss_type == 'mol':
            mol_mix = self.hparams.mol_mix
            self.out_width = mol_mix * 3
        else:
            raise ValueError('[{}] loss is not supported')

    def sample(self, inputs):
        """Build the graph for this configuration.

        Args:
          inputs: A dict of inputs. For training, should contain 'wav'.

        Returns:
          A dict of outputs that includes the 'predictions',
          'init_ops', the 'push_ops', and the 'quantized_input'.
        """
        batch_size = self.batch_size
        num_stages = self.hparams.num_stages
        num_layers = self.hparams.num_layers
        filter_length = self.hparams.filter_length
        width = self.hparams.width
        skip_width = self.hparams.skip_width
        use_mu_law = self.use_mu_law
        quant_chann = self.quant_chann
        out_width = self.out_width
        deconv_width = self.hparams.deconv_width
        loss_type = self.loss_type

        # mel information is trans_conv_stack output, different from wavenet.feed_forward
        mel_en = inputs['encoding']  # [batch_size, deconv_width]
        mel_en = tf.expand_dims(mel_en, 1)  # [batch_size, 1, deconv_width]

        x = inputs['wav']  # [batch_size, 1]
        if use_mu_law:
            # Encode the source with 8-bit Mu-Law.
            x_quantized = utils.mu_law(x)
            x_scaled = tf.cast(x_quantized, tf.float32) / (quant_chann / 2)
        else:
            x_scaled = x
        x_scaled = tf.expand_dims(x_scaled, 2)  # [batch_size, 1, 1]

        init_ops, push_ops = [], []

        ###
        # The WaveNet Decoder.
        ###
        l = x_scaled
        l, inits, pushs = utils.causal_linear(
            x=l,
            n_inputs=1,
            n_outputs=width,
            name='startconv',
            rate=1,
            batch_size=batch_size,
            filter_length=filter_length)

        for init in inits:
            init_ops.append(init)
        for push in pushs:
            push_ops.append(push)

        # Set up skip connections.
        s = utils.linear(l, width, skip_width, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2 ** (i % num_stages)

            # dilated masked cnn
            d, inits, pushs = utils.causal_linear(
                x=l,
                n_inputs=width,
                n_outputs=width * 2,
                name='dilated_conv_%d' % (i + 1),
                rate=dilation,
                batch_size=batch_size,
                filter_length=filter_length)

            for init in inits:
                init_ops.append(init)
            for push in pushs:
                push_ops.append(push)

            # local conditioning
            d += utils.linear(
                mel_en, deconv_width, width * 2, name='mel_cond_%d' % (i + 1))

            # gated cnn
            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

            # residuals
            l += utils.linear(d, width, width, name='res_%d' % (i + 1))

            # skips
            s += utils.linear(d, width, skip_width, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = (utils.linear(s, skip_width, skip_width, name='out1') +
             utils.linear(mel_en, deconv_width, skip_width, name='mel_cond_out1'))
        s = tf.nn.relu(s)
        out = utils.linear(s, skip_width, out_width, name='out2')  # [batch_size, 1, out_width]

        if loss_type == 'ce':
            sample = loss_func.ce_sample(out, quant_chann)
        elif loss_type == 'mol':
            sample = loss_func.mol_sample(out, quant_chann)
        else:
            raise ValueError('[{}] loss is not supported.'.format(loss_type))

        return {'init_ops': init_ops,
                'push_ops': push_ops,
                'sample': sample}
