import json
import numpy as np
import tensorflow as tf
import shutil
import os
import glob
from argparse import ArgumentParser, Namespace
from wavenet import wavenet
from deployment import model_deploy
from auxilaries import reader, config_str

slim = tf.contrib.slim
EXP_TAG = ''


def _init_logging(array, array_name):
    tf.logging.info(
        'initial {0}.m {1:.5f}, {0}.std {2:.5f}, '
        '{0}.min {3:.5f}, {0}.max {4:.5f}'.format(
            array_name, array.mean(), array.std(),
            array.min(), array.max()))


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf.logging.set_verbosity(args.log)
    clone_on_cpu = args.gpu_id == ''
    num_clones = len(args.gpu_id.split(','))

    if args.log_root:
        if args.config is None:
            raise RuntimeError('No config json specified.')
        tf.logging.info('using config form {}'.format(args.config))
        with open(args.config, 'rt') as F:
            configs = json.load(F)
        hparams = Namespace(**configs)
        logdir_name = config_str.get_config_time_str(hparams, 'wavenet', EXP_TAG)
        logdir = os.path.join(args.log_root, logdir_name)
        os.makedirs(logdir, exist_ok=True)
        shutil.copy(args.config, logdir)
    else:
        logdir = args.logdir
        config_json = glob.glob(os.path.join(logdir, '*.json'))[0]
        tf.logging.info('using config form {}'.format(config_json))
        with open(config_json, 'rt') as F:
            configs = json.load(F)
        hparams = Namespace(**configs)
    tf.logging.info('Saving to {}'.format(logdir))

    wn = wavenet.Wavenet(hparams, os.path.abspath(os.path.expanduser(args.train_path)))

    def _data_dep_init():
        # slim.learning.train runs init_fn earlier than start_queue_runner
        # so the the function got dead locker if use the `input_dict` in L76 as input
        inputs_val = reader.get_init_batch(
            wn.train_path, batch_size=args.total_batch_size, seq_len=wn.wave_length)
        wave_data = inputs_val['wav']
        mel_data = inputs_val['mel']

        _inputs_dict = {
            'wav': tf.placeholder(dtype=tf.float32, shape=wave_data.shape),
            'mel': tf.placeholder(dtype=tf.float32, shape=mel_data.shape)}

        encode_dict = wn.encode_signal(_inputs_dict)
        _inputs_dict.update(encode_dict)
        init_ff_dict = wn.feed_forward(_inputs_dict, init=True)

        def callback(session):
            tf.logging.info('Calculate initial statistics.')
            init_out = session.run(
                init_ff_dict, feed_dict={_inputs_dict['wav']: wave_data,
                                         _inputs_dict['mel']: mel_data})
            init_out_params = init_out['out_params']
            if wn.loss_type == 'mol':
                _, mean, log_scale = np.split(init_out_params, 3, axis=2)
                scale = np.exp(np.maximum(log_scale, -7.0))
                _init_logging(mean, 'mean')
                _init_logging(scale, 'scale')
            elif wn.loss_type == 'gauss':
                mean, log_std = np.split(init_out_params, 2, axis=2)
                std = np.exp(np.maximum(log_std, -7.0))
                _init_logging(mean, 'mean')
                _init_logging(std, 'std')
            tf.logging.info('Done Calculate initial statistics.')
        return callback

    def _model_fn(_inputs_dict):
        encode_dict = wn.encode_signal(_inputs_dict)
        _inputs_dict.update(encode_dict)
        ff_dict = wn.feed_forward(_inputs_dict)
        ff_dict.update(encode_dict)
        loss_dict = wn.calculate_loss(ff_dict)
        loss = loss_dict['loss']
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    with tf.Graph().as_default():
        total_batch_size = args.total_batch_size
        assert total_batch_size % num_clones == 0
        clone_batch_size = int(total_batch_size / num_clones)

        deploy_config = model_deploy.DeploymentConfig(
            num_clones=num_clones, clone_on_cpu=clone_on_cpu,
            num_ps_tasks=0,
            worker_job_name='localhost', ps_job_name='localhost')

        with tf.device(deploy_config.inputs_device()):
            inputs_dict = wn.get_batch(clone_batch_size)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, _model_fn, [inputs_dict])
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        summaries.update(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        with tf.device(deploy_config.variables_device()):
            global_step = tf.get_variable(
                "global_step", [],
                tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False)

        with tf.device(deploy_config.optimizer_device()):
            lr = tf.constant(wn.learning_rate_schedule[0])
            for key, value in wn.learning_rate_schedule.items():
                lr = tf.cond(
                    tf.less(global_step, key), lambda: lr, lambda: tf.constant(value))
            summaries.add(tf.summary.scalar("learning_rate", lr))

            optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-8)
            ema = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates=global_step)

            loss, clone_grads_vars = model_deploy.optimize_clones(
                clones, optimizer, var_list=tf.trainable_variables())
            update_ops.append(
                optimizer.apply_gradients(clone_grads_vars, global_step=global_step))
            update_ops.append(ema.apply(tf.trainable_variables()))

            summaries.add(tf.summary.scalar("train_loss", loss))

            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(loss, name='train_op')

        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        data_dep_init_fn = _data_dep_init()

        slim.learning.train(
            train_tensor,
            logdir=logdir,
            number_of_steps=wn.num_iters,
            summary_op=summary_op,
            global_step=global_step,
            log_every_n_steps=100,
            save_summaries_secs=600,
            save_interval_secs=3600,
            session_config=session_config,
            init_fn=data_dep_init_fn)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=False,
                        help="Model configuration name")
    parser.add_argument("--train_path", required=True,
                        help="The path to the train tfrecord.")
    parser.add_argument("--logdir", default="/tmp/nsynth",
                        help="The log directory for this experiment.")
    parser.add_argument("--log_root", default="",
                        help="The log directory for this experiment.")
    parser.add_argument("--total_batch_size", default=4, type=int,
                        help="Batch size spread across all sync replicas."
                             "We use a size of 32.")
    parser.add_argument("--log", default="INFO",
                        help="The threshold for what messages will be logged."
                             "DEBUG, INFO, WARN, ERROR, or FATAL.")
    parser.add_argument("--gpu_id", default='0',
                        help="gpu device for generation, "
                             "cpu e.g. \"\", single gpu e.g. \"0\", multiple gpu e.g. \"1,3,5\"")
    args = parser.parse_args()
    train(args)
