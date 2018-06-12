import json
import tensorflow as tf
import shutil
import os
from argparse import ArgumentParser, Namespace
from wavenet import wavenet
from deployment import model_deploy

slim = tf.contrib.slim


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf.logging.set_verbosity(args.log)
    clone_on_cpu = args.gpu_id == ''
    num_clones = len(args.gpu_id.split(','))

    if args.config is None:
        raise RuntimeError('No config json specified.')
    with open(args.config, 'rt') as F:
        configs = json.load(F)
    hparams = Namespace(**configs)

    wn = wavenet.Wavenet(hparams, args.train_path)

    def _model_fn(_inputs_dict):
        encode_dict = wn.encode_signal(_inputs_dict)
        _inputs_dict.update(encode_dict)
        ff_dict = wn.feed_forward(_inputs_dict)
        ff_dict.update(encode_dict)
        loss_dict = wn.calculate_loss(ff_dict)
        loss = loss_dict['loss']
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    logdir = args.logdir
    tf.logging.info('Saving to {}'.format(logdir))

    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, logdir)

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

        slim.learning.train(
            train_tensor,
            logdir=logdir,
            number_of_steps=wn.num_iters,
            summary_op=summary_op,
            global_step=global_step,
            log_every_n_steps=100,
            save_summaries_secs=600,
            save_interval_secs=3600,
            session_config=session_config,)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Model configuration name")
    parser.add_argument("--train_path", required=True,
                        help="The path to the train tfrecord.")
    parser.add_argument("--logdir", default="/tmp/nsynth",
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
