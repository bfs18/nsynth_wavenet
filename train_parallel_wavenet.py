import json
import glob
import tensorflow as tf
import shutil
import os
from argparse import ArgumentParser, Namespace
from wavenet import parallel_wavenet, wavenet
from auxilaries import utils

slim = tf.contrib.slim


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf.logging.set_verbosity(args.log)

    ###
    # get teacher info.
    ###
    teacher_dir = utils.shell_path(args.teacher_dir)
    assert tf.gfile.IsDirectory(teacher_dir)
    json_in_dir = glob.glob(os.path.join(teacher_dir, '*.json'))
    assert len(json_in_dir) == 1
    te_json = json_in_dir[0]
    te_ckpt = tf.train.latest_checkpoint(teacher_dir)
    assert tf.train.checkpoint_exists(te_ckpt)

    with open(te_json, 'rt') as F:
        configs = json.load(F)
    te_hparams = Namespace(**configs)
    teacher = wavenet.Wavenet(te_hparams)

    ###
    # get student info.
    ###
    if args.config is None:
        raise RuntimeError('No config json specified.')
    with open(args.config, 'rt') as F:
        configs = json.load(F)
    st_hparams = Namespace(**configs)
    pwn = parallel_wavenet.ParallelWavenet(st_hparams, teacher, args.train_path)

    logdir = args.logdir
    tf.logging.info('Saving to {}'.format(logdir))

    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, logdir)

    with tf.Graph().as_default():
        total_batch_size = args.total_batch_size
        worker_replicas = args.worker_replicas
        assert total_batch_size % worker_replicas == 0
        worker_batch_size = int(total_batch_size / worker_replicas)

        # Run the Reader on the CPU
        cpu_device = "/job:localhost/replica:0/task:0/cpu:0"
        if args.ps_tasks:
            cpu_device = "/job:worker/cpu:0"

        with tf.device(cpu_device):
            inputs_dict = pwn.get_batch(worker_batch_size)

        with tf.device(
            tf.train.replica_device_setter(ps_tasks=args.ps_tasks,
                                           merge_devices=True)):
            global_step = tf.get_variable(
                "global_step", [],
                tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False)

            lr = tf.constant(pwn.learning_rate_schedule[0])
            for key, value in pwn.learning_rate_schedule.items():
                lr = tf.cond(
                    tf.less(global_step, key), lambda: lr, lambda: tf.constant(value))

            tf.summary.scalar("learning_rate", lr)
            # build the model graph
            ff_dict = pwn.feed_forward(inputs_dict)
            ff_dict.update(inputs_dict)
            loss_dict = pwn.calculate_loss(ff_dict)
            loss = loss_dict['loss']
            tf.summary.scalar("train_loss", loss)
            tf.summary.scalar("kl_loss", loss_dict['kl_loss'])
            tf.summary.scalar("H_Ps", loss_dict['H_Ps'])
            tf.summary.scalar("H_Ps_Pt", loss_dict['H_Ps_Pt'])
            if 'power_loss' in loss_dict:
                tf.summary.scalar('power_loss', loss_dict['power_loss'])

            ###
            # restore teacher
            ###
            te_vars = slim.get_variables_to_restore(exclude=['global_step', 'iaf'])
            # teacher use EMA
            te_vars = {'{}/ExponentialMovingAverage'.format(tv.name[:-2]): tv for tv in te_vars}
            restore_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(te_ckpt, te_vars)

            ###
            # variables to train
            ###
            st_vars = [var for var in tf.trainable_variables() if 'iaf' in var.name]

            ema = tf.train.ExponentialMovingAverage(
                decay=0.9999, num_updates=global_step)
            opt = tf.train.SyncReplicasOptimizer(
                tf.train.AdamOptimizer(lr, epsilon=1e-8),
                worker_replicas,
                total_num_replicas=worker_replicas,
                variable_averages=ema,
                variables_to_average=st_vars)

            train_op = slim.learning.create_train_op(
                total_loss=loss,
                variables_to_train=st_vars,
                optimizer=opt,
                global_step=global_step,
                colocate_gradients_with_ops=True)

            session_config = tf.ConfigProto(allow_soft_placement=True)

            is_chief = (args.task == 0)
            local_init_op = opt.chief_init_op if is_chief else opt.local_step_init_op

            slim.learning.train(
                train_op=train_op,
                logdir=logdir,
                is_chief=is_chief,
                master=args.master,
                number_of_steps=pwn.num_iters,
                global_step=global_step,
                log_every_n_steps=250,
                local_init_op=local_init_op,
                save_interval_secs=3600,
                sync_optimizer=opt,
                session_config=session_config,
                init_fn=restore_init_fn)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Model configuration name")
    parser.add_argument("--train_path", required=True,
                        help="The path to the train tfrecord.")
    parser.add_argument("--teacher_dir", required=True,
                        help="The path saves the teacher config and json.")
    parser.add_argument("--logdir", default="/tmp/nsynth",
                        help="The log directory for this experiment.")
    parser.add_argument("--total_batch_size", default=4, type=int,
                        help="Batch size spread across all sync replicas."
                             "We use a size of 32.")
    parser.add_argument("--master", default="",
                        help="BNS name of the TensorFlow master to use.")
    parser.add_argument("--task", default=0, type=int,
                        help="Task id of the replica running the training.")
    parser.add_argument("--worker_replicas", default=1, type=int,
                        help="Number of replicas. We train with 32.")
    parser.add_argument("--ps_tasks", default=0, type=int,
                        help="Number of tasks in the ps job. If 0 no ps job is "
                             "used. We typically use 11.")
    parser.add_argument("--log", default="INFO",
                        help="The threshold for what messages will be logged."
                             "DEBUG, INFO, WARN, ERROR, or FATAL.")
    parser.add_argument("--gpu_id", default='0',
                        help="gpu device for generation.")
    args = parser.parse_args()
    train(args)
