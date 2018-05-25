import json
import tensorflow as tf
import shutil
import os
from argparse import ArgumentParser, Namespace
from wavenet import wavenet

slim = tf.contrib.slim


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf.logging.set_verbosity(args.log)

    if args.config is None:
        raise RuntimeError('No config json specified.')
    with open(args.config, 'rt') as F:
        configs = json.load(F)
    hparams = Namespace(**configs)

    wn = wavenet.Wavenet(hparams, args.train_path)

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
            inputs_dict = wn.get_batch(worker_batch_size)

        with tf.device(
            tf.train.replica_device_setter(ps_tasks=args.ps_tasks,
                                           merge_devices=True)):
            global_step = tf.get_variable(
                "global_step", [],
                tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False)

            lr = tf.constant(wn.learning_rate_schedule[0])
            for key, value in wn.learning_rate_schedule.items():
                lr = tf.cond(
                    tf.less(global_step, key), lambda: lr, lambda: tf.constant(value))
            tf.summary.scalar("learning_rate", lr)

            # build the model graph
            ff_dict = wn.feed_forward(inputs_dict)
            loss_dict = wn.calculate_loss(ff_dict)
            loss = loss_dict['loss']
            tf.summary.scalar("train_loss", loss)

            ema = tf.train.ExponentialMovingAverage(
                decay=0.9999, num_updates=global_step)
            opt = tf.train.SyncReplicasOptimizer(
                tf.train.AdamOptimizer(lr, epsilon=1e-8),
                worker_replicas,
                total_num_replicas=worker_replicas,
                variable_averages=ema,
                variables_to_average=tf.trainable_variables())

            train_op = slim.learning.create_train_op(
                total_loss=loss,
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
                number_of_steps=wn.num_iters,
                global_step=global_step,
                log_every_n_steps=250,
                local_init_op=local_init_op,
                save_interval_secs=3600,
                sync_optimizer=opt,
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
