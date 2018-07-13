import os
import argparse
import shutil
import tensorflow as tf


def save_eval_model(ckpt_dir, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    save_ckpt_path = os.path.join(save_dir, os.path.basename(ckpt.model_checkpoint_path))
    with tf.Session() as sess:
        eval_var_names = []
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_dir):
            if 'ExponentialMovingAverage' in var_name:
                eval_var_names.append(var_name)
        eval_vars = []
        for vn in eval_var_names:
            np_var = tf.contrib.framework.load_variable(ckpt_dir, vn)
            eval_vars.append(tf.Variable(np_var, name=vn))

        saver = tf.train.Saver(var_list=eval_vars)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, save_ckpt_path, write_meta_graph=False, write_state=False)

    os.system('cp {}/*.json {}'.format(ckpt_dir, save_dir))
    os.system('cp {}/checkpoint {}'.format(ckpt_dir, save_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir')
    parser.add_argument('--save_dir')
    args = parser.parse_args()
    save_eval_model(args.ckpt_dir, args.save_dir)
