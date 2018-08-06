import json
import os
import glob
import tensorflow as tf

from argparse import ArgumentParser, Namespace
from wavenet import fastgen
from auxilaries import utils


def generate(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    source_path = utils.shell_path(args.source_path)
    ckpt_dir = utils.shell_path(args.ckpt_dir)
    save_path = utils.shell_path(args.save_path)
    if not os.path.exists(save_path):
        tf.logging.info("save_path does not exist, make it.")
        os.mkdir(save_path)
    tf.logging.set_verbosity(args.log)

    assert tf.gfile.IsDirectory(ckpt_dir)
    checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)
    assert tf.train.checkpoint_exists(checkpoint_path)
    json_in_dir = glob.glob(os.path.join(ckpt_dir, '*.json'))
    assert len(json_in_dir) == 1
    config_json = json_in_dir[0]

    with open(config_json, 'rt') as F:
        configs = json.load(F)
    hparams = Namespace(**configs)

    # Generate from wav files
    if tf.gfile.IsDirectory(source_path):
        files = tf.gfile.ListDirectory(source_path)
        exts = [os.path.splitext(f)[1] for f in files]
        if ".wav" in exts:
            postfix = ".wav"
        elif ".npy" in exts:
            postfix = ".npy"
        else:
            raise RuntimeError("Folder must contain .wav or .npy files.")
        postfix = ".npy" if args.npy_only else postfix
        files = sorted(
            [os.path.join(source_path, fname)
             for fname in files
             if fname.lower().endswith(postfix)])
    elif source_path.lower().endswith((".wav", ".npy")):
        files = [source_path]
    else:
        files = []

    batch_size = args.batch_size
    sample_length = args.sample_length
    n = len(files)
    for start in range(0, n, batch_size):
        tf.logging.info('generating batch {:d}'.format(start // batch_size))

        end = start + batch_size
        batch_files = files[start: end]
        save_names = [
            os.path.join(save_path,
                         'gen_' + os.path.splitext(os.path.basename(f))[0] + '.wav')
            for f in batch_files]
        # use the original wave length.
        batch_data = fastgen.load_batch(batch_files, sample_length=sample_length)

        # extract and encode mel
        encoding = fastgen.encode(hparams, batch_data, checkpoint_path)
        fastgen.synthesis(hparams, encoding, save_names, checkpoint_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--ckpt_dir", required=True,
                        help="Path or directory to checkpoint.")
    parser.add_argument("--source_path", required=True,
                        help="Path to directory with either "
                             ".wav files or precomputed encodings in .npy files."
                             "If .wav files are present, use wav files. If no "
                             ".wav files are present, use .npy files")
    parser.add_argument("--save_path", required=True,
                        help="Path to output file dir.")
    parser.add_argument("--sample_length", default=-1, type=int,
                        help="Max output file size in samples.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Number of samples per a batch.")
    parser.add_argument("--npy_only", default=False, type=bool,
                        help="If True, use only .npy files.")
    parser.add_argument("--log", default="INFO",
                        help="The threshold for what messages will be logged."
                             "DEBUG, INFO, WARN, ERROR, or FATAL.")
    parser.add_argument("--gpu_id", default='0',
                        help="gpu device for generation.")
    args = parser.parse_args()
    generate(args)
