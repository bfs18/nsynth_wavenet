import os
import sys
import librosa
import shutil
import tensorflow as tf
sys.path.append(os.path.dirname(os.getcwd()))
from auxilaries import reader

os.environ['CUDA_VISIBLE_DEVICES'] = ''

tfr_path = '/data/corpus/LJSpeech-1.1/ljspeech.tfrecord'


def test_tf_reader():
    dataset = reader.Dataset(tfr_path, is_training=False)
    inputs = dataset.get_batch(8)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    for i in range(10):
        in_vals = sess.run(inputs)
        print(in_vals['key'])
        wp = os.path.join('test_data', in_vals['key'][0].decode('utf-8') + '.wav')
        librosa.output.write_wav(wp, in_vals['wav'][0], sr=16000)


def test_np_reader():
    input_dict = reader.get_init_batch(tfr_path, 4, first_n=10)
    waves = input_dict['wav']
    out_dir = 'test_reader'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for i, wave in enumerate(waves):
        save_name = '{}/test_reader-{}.wav'.format(out_dir, i)
        librosa.output.write_wav(save_name, wave, sr=16000)


if __name__ == "__main__":
    test_tf_reader()
    test_np_reader()
