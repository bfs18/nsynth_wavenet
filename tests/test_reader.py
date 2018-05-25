import os
import sys
import librosa
import tensorflow as tf
sys.path.append(os.path.dirname(os.getcwd()))
from auxilaries import reader

os.environ['CUDA_VISIBLE_DEVICES'] = ''

tfr_path = '/data/corpus/LJSpeech-1.1/ljspeech.tfrecord'
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
