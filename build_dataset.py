import os
import glob
import librosa
import numpy as np
import tensorflow as tf
import argparse

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _pad_wave(waveform, min_len):
    if waveform.shape[0] < min_len:
        return np.pad(waveform, [0, min_len - waveform.shape[0]], 'constant')
    else:
        return waveform


def _make_example(wf, sr, min_len):
    audio_id = os.path.splitext(os.path.basename(wf))[0]
    audio_id_bytes = bytes(audio_id, encoding='utf-8')
    waveform, _ = librosa.load(wf, sr=sr)
    padded_waveform = _pad_wave(waveform, min_len)
    waveform_bytes = padded_waveform.astype(np.float32).tobytes()
    feature = {
        'audio_id': _bytes_feature(bytes(audio_id_bytes)),
        'audio': _bytes_feature(waveform_bytes),
        'length': _int64_feature(waveform.shape[0])}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    sec = waveform.shape[0] / float(sr)
    padded = 1 if padded_waveform is not waveform else 0
    return example.SerializeToString(), sec, padded


def build_dataset(wave_dir, save_path, sr=16000, min_len=64000, num_workers=10):
    wave_files = glob.glob(os.path.join(wave_dir, '*.wav'))
    wave_files = list(sorted(wave_files))

    executor = ThreadPoolExecutor(max_workers=num_workers)
    results = []
    for wf in wave_files:
        results.append(executor.submit(_make_example, wf, sr, min_len))
    res_vals = [res.result() for res in tqdm(results)]
    total_sec = sum([rv[1] for rv in res_vals])

    writer = tf.python_io.TFRecordWriter(save_path)
    padded_count = sum([rv[2] for rv in res_vals])
    [writer.write(rv[0]) for rv in res_vals if rv[0] is not None]
    writer.close()

    print('total duration: {:.5f} hours'.format(total_sec / 3600.))
    print('padded samples: {}/{} pieces'.format(padded_count, len(wave_files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_dir', required=True, help='input wave directory')
    parser.add_argument('--save_path', required=True, help='tfrecord save path')
    parser.add_argument('--sample_rate', default=16000, help='wave sample rate')
    parser.add_argument('--min_len', default=16000, help='minimum length for padding')
    parser.add_argument('--num_workers', default=10, help='number of workers')

    args = parser.parse_args()
    build_dataset(args.wave_dir, args.save_path, args.sample_rate, args.min_len)
