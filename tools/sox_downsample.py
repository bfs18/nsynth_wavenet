import os
import glob
import shutil
import argparse
from tqdm import tqdm

SOX = 'sox'


def downsample(orig_wave, target_wave, target_sr):
    cmd = [SOX, orig_wave, '-r {}'.format(target_sr), target_wave]
    os.system(' '.join(cmd))


def downsample_dir(wave_dir, target_dir, target_sr=16000):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    wave_file = glob.glob(os.path.join(wave_dir, '*.wav'))
    for wf in tqdm(wave_file):
        wave_name = os.path.basename(wf)
        save_wave = os.path.join(target_dir, wave_name)
        downsample(wf, save_wave, target_sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_dir', '-wd', required=True)
    parser.add_argument('--target_dir', '-td', required=True)
    parser.add_argument('--target_sr', '-sr', default=16000, type=int)
    args = parser.parse_args()

    wave_dir = args.wave_dir
    target_dir = args.target_dir
    target_sr = args.target_sr
    downsample_dir(wave_dir, target_dir, target_sr)
