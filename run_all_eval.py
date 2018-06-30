import paramiko
import os
import re
import shutil
import json
import time
import argparse

from scp import SCPClient
from contextlib import closing


def create_ssh_client(server, port, user, password):
    print("Create ssh client {}".format(server))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def create_scp_client(ssh_client):
    client = SCPClient(ssh_client.get_transport(), sanitize=lambda x: x)
    return client


def list_log_dir(ssh_client, logdir):
    if ssh_client is None:
        log_files = os.listdir(logdir)
    else:
        _, stdout, _ = ssh_client.exec_command("ls {}".format(logdir))
        log_files = [l.strip() for l in stdout.readlines()]
    return log_files


def get_last_model_prefix(index_files):
    def _get_iter(index_file):
        matched = re.match('model.ckpt-(\d+).index', index_file)
        return int(matched.group(1))
    last_iter = max([_get_iter(idx_f) for idx_f in index_files])
    return 'model.ckpt-{}'.format(last_iter), last_iter


def write_checkpoint(model_path, save_path):
    with open(save_path, 'wt') as F:
        print('model_checkpoint_path:', file=F, end=' ')
        print('\"{}\"'.format(model_path), file=F, end='\n')
        print('all_model_checkpoint_paths:', file=F, end=' ')
        print('\"{}\"'.format(model_path), file=F, end='\n')


def copy_useful_data(ssh_client, source_logdir, target_dir):
    if not ssh_client:
        source_logdir = os.path.expanduser(source_logdir)

    log_files = list_log_dir(ssh_client, source_logdir)
    events_files = [lf for lf in log_files if lf.startswith('events.')]
    json_file = [lf for lf in log_files if lf.endswith('.json')][0]

    index_files = [lf for lf in log_files
                   if lf.startswith('model.') and lf.endswith('.index')]
    last_model_prefix, last_iter = get_last_model_prefix(index_files)

    exp_tag = os.path.basename(source_logdir)
    events_dir = os.path.join(target_dir, exp_tag)
    model_dir = os.path.join(target_dir, exp_tag + '-model')
    model_path = os.path.join(model_dir, last_model_prefix)
    wave_dir = os.path.join(
        target_dir, 'waves', exp_tag + '-iter_{}'.format(last_iter))
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(wave_dir, exist_ok=True)

    checkpoint_txt_path = os.path.join(model_dir, 'checkpoint')
    write_checkpoint(last_model_prefix, checkpoint_txt_path)

    src_model_patt = os.path.join(source_logdir, last_model_prefix + '.*')
    src_json_file = os.path.join(source_logdir, json_file)
    src_events_patt = os.path.join(source_logdir, 'events.*')

    print('Copying data')
    if ssh_client is None:
        os.system(' '.join(['cp', src_model_patt, model_dir]))
        os.system(' '.join(['cp', src_json_file, model_dir]))
        os.system(' '.join(['cp', src_events_patt, events_dir]))
    else:
        with closing(create_scp_client(ssh_client)) as scp:
            scp.get(src_model_patt, model_dir)
            scp.get(src_json_file, model_dir)
            scp.get(src_events_patt, events_dir)

    return events_dir, model_dir, wave_dir


def syn_wave(eval_script, ckpt_dir, source_path, save_path, gpu_id):
    print('Running evaluation')
    cmd = ['python3', eval_script, '--ckpt_dir', ckpt_dir,
           '--source_path', source_path, '--save_path', save_path,
           '--gpu_id', gpu_id]
    os.system(' '.join(cmd))


def copy_run(server_ip, user, password, source_logdir, target_dir,
             eval_script, source_waves, gpu_id):
    print('Running eval for {} from {}'.format(
        os.path.basename(source_logdir),
        server_ip if server_ip else 'localhost'))

    if server_ip:
        ssh_client = create_ssh_client(server_ip, '36000', user, password)
    else:
        ssh_client = None

    events_dir, model_dir, wave_dir = copy_useful_data(
        ssh_client, source_logdir, target_dir)

    if ssh_client:
        ssh_client.close()

    syn_wave(eval_script, model_dir, source_waves, wave_dir, gpu_id)
    shutil.rmtree(model_dir)


def run_all(json_path, source_waves, target_dir, gpu_id):
    with open(json_path, 'rt') as F:
        configs = json.load(F)

    target_dir = os.path.abspath(os.path.expanduser(target_dir))
    source_waves = os.path.abspath(os.path.expanduser(source_waves))
    time_str = time.strftime("%m_%d_%H", time.localtime())
    target_dir = '-'.join([target_dir, time_str])

    print('Save all data to {}'.format(target_dir))
    print('Use source waves in {}'.format(source_waves))

    for host, user, passwd, src_logdir, eval_script in zip(
        configs['hosts'], configs['users'], configs['passwords'],
            configs['exp_dirs'], configs['eval_scripts']):
        copy_run(host, user, passwd, src_logdir, target_dir,
                 eval_script, source_waves, gpu_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, help='Config json file')
    parser.add_argument('--wave_dir', '-w', required=True, help='Source wave directory')
    parser.add_argument('--target_dir', '-t', required=True, help='Target directory')
    parser.add_argument('--gpu_id', '-g', default="\"\"", help='Gpu id')

    args = parser.parse_args()
    run_all(args.config, args.wave_dir, args.target_dir, args.gpu_id)
