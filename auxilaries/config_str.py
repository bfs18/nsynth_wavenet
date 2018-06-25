from wavenet import parallel_wavenet, wavenet, masked
import json
import subprocess
from argparse import Namespace
import time


def get_config_srt(hparams, model, tag=''):
    prefix = 'ns_'  # nsynth
    if model == 'wavenet':
        model_str = 'wn'
    elif model == 'parallel_wavenet':
        model_str = 'pwn'
    else:
        raise ValueError('unsupported model type {}'.format(model))

    branch_names = subprocess.check_output(['git', 'branch'])
    current_branch_name = [bn for bn in branch_names.decode('utf-8').split('\n')
                           if '*' in bn][0]
    current_branch_name = current_branch_name.split()[1]

    if getattr(hparams, 'use_mu_law', False):
        mu_law_tag = 'MU'
    else:
        mu_law_tag = 'n_MU'

    if getattr(hparams, 'use_weight_norm', False):
        weight_norm_tag = 'WN'
        if current_branch_name == 'data_dep_init':
            weight_norm_tag += '_DDI'
            if parallel_wavenet.MANUAL_FINAL_INIT and model == 'parallel_wavenet':
                weight_norm_tag += '_mfinit'
    else:
        weight_norm_tag = 'n_WN'

    if getattr(hparams, 'use_log_scale', False):
        log_scale_tag = 'LOGS'
    else:
        log_scale_tag = 'n_LOGS'

    loss_type = getattr(hparams, 'loss_type', '').upper()

    cstr = '-'.join([prefix + model_str, mu_law_tag, weight_norm_tag])

    if model == 'parallel_wavenet':
        cstr += '-{}'.format(log_scale_tag)
        if parallel_wavenet.CLIP:
            cstr += '-CLIP'
        else:
            cstr += '-n_CLIP'

        if parallel_wavenet.LOG_SPEC:
            cstr += '-LABS'
        else:
            cstr += '-NMAG'

    if model == 'wavenet' and getattr(hparams, 'add_noise', False):
        cstr += '-NOISE'

    if loss_type:
        cstr += '-{}'.format(loss_type)

    if tag:
        cstr += '-{}'.format(tag)

    return cstr


def get_time_str():
    return time.strftime("%m_%d", time.localtime())


def get_config_time_str(hparams, model, tag=''):
    cstr = get_config_srt(hparams, model, tag) + '-' + get_time_str()
    return cstr


if __name__ == '__main__':
    config1 = '../config_jsons/wavenet_mol.json'
    with open(config1, 'rt') as F:
        configs = json.load(F)
    hparams = Namespace(**configs)
    print(get_config_srt(hparams, 'wavenet'))

    config1 = '../config_jsons/parallel_wavenet.json'
    with open(config1, 'rt') as F:
        configs = json.load(F)
    hparams = Namespace(**configs)
    print(get_config_srt(hparams, 'parallel_wavenet'))
    print(get_config_time_str(hparams, 'parallel_wavenet'))
