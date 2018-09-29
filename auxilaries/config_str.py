from wavenet import parallel_wavenet, wavenet, masked
import json
import subprocess
from argparse import Namespace
import time
from auxilaries import reader


def get_config_srt(hparams, model, tag=''):
    prefix = 'ns_'  # nsynth
    if model == 'wavenet':
        model_str = 'wn'
    elif model == 'parallel_wavenet':
        model_str = 'pwn'
    else:
        raise ValueError('unsupported model type {}'.format(model))
    model_str = '{}_{}'.format(model_str, tag) if tag else model_str

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

    loss_type = getattr(hparams, 'loss_type', '').upper()

    cstr = '-'.join([prefix + model_str, mu_law_tag, weight_norm_tag])

    if reader.USE_NEW_MEL_EXTRACTOR:
        cstr += '-NM'

    if getattr(hparams, 'use_resize_conv', False):
        cstr += '-RS'
    else:
        cstr += '-TS'

    upsample_act = getattr(hparams, 'upsample_act', 'tanh')
    cstr += ('-' + upsample_act)

    if model == 'parallel_wavenet':
        if parallel_wavenet.USE_LOG_SCALE:
            cstr += '-LOGS'
        else:
            cstr += '-n_LOGS'

        if parallel_wavenet.CLIP:
            cstr += '-CLIP'
        else:
            cstr += '-n_CLIP'

        if parallel_wavenet.SPEC_ENHANCE_FACTOR == 0:
            cstr += '-NLABS' if parallel_wavenet.NORM_FEAT else '-LABS'
        elif parallel_wavenet.SPEC_ENHANCE_FACTOR == 1:
            cstr += '-NABS' if parallel_wavenet.NORM_FEAT else '-ABS'
        elif parallel_wavenet.SPEC_ENHANCE_FACTOR == 2:
            cstr += '-NPOW' if parallel_wavenet.NORM_FEAT else '-POW'
        elif parallel_wavenet.SPEC_ENHANCE_FACTOR == 3:
            cstr += '-NCOM' if parallel_wavenet.NORM_FEAT else '-COM'
        else:
            raise ValueError("SPEC_ENHANCE_FACTOR Value Error.")

        if parallel_wavenet.USE_MEL:
            cstr += '-MEL'
        else:
            cstr += '-n_MEL'

        if parallel_wavenet.USE_L1_LOSS:
            cstr += '-L1'
        else:
            cstr += '-L2'

        if parallel_wavenet.USE_PRIORITY_FREQ:
            cstr += '-PFS'
        else:
            cstr += '-n_PFS'

        if getattr(hparams, 'use_share_deconv', False):
            cstr += '-share_deconv'
    else:
        if getattr(hparams, 'use_input_noise', False):
            cstr += '-IN'
        else:
            cstr += '-n_IN'

        if getattr(hparams, 'dropout_inputs', False):
            cstr += '-DO'
        else:
            cstr += '-n_DO'

    if loss_type:
        cstr += '-{}'.format(loss_type)

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
