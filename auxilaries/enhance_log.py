import logging
import numbers
import collections
import os


def add_log_file(logdir):
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(logdir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def format_value(value):
    if isinstance(value, collections.Mapping):
        return format_value(value.items())
    elif isinstance(value, collections.Iterable) and not isinstance(value, str):
        value_str = []
        for v in value:
            value_str.append(format_value(v))
        return '(' + ', '.join(value_str) + ')'
    elif isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return '{:.2e}'.format(value)
    else:
        return '{}'.format(value)


def instance_attr_to_str(instance):
    attrs = []
    hparams = vars(instance.hparams)

    for k, v in hparams.items():
        attrs.append((k, v))

    for k, v in vars(instance).items():
        if k == 'hparams' or k in hparams:
            continue
        attrs.append((k, v))

    sorted_attrs = sorted(attrs, key=lambda x: x[0])

    attr_str = []
    for attr in sorted_attrs:
        attr_str.append('{}: {}'.format(attr[0], format_value(attr[1])))

    return '\n'.join(attr_str)
