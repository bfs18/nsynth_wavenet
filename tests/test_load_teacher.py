import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tensorflow


te_ckpt_dir = '/data/logs-wavenet/nsynth_wavenet_mol'
st_ckpt_dir = '/data/logs-wavenet/nsynth_parallel_wavenet'

te_ckpt = tf.train.latest_checkpoint(te_ckpt_dir)
assert tf.train.checkpoint_exists(te_ckpt)
st_ckpt = tf.train.latest_checkpoint(st_ckpt_dir)
assert tf.train.checkpoint_exists(st_ckpt)

te_reader = pywrap_tensorflow.NewCheckpointReader(te_ckpt)
te_var_to_shape_map = te_reader.get_variable_to_shape_map()
te_var_names = list(te_var_to_shape_map.keys())

st_reader = pywrap_tensorflow.NewCheckpointReader(st_ckpt)
st_var_to_shape_map = st_reader.get_variable_to_shape_map()
st_var_names = list(st_var_to_shape_map.keys())

inspect_var_names = [vn for vn in te_var_names
                     if (vn.endswith('W') or vn.endswith('biases') or
                         vn.endswith('bias') or vn.endswith('kernel'))]

for ivn in inspect_var_names:
    assert np.allclose(
        st_reader.get_tensor(ivn),
        te_reader.get_tensor('{}/ExponentialMovingAverage'.format(ivn)))
