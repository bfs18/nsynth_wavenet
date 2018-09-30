import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tensorflow


te_ckpt_dir = '/data/logs-wavenet/nsynth_wavenet-mol'
st_ckpt_dir = '/data/logs-wavenet/ns_pwn'

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


te_trans_conv_var_names = [vn for vn in inspect_var_names if 'trans_conv' in vn]
st_trans_conv_var_names_flow_nested = []
for te_tcvn in te_trans_conv_var_names:
    st_tcvn_for_flows = []
    for vn in st_var_names:
        if vn.endswith(te_tcvn):
            st_tcvn_for_flows.append(vn)
    st_trans_conv_var_names_flow_nested.append(st_tcvn_for_flows)

for te_tcvn, st_tcvn_for_flows in zip(
        te_trans_conv_var_names, st_trans_conv_var_names_flow_nested):
    for st_tcvn in st_tcvn_for_flows:
        assert np.allclose(
            st_reader.get_tensor(st_tcvn),
            te_reader.get_tensor('{}/ExponentialMovingAverage'.format(te_tcvn)))
        print(st_tcvn, te_tcvn, 'OK.')
