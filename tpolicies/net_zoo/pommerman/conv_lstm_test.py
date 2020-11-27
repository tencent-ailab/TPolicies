from gym import spaces
import numpy as np
import tensorflow as tf

from tpolicies.net_zoo.pommerman.conv_lstm import net_config_cls
from tpolicies.net_zoo.pommerman.conv_lstm import net_inputs_placeholders_fun
from tpolicies.net_zoo.pommerman.conv_lstm import net_build_fun


def conv_lstm_inputs_test():
  mycfg = {
    'test': False,
    'use_lstm': True,
    'batch_size': 32,
    'rollout_len': 2,
    'nlstm': 64,
    'hs_len': 64 * 2,
    'lstm_layer_norm': True
  }

  ob_space = spaces.Tuple(
    [spaces.Tuple([spaces.Box(shape=(11, 11, 22), dtype=np.float32, low=0, high=1),
                   spaces.Box(shape=[6], dtype=np.bool, low=0, high=1)])] * 2)
  ac_space = spaces.Tuple([spaces.Discrete(n=6)] * 2)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  print(inputs.X)
  print(inputs.A)
  print(inputs.S)
  print(inputs.M)
  pass


def conv_lstm_test():
  mycfg = {
    'test': False,
    'use_loss_type': 'rl',
    'use_value_head': True,
    'n_v': 1,
    'sync_statistics': None,
    'use_lstm': True,
    'batch_size': 32,
    'rollout_len': 8,
    'nlstm': 64,
    'hs_len': 64 * 2,
    'lstm_layer_norm': True,
    'weight_decay': 0.0005
  }

  ob_space = spaces.Tuple(
    [spaces.Tuple([spaces.Box(shape=(11, 11, 22), dtype=np.float32, low=0, high=1),
                   spaces.Box(shape=(2,), dtype=np.int32, low=0, high=10),
                   spaces.Box(shape=[6], dtype=np.bool, low=0, high=1)])] * 2)
  ac_space = spaces.Tuple([spaces.Discrete(n=6)] * 2)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='conv_lstm')

  print(out.value_head)
  assert out.value_head is not None

  print(out.loss.total_reg_loss)
  print(out.loss.loss_endpoints)
  print(out.loss.pg_loss)
  assert out.loss.pg_loss is not None
  print(out.loss.value_loss)
  assert out.loss.value_loss is not None

  print(out.vars.lstm_vars)
  print(len(out.vars.lstm_vars))
  print(out.vars.all_vars)
  print(len(out.vars.all_vars))

  for v in out.vars.all_vars:
    print(v.name)

  print(out.endpoints)
  pass


def conv_lstm_actor_test():
  mycfg = {
    'test': False,
    'use_loss_type': 'none',
    'use_value_head': False,
    'n_v': 4,
    'sync_statistics': None,
    'use_lstm': True,
    'batch_size': 1,
    'rollout_len': 1,
    'nlstm': 64,
    'hs_len': 64 * 2,
    'lstm_layer_norm': True,
    'weight_decay': 0.0005
  }

  ob_space = spaces.Tuple(
    [spaces.Tuple([spaces.Box(shape=(11, 11, 22), dtype=np.float32, low=0, high=1),
                   spaces.Box(shape=(2,), dtype=np.int32, low=0, high=10),
                   spaces.Box(shape=[6], dtype=np.bool, low=0, high=1)])] * 2)
  ac_space = spaces.Tuple([spaces.Discrete(n=6)] * 2)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='conv_lstm')
  sample = ob_space.sample()
  sess = tf.Session()
  tf.global_variables_initializer().run(session=sess)
  feed_dict = {}
  for s, input in zip(sample, inputs.X):
    for x_np, x in zip(s, input):
      feed_dict[x] = [x_np]
  feed_dict[inputs.S] = np.zeros(shape=[1, nc.hs_len])
  feed_dict[inputs.M] = np.zeros(shape=[1])
  from tensorflow.contrib.framework import nest
  import tpolicies.tp_utils as tp_utils
  ac_structure = tp_utils.template_structure_from_gym_space(ac_space)
  a = nest.map_structure_up_to(ac_structure, lambda head: head.sam,
                               out.self_fed_heads)
  sam = sess.run(a, feed_dict=feed_dict)
  print(sam)
  pass


def conv_lstm_player_test():
  mycfg = {
    'test': False,
    'use_loss_type': 'none',
    'use_value_head': False,
    'n_v': 4,
    'sync_statistics': None,
    'use_lstm': True,
    'batch_size': 1,
    'rollout_len': 1,
    'nlstm': 32,
    'hs_len': 32 * 2,
    'lstm_layer_norm': True,
    'weight_decay': 0.0005,
    'n_player': 1,
  }

  ob_space = spaces.Tuple([spaces.Box(shape=(11, 11, 22), dtype=np.float32, low=0, high=1),
                           spaces.Box(shape=(2,), dtype=np.int32, low=0, high=10),
                           spaces.Box(shape=[6], dtype=np.bool, low=0, high=1)])
  ac_space = spaces.Discrete(n=6)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='conv_lstm')
  sample = ob_space.sample()
  sess = tf.Session()
  tf.global_variables_initializer().run(session=sess)
  feed_dict = {}
  for s, input in zip(sample, inputs.X):
    feed_dict[input] = [s]
  feed_dict[inputs.S] = np.zeros(shape=[1, nc.hs_len])
  feed_dict[inputs.M] = np.zeros(shape=[1])
  from tensorflow.contrib.framework import nest
  import tpolicies.tp_utils as tp_utils
  ac_structure = tp_utils.template_structure_from_gym_space(ac_space)
  a = nest.map_structure_up_to(ac_structure, lambda head: head.sam,
                               out.self_fed_heads)
  sam = sess.run(a, feed_dict=feed_dict)
  print(sam)
  pass

if __name__ == '__main__':
  conv_lstm_inputs_test()
  conv_lstm_test()
  conv_lstm_actor_test()
  conv_lstm_player_test()
  pass