from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
import numpy as np

from tpolicies.net_zoo.conv_lstm.conv_lstm import net_config_cls
from tpolicies.net_zoo.conv_lstm.conv_lstm import net_inputs_placeholders_fun
from tpolicies.net_zoo.conv_lstm.conv_lstm import net_build_fun


def conv_lstm_inputs_test():
  mycfg = {
    'test': False,
    'use_lstm': True,
    'batch_size': 32,
    'rollout_len': 16,
    'nlstm': 128,
    'hs_len': 128 * 2,
    'lstm_layer_norm': True
  }

  ob_space = GymBox(low=0.0, high=255.0, dtype=np.float32, shape=(84, 84, 3))
  ac_space = GymDiscrete(n=6)

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
    'rollout_len': 16,
    'nlstm': 128,
    'hs_len': 128 * 2,
    'lstm_layer_norm': True,
    'weight_decay': 0.0005
  }

  ob_space = GymBox(low=0.0, high=255.0, dtype=np.float32, shape=(84, 84, 3))
  ac_space = GymDiscrete(n=6)

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


if __name__ == '__main__':
  conv_lstm_inputs_test()
  conv_lstm_test()
  pass