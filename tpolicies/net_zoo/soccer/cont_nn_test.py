from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym import spaces
import numpy as np

from tpolicies.net_zoo.soccer.cont_nn import net_config_cls
from tpolicies.net_zoo.soccer.cont_nn import net_inputs_placeholders_fun
from tpolicies.net_zoo.soccer.cont_nn import net_build_fun


def cont_nn_inputs_test():
  mycfg = {
    'test': False,
    'batch_size': 32,
    'rollout_len': 16
  }

  # a team has two agents (obs 119*2, act 3*2)
  ob_space = GymBox(-2., 2., (238,), dtype=np.float32)
  ac_space = GymBox(-1., 1., (6,), dtype=np.float32)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  print(inputs.X)
  print(inputs.A)
  pass


def cont_nn_test():
  mycfg = {
    'test': False,
    'use_loss_type': 'rl',
    'use_value_head': True,
    'n_v': 1,
    'sync_statistics': None,
    'batch_size': 32,
    'weight_decay': 0.0005
  }

  ob_space = GymBox(-2., 2., (34,), dtype=np.float32)
  ac_space = GymBox(-1., 1., (6,), dtype=np.float32)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='cont_nn')

  print(out.value_head)
  assert out.value_head is not None

  print(out.loss.total_reg_loss)
  print(out.loss.loss_endpoints)
  print(out.loss.pg_loss)
  assert out.loss.pg_loss is not None
  print(out.loss.value_loss)
  assert out.loss.value_loss is not None

  print(out.vars.all_vars)
  print(len(out.vars.all_vars))

  for v in out.vars.all_vars:
    print(v.name)

  print(out.endpoints)
  pass


if __name__ == '__main__':
  cont_nn_inputs_test()
  cont_nn_test()
  pass