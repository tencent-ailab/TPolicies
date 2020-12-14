from gym import spaces
import numpy as np
import tensorflow as tf

from tpolicies.net_zoo.gym_ddpg.ddpg import net_config_cls
from tpolicies.net_zoo.gym_ddpg.ddpg import net_inputs_placeholders_fun
from tpolicies.net_zoo.gym_ddpg.ddpg import net_build_fun


def gym_ddpg_test():
  mycfg = {
    'test': False,
    'use_loss_type': 'rl',
    'use_value_head': True,
    'use_target_net': True,
    'n_v': 1,
    'use_lstm': True,
    'batch_size': 32,
    'rollout_len': 8,
    'nlstm': 64,
    'hs_len': 64 * 2,
    'lstm_layer_norm': True,
    'weight_decay': 0.0005,
    'lam': 0.99,
  }

  ob_space = spaces.Box(shape=(11,), dtype=np.float32, low=0, high=1)
  ac_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  nc.reward_weights = np.ones(shape=nc.reward_weights_shape, dtype=np.float32)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='gym_ddpg')

  print(out.self_fed_value_head)
  print(out.outer_fed_value_head)

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


def gym_ddpg_actor_test():
  mycfg = {
    'test': False,
    'use_loss_type': 'none',
    'use_value_head': False,
    'n_v': 4,
    'use_lstm': True,
    'batch_size': 1,
    'rollout_len': 1,
    'nlstm': 64,
    'hs_len': 64 * 2,
    'lstm_layer_norm': True,
    'weight_decay': 0.0005
  }

  ob_space = spaces.Box(shape=(11,), dtype=np.float32, low=0, high=1)
  ac_space = spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32)

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='gym_ddpg')
  sample = ob_space.sample()
  sess = tf.Session()
  tf.global_variables_initializer().run(session=sess)
  feed_dict = {inputs.X: [sample]}
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
  gym_ddpg_test()
  gym_ddpg_actor_test()
  pass