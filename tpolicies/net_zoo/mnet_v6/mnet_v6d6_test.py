from os import path

import tensorflow as tf
from timitate.lib6.pb2all_converter import PB2AllConverter
import numpy as np
from absl import flags
from absl import app
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
from timitate.utils_replay import get_replay_actor_interface
from timitate.lib6.z_actions import ZERG_ABILITIES
from tleague.envs.sc2 import sc2_env_space

from tpolicies.net_zoo.mnet_v6.mnet_v6d6 import net_config_cls
from tpolicies.net_zoo.mnet_v6.mnet_v6d6 import net_inputs_placeholders_fun
from tpolicies.net_zoo.mnet_v6.mnet_v6d6 import net_build_fun


FLAGS = flags.FLAGS
flags.DEFINE_string("zstat_data_src", '/Users/jcxiong/SC2/rp1522-mmr-ge6000-winner-selected-8', "")
flags.DEFINE_string("game_version", '4.10.0', "")
flags.DEFINE_string("replay_name", 'temp', "")
flags.DEFINE_string("replay_dir", '/Users/jcxiong/SC2/rp1522-mmr-ge6000-winner-selected-8/', "")
flags.DEFINE_string("map_name", 'KairosJunction', "")
flags.DEFINE_integer("player_id", 2, "")

converter = PB2AllConverter(dict_space=True, zmaker_version='v5')
ob_space, ac_space = converter.space.spaces


def mnet_v6d6_inputs_test():
  mycfg = {
    'use_self_fed_heads': False,
    'use_lstm': True,
    'batch_size': 32,
    'rollout_len': 1,
    'lstm_duration': 4,
    'nlstm': 256,
    'hs_len': 256 * 2 + 1,
    'lstm_cell_type': 'k_lstm'
  }

  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  print(inputs.X)
  print(inputs.A)
  print(inputs.S)
  print(inputs.M)
  pass


def mnet_v6d6_test():
  mycfg = {
    'test': True,
    'use_self_fed_heads': True,
    'use_value_head': False,
    'use_loss_type': 'none',
    'use_lstm': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'batch_size': 32,
    'rollout_len': 16,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'weight_decay': 0.0005,
    'arg_scope_type': 'mnet_v5_type_a',
    'use_base_mask': True,
    'reward_weights': [[0.0, 0.0, 0.0, 0.0]],
    'n_v': 4,
    'vec_embed_version': 'v3',
    'last_act_embed_version': 'v2',
    'zstat_embed_version': 'v7',
    'trans_version': 'v4',
  }
  nc = net_config_cls(ob_space, ac_space, **mycfg)

  inputs = net_inputs_placeholders_fun(nc)

  out = net_build_fun(inputs, nc)
  print(out.self_fed_heads)
  print(out.outer_fed_heads)
  print(out.S)
  print(out.embed)

  print(out.value_head)
  assert out.value_head is None

  print(out.loss.total_reg_loss)

  print(out.loss.loss_endpoints)
  assert out.loss.loss_endpoints == {}

  print(out.loss.total_il_loss)
  assert out.loss.total_il_loss is None

  print(out.loss.pg_loss)
  assert out.loss.pg_loss is None

  print(out.vars.lstm_vars)
  print(out.vars.all_vars)
  for v in out.vars.all_vars:
    print(v.name)
  print(len(out.vars.all_vars))

  print(out.endpoints)
  pass


def mnet_v6d6_endpoints_test():
  mycfg = {
    'use_self_fed_heads': False,
    'use_lstm': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'batch_size': 32,
    'rollout_len': 16,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'rl': False,
    'vec_embed_version': 'v3',
    'last_act_embed_version': 'v2',
    'zstat_embed_version': 'v7',
    'trans_version': 'v1',
    'embed_for_action_heads': 'int',
    'weight_decay': None,
    'arg_scope_type': 'mnet_v5_type_a',
    'endpoints_norm': False,
    'endpoints_verbosity': 20,
  }
  ob_space, ac_space = sc2_env_space('sc2full_formal8_dict')
  nc = net_config_cls(ob_space, ac_space, **mycfg)

  with tf.variable_scope('mnet_v6d6_ep', reuse=tf.AUTO_REUSE) as sc:
    pass
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope=sc)

  all_eps = tf.get_collection('mnet_v6d6_ep_1_endpoints')
  for v in all_eps:
    print(v)
  print('number of all eps: {}'.format(len(all_eps)))

  #print(out.endpoints)
  for k, v in out.endpoints.items():
    print('{}: {}'.format(k, v))
  print('number of endpoints: {}'.format(len(out.endpoints)))


def mnet_v6d6_run_il_loss_test():
  mycfg = {
    'test': False,
    'use_self_fed_heads': False,
    'use_value_head': False,
    'use_loss_type': 'il',
    'il_multi_label_loss': True,
    'use_lstm': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'batch_size': 2,
    'rollout_len': 1,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'weight_decay': 0.0005,
    'arg_scope_type': 'mnet_v6_type_a',
    'use_base_mask': True,
    'vec_embed_version': 'v2d1',
    'last_act_embed_version': 'v2',
    'zstat_embed_version': 'v7',
    'trans_version': 'v5',
    'gather_batch': True,
    'use_astar_glu': True,
    'use_astar_func_embed': True,
    'pos_logits_mode': '3x3up2',
    'pos_n_blk': 3,
    'pos_n_skip': 2,
  }
  converter = PB2AllConverter(dict_space=True,
                              zmaker_version='v5',
                              zstat_data_src=FLAGS.zstat_data_src,
                              game_version=FLAGS.game_version,
                              sort_executors='v1',
                              delete_dup_action='v2',
                              input_map_size=(128, 128),
                              output_map_size=(256, 256))
  ob_space, ac_space = converter.space.spaces

  # build the net
  nc = net_config_cls(ob_space, ac_space, **mycfg)
  inputs = net_inputs_placeholders_fun(nc)
  out = net_build_fun(inputs, nc, scope='mnet_v6d6_il_loss')
  print('Successfully created the net.')

  converter.reset(
    replay_name=FLAGS.replay_name,
    player_id=FLAGS.player_id,
    mmr=6000,
    map_name=FLAGS.map_name
  )
  run_config = run_configs.get()
  replay_data = run_config.replay_data(path.join(
    FLAGS.replay_dir, FLAGS.replay_name + '.SC2Replay'
  ))

  with run_config.start(version=FLAGS.game_version) as controller:
    replay_info = controller.replay_info(replay_data)
    print(replay_info)
    controller.start_replay(sc_pb.RequestStartReplay(
      replay_data=replay_data,
      map_data=None,
      options=get_replay_actor_interface(FLAGS.map_name),
      observed_player_id=FLAGS.player_id,
      disable_fog=False)
    )

    controller.step()
    last_pb = None
    last_game_info = None
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    for _ in range(1000):
      pb_obs = controller.observe()
      game_info = controller.game_info()
      if last_pb is None:
        last_pb = pb_obs
        last_game_info = game_info
        continue
      if pb_obs.player_result:
        # episode ends, the zstat to this extent is what we need
        break
      # pb2all: X, A, weight as data
      data = converter.convert(
        pb=(last_pb, last_game_info),
        next_pb=(pb_obs, game_info)
      )
      if len(data) > 0:
        X, A = data[0][0]
        feed_dict = {}
        for key in X:
          feed_dict[inputs.X[key]] = [X[key]] * mycfg['batch_size']
        for key in A:
          feed_dict[inputs.A[key]] = [A[key]] * mycfg['batch_size']
        feed_dict[inputs.S] = (
            [np.array([0] * mycfg['hs_len'], dtype=np.float32)]
            * mycfg['batch_size']
        )
        feed_dict[inputs.M] = ([np.array(1, dtype=np.bool)]
                               * mycfg['batch_size'])
        loss = sess.run(out.loss.loss_endpoints, feed_dict)
        ab = A['A_AB']
        avail_actions = np.nonzero(X['MASK_AB'])[0]
        avail_selections = np.nonzero(X['MASK_SELECTION'][ab])[0]
        selection_units = A['A_SELECT']
        print('Avail action num: {}, select {}'.format(
          len(avail_actions), ZERG_ABILITIES[ab][0]))
        # print(ZERG_ABILITIES[ab][0])
        print('Avail unit num: {}, select {}'.format(
          len(avail_selections),
          selection_units[:sum([(i != 600) for i in selection_units])]))
        print('Loss endpoints: {}'.format(loss))
      # update & step the replay
      last_pb = pb_obs
      last_game_info = game_info
      controller.step(1)  # step_mul
    controller.quit()


def mnet_v6d6_run_rl_loss_test():
  mycfg = {
    'test': False,
    'use_self_fed_heads': False,
    'use_value_head': True,
    'use_loss_type': 'rl',
    'use_lstm': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'batch_size': 2,
    'rollout_len': 1,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'weight_decay': 0.0005,
    'arg_scope_type': 'mnet_v6_type_a',
    'use_base_mask': True,
    'vec_embed_version': 'v3',
    'last_act_embed_version': 'v2',
    'zstat_embed_version': 'v4d1',
    'trans_version': 'v4',
    'use_astar_glu': True,
    'use_astar_func_embed': True,
    'n_v': 5,
    'gather_batch': True,
    'merge_pi': False,
    'value_net_version': 'v4',
  }
  converter = PB2AllConverter(dict_space=True, zmaker_version='v5',
                              zstat_data_src=FLAGS.zstat_data_src,
                              game_version=FLAGS.game_version,
                              sort_executors='v1')
  ob_space, ac_space = converter.space.spaces

  # build the net
  nc = net_config_cls(ob_space, ac_space, **mycfg)
  nc.reward_weights = np.ones(shape=nc.reward_weights_shape, dtype=np.float32)
  inputs = net_inputs_placeholders_fun(nc)
  keys = list(inputs.X.keys())
  for key in keys:
    inputs.X['OPPO_'+key] = inputs.X[key]
  out = net_build_fun(inputs, nc, scope='mnet_v6d6_rl_loss')

  converter.reset(
    replay_name=FLAGS.replay_name,
    player_id=FLAGS.player_id,
    mmr=6000,
    map_name=FLAGS.map_name)
  run_config = run_configs.get()
  replay_data = run_config.replay_data(path.join(
    FLAGS.replay_dir, FLAGS.replay_name + '.SC2Replay'
  ))
  with run_config.start(version=FLAGS.game_version) as controller:
    replay_info = controller.replay_info(replay_data)
    print(replay_info)
    controller.start_replay(sc_pb.RequestStartReplay(
      replay_data=replay_data,
      map_data=None,
      options=get_replay_actor_interface(FLAGS.map_name),
      observed_player_id=FLAGS.player_id,
      disable_fog=False))
    controller.step()
    last_pb = None
    last_game_info = None
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    for _ in range(1000):
      pb_obs = controller.observe()
      game_info = controller.game_info()
      if last_pb is None:
        last_pb = pb_obs
        last_game_info = game_info
        continue
      if pb_obs.player_result:
        # episode end, the zstat to this extent is what we need
        break
      # pb2all: X, A, weights as data
      data = converter.convert(pb=(last_pb, last_game_info),
                               next_pb=(pb_obs, game_info))
      if data:
        feed_dict = {}
        X, A = data[0][0]
        for key in X:
          feed_dict[inputs.X[key]] = [X[key]] * mycfg['batch_size']
          # if 'OPPO_'+key in inputs.X:
          #   feed_dict[inputs.X['OPPO_'+key]] = [X[key]] * mycfg['batch_size']
        for key in A:
          feed_dict[inputs.A[key]] = [A[key]] * mycfg['batch_size']
          feed_dict[inputs.neglogp[key]] = [0] * mycfg['batch_size']
        feed_dict[inputs.V] = [[0] * nc.n_v] * mycfg['batch_size']
        feed_dict[inputs.R] = [[0] * nc.n_v] * mycfg['batch_size']
        feed_dict[inputs.S] = [np.array([0]*mycfg['hs_len'],
                                        dtype=np.float32)] * mycfg['batch_size']
        feed_dict[inputs.M] = [np.array(1, dtype=np.bool)] * mycfg['batch_size']
        loss = sess.run([out.loss.pg_loss,
                         out.loss.value_loss,
                         out.loss.entropy_loss], feed_dict)
        print('Loss endpoints: {}'.format(loss))
      # update and step the replay
      last_pb = pb_obs
      last_game_info = game_info
      controller.step(1)  # step_mul
    controller.quit()


def mnet_v6d6_run_vtrace_loss_test():
  mycfg = {
    'test': False,
    'use_self_fed_heads': False,
    'use_value_head': True,
    'use_loss_type': 'rl_vtrace',
    'use_lstm': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'batch_size': 4,
    'rollout_len': 2,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'weight_decay': 0.0005,
    'arg_scope_type': 'mnet_v6_type_a',
    'use_base_mask': True,
    'vec_embed_version': 'v3',
    'last_act_embed_version': 'v2',
    'zstat_embed_version': 'v7',
    'trans_version': 'v4',
    'use_astar_glu': True,
    'use_astar_func_embed': True,
    'n_v': 6,
    'lam': 0.8,
    'gather_batch': True,
    'merge_pi': False,
    'distillation': True,
  }
  converter = PB2AllConverter(dict_space=True, zmaker_version='v5',
                              zstat_data_src=FLAGS.zstat_data_src,
                              game_version=FLAGS.game_version,
                              sort_executors='v1')
  ob_space, ac_space = converter.space.spaces

  # build the net
  nc = net_config_cls(ob_space, ac_space, **mycfg)
  nc.reward_weights = np.ones(shape=nc.reward_weights_shape, dtype=np.float32)
  inputs = net_inputs_placeholders_fun(nc)
  keys = list(inputs.X.keys())
  for key in keys:
    inputs.X['OPPO_'+key] = inputs.X[key]
  out = net_build_fun(inputs, nc, scope='mnet_v6d6_rl_vtrace_loss')

  converter.reset(
    replay_name=FLAGS.replay_name,
    player_id=FLAGS.player_id,
    mmr=6000,
    map_name=FLAGS.map_name)
  run_config = run_configs.get()
  replay_data = run_config.replay_data(path.join(
    FLAGS.replay_dir, FLAGS.replay_name + '.SC2Replay'
  ))
  with run_config.start(version=FLAGS.game_version) as controller:
    replay_info = controller.replay_info(replay_data)
    print(replay_info)
    controller.start_replay(sc_pb.RequestStartReplay(
      replay_data=replay_data,
      map_data=None,
      options=get_replay_actor_interface(FLAGS.map_name),
      observed_player_id=FLAGS.player_id,
      disable_fog=False))
    controller.step()
    last_pb = None
    last_game_info = None
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    for _ in range(1000):
      pb_obs = controller.observe()
      game_info = controller.game_info()
      if last_pb is None:
        last_pb = pb_obs
        last_game_info = game_info
        continue
      if pb_obs.player_result:
        # episode end, the zstat to this extent is what we need
        break
      # pb2all: X, A, weights as data
      data = converter.convert(pb=(last_pb, last_game_info),
                               next_pb=(pb_obs, game_info))
      if data:
        feed_dict = {}
        X, A = data[0][0]
        for key in X:
          feed_dict[inputs.X[key]] = [X[key]] * mycfg['batch_size']
          # if 'OPPO_'+key in inputs.X:
          #   feed_dict[inputs.X['OPPO_'+key]] = [X[key]] * mycfg['batch_size']
        for key in A:
          feed_dict[inputs.A[key]] = [A[key]] * mycfg['batch_size']
          feed_dict[inputs.neglogp[key]] = [0] * mycfg['batch_size']
          feed_dict[inputs.flatparam[key]] = np.zeros(shape=inputs.flatparam[key].shape)
        feed_dict[inputs.r] = [[0] * nc.n_v] * mycfg['batch_size']
        feed_dict[inputs.discount] = [1] * mycfg['batch_size']
        feed_dict[inputs.S] = [np.array([0]*mycfg['hs_len'],
                                        dtype=np.float32)] * mycfg['batch_size']
        feed_dict[inputs.M] = [np.array(1, dtype=np.bool)] * mycfg['batch_size']
        loss = sess.run([out.loss.pg_loss,
                         out.loss.value_loss,
                         out.loss.entropy_loss,
                         out.loss.loss_endpoints], feed_dict)
        print('Loss endpoints: {}'.format(loss))
      # update and step the replay
      last_pb = pb_obs
      last_game_info = game_info
      controller.step(1)  # step_mul
    controller.quit()


def mnet_v6d6_run_ppo2_loss_test():
  mycfg = {
    'test': False,
    'use_self_fed_heads': False,
    'use_value_head': True,
    'use_loss_type': 'rl_ppo2',
    'use_lstm': True,
    'lstm_cell_type': 'lstm',
    'lstm_layer_norm': True,
    'batch_size': 8,
    'rollout_len': 4,
    'lstm_duration': 1,
    'nlstm': 256,
    'hs_len': 256 * 2,
    'weight_decay': 0.0005,
    'arg_scope_type': 'mnet_v6_type_a',
    'use_base_mask': True,
    'vec_embed_version': 'v3',
    'last_act_embed_version': 'v2',
    'zstat_embed_version': 'v4d1',
    'zstat_index_base_wavelen': 555.6,
    'trans_version': 'v4',
    'use_astar_glu': True,
    'use_astar_func_embed': True,
    'n_v': 5,
    'lam': 0.8,
    'gather_batch': True,
    'merge_pi': False,
    'distillation': True,
    'value_net_version': 'trans_v1',
  }
  converter = PB2AllConverter(dict_space=True, zmaker_version='v5',
                              zstat_data_src=FLAGS.zstat_data_src,
                              game_version=FLAGS.game_version,
                              sort_executors='v1')
  ob_space, ac_space = converter.space.spaces

  # build the net
  nc = net_config_cls(ob_space, ac_space, **mycfg)
  nc.reward_weights = np.ones(shape=nc.reward_weights_shape, dtype=np.float32)
  inputs = net_inputs_placeholders_fun(nc)
  keys = list(inputs.X.keys())
  for key in keys:
    inputs.X['OPPO_'+key] = inputs.X[key]
  out = net_build_fun(inputs, nc, scope='mnet_v6d6_rl_ppo2_loss')

  converter.reset(
    replay_name=FLAGS.replay_name,
    player_id=FLAGS.player_id,
    mmr=6000,
    map_name=FLAGS.map_name)
  run_config = run_configs.get()
  replay_data = run_config.replay_data(path.join(
    FLAGS.replay_dir, FLAGS.replay_name + '.SC2Replay'
  ))
  with run_config.start(version=FLAGS.game_version) as controller:
    replay_info = controller.replay_info(replay_data)
    print(replay_info)
    controller.start_replay(sc_pb.RequestStartReplay(
      replay_data=replay_data,
      map_data=None,
      options=get_replay_actor_interface(FLAGS.map_name),
      observed_player_id=FLAGS.player_id,
      disable_fog=False))
    controller.step()
    last_pb = None
    last_game_info = None
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    while True:
      pb_obs = controller.observe()
      game_info = controller.game_info()
      if last_pb is None:
        last_pb = pb_obs
        last_game_info = game_info
        continue
      if pb_obs.player_result:
        # episode end, the zstat to this extent is what we need
        break
      # pb2all: X, A, weights as data
      data = converter.convert(pb=(last_pb, last_game_info),
                               next_pb=(pb_obs, game_info))
      if data:
        feed_dict = {}
        X, A = data[0][0]
        for key in X:
          feed_dict[inputs.X[key]] = [X[key]] * mycfg['batch_size']
          # if 'OPPO_'+key in inputs.X:
          #   feed_dict[inputs.X['OPPO_'+key]] = [X[key]] * mycfg['batch_size']
        for key in A:
          feed_dict[inputs.A[key]] = [A[key]] * mycfg['batch_size']
          feed_dict[inputs.neglogp[key]] = [0] * mycfg['batch_size']
          feed_dict[inputs.flatparam[key]] = np.zeros(shape=inputs.flatparam[key].shape)
        feed_dict[inputs.r] = [[0] * nc.n_v] * mycfg['batch_size']
        feed_dict[inputs.discount] = [1] * mycfg['batch_size']
        feed_dict[inputs.S] = [np.array([0]*mycfg['hs_len'],
                                        dtype=np.float32)] * mycfg['batch_size']
        feed_dict[inputs.M] = [np.array(1, dtype=np.bool)] * mycfg['batch_size']
        loss = sess.run([out.loss.pg_loss,
                         out.loss.value_loss,
                         out.loss.entropy_loss,
                         out.loss.loss_endpoints], feed_dict)
        print('Loss endpoints: {}'.format(loss))
      # update and step the replay
      last_pb = pb_obs
      last_game_info = game_info
      controller.step(1)  # step_mul


def main(_):
  # mnet_v6d6_inputs_test()
  # mnet_v6d6_test()
  # mnet_v6d6_endpoints_test()
  # mnet_v6d6_run_il_loss_test()
  mnet_v6d6_run_rl_loss_test()
  # mnet_v6d6_run_vtrace_loss_test()
  # mnet_v6d6_run_ppo2_loss_test()
  # pass


if __name__ == '__main__':
  app.run(main)
