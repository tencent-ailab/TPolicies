from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.layers.python.layers import utils as lutils
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import nest

import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
import tpolicies.ops as tp_ops
import tpolicies.tp_utils as tp_utils
from tpolicies.utils.distributions import CategoricalPdType
from tpolicies.net_zoo.pommerman.data import ConvLstmConfig
from tpolicies.net_zoo.pommerman.data import ConvLstmTrainableVariables
from tpolicies.net_zoo.pommerman.data import ConvLstmInputs
from tpolicies.net_zoo.pommerman.data import ConvLstmOutputs
from tpolicies.net_zoo.pommerman.data import ConvLstmLosses


@add_arg_scope
def _lstm_embed_block(inputs_x, inputs_hs, inputs_mask, nc,
                      outputs_collections=None):
  """ lstm embedding block.

  Args
    inputs_x: current state - (nrollout*rollout_len, input_dim)
    inputs_hs: hidden state - (nrollout*rollout_len, hs_len), NOTE: it's the
    states at every time steps of the rollout.
    inputs_mask: hidden state mask - (nrollout*rollout_len,)
    nc:

  Returns
    A Tensor, the lstm embedding outputs - (nrollout*rollout_len, out_idm)
    A Tensor, the new hidden state - (nrollout, hs_len), NOTE: it's the state at
     a single time step.
  """
  with tf.variable_scope('lstm_embed') as sc:
    # add dropout before LSTM cell
    if 1 > nc.lstm_dropout_rate > 0 and not nc.test:
      inputs_x = tf.nn.dropout(inputs_x, keep_prob=1 - nc.lstm_dropout_rate)

    # to list sequence and call the lstm cell
    x_seq = tp_ops.batch_to_seq(inputs_x, nc.nrollout, nc.rollout_len)
    hsm_seq = tp_ops.batch_to_seq(tp_ops.to_float32(inputs_mask),
                                  nc.nrollout, nc.rollout_len)
    inputs_hs = tf.reshape(inputs_hs, [nc.nrollout, nc.rollout_len,
                                       int(nc.hs_len/nc.n_player)])
    initial_hs = inputs_hs[:, 0, :]
    lstm_embed, hs_new = tp_layers.lstm(inputs_x_seq=x_seq,
                                        inputs_terminal_mask_seq=hsm_seq,
                                        inputs_state=initial_hs,
                                        nh=nc.nlstm/nc.n_player,
                                        forget_bias=nc.forget_bias,
                                        use_layer_norm=nc.lstm_layer_norm,
                                        scope='lstm')
    lstm_embed = tp_ops.seq_to_batch(lstm_embed)

    # add dropout after LSTM cell
    if 1 > nc.lstm_dropout_rate > 0 and not nc.test:
      lstm_embed = tf.nn.dropout(lstm_embed, keep_prob=1 - nc.lstm_dropout_rate)
    return (
      lutils.collect_named_outputs(outputs_collections, sc.name + '_out',
                                   lstm_embed),
      lutils.collect_named_outputs(outputs_collections, sc.name + '_hs', hs_new)
    )


def _make_vars(scope) -> ConvLstmTrainableVariables:
  scope = scope if isinstance(scope, str) else scope.name + '/'
  all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
  lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'lstm_embed'))
  return ConvLstmTrainableVariables(all_vars=all_vars, lstm_vars=lstm_vars)


def conv_lstm_inputs_placeholder(nc: ConvLstmConfig):
  """create the inputs placeholder for pommerman"""
  X_ph = tp_utils.placeholders_from_gym_space(
    nc.ob_space, batch_size=nc.batch_size, name='ob_ph')

  if nc.test:
    # when testing, there are no ground-truth actions
    A_ph = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
  else:
    A_ph = tp_utils.placeholders_from_gym_space(
      nc.ac_space, batch_size=nc.batch_size, name='ac_ph')

  neglogp = tp_utils.map_gym_space_to_structure(
    func=lambda x_sp: tf.placeholder(shape=(nc.batch_size, ),
                                     dtype=tf.float32,
                                     name='neglogp'),
    gym_sp=nc.ac_space
  )
  n_v = nc.n_v
  R = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'R')
  V = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'V')
  r = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'r')
  discount = tf.placeholder(tf.float32, (nc.batch_size,), 'discount')
  S = tf.placeholder(tf.float32, (nc.batch_size, nc.hs_len), 'hs')
  M = tf.placeholder(tf.float32, (nc.batch_size,), 'hsm')

  return ConvLstmInputs(
    X=X_ph,
    A=A_ph,
    neglogp=neglogp,
    R=R,
    V=V,
    S=S,
    M=M,
    r=r,
    discount=discount,
  )


def conv_lstm(inputs: ConvLstmInputs,
              nc: ConvLstmConfig,
              scope=None) -> ConvLstmOutputs:
  """create the whole net for conv-lstm"""
  with tf.variable_scope(scope, default_name='pommerman') as sc:
    # NOTE: use name_scope, in case multiple parameter-sharing nets are built
    net_name_scope = tf.get_default_graph().get_name_scope()
    endpoints_collections = net_name_scope + '_endpoints'
    X = inputs.X
    if nc.n_player == 1:
      X = (X,)
      ac_spaces = (nc.ac_space,)
    else:
      ac_spaces = tuple(nc.ac_space.spaces)
    S = tf.split(inputs.S, nc.n_player, axis=1)
    # make body
    y = []
    hs_new = []
    heads = []
    for input, s, ac_space in zip(X, S, ac_spaces):
      with tf.variable_scope('body', reuse=tf.AUTO_REUSE):
        x = tfc_layers.conv2d(input[0], nc.spa_ch_dim, [3, 3], scope='conv0')
        x = tfc_layers.conv2d(x, nc.spa_ch_dim, [5, 5], scope='conv1')
        x = tfc_layers.conv2d(x, nc.spa_ch_dim * 2, [3, 3], scope='conv2')
        x = tfc_layers.conv2d(x, nc.spa_ch_dim * 2, [5, 5], scope='conv3')
        x = tfc_layers.conv2d(x, nc.spa_ch_dim * 4, [3, 3], scope='conv4')
        pos = tf.to_int32(input[1])
        ind = tf.concat(
          [tf.expand_dims(tf.range(nc.batch_size), 1), pos], axis=1)
        x = tf.gather_nd(x, ind)
        if nc.use_lstm:
          with tf.variable_scope('lstm_embed'):
            x, hs = _lstm_embed_block(inputs_x=x,
                                      inputs_hs=s,
                                      inputs_mask=inputs.M,
                                      nc=nc)
            hs_new.append(hs)
        y.append(x)

      # make action head
      with tf.variable_scope('action', reuse=tf.AUTO_REUSE):
        head_logits = tfc_layers.fully_connected(x, ac_space.n,
                                                 activation_fn=None,
                                                 normalizer_fn=None,
                                                 scope='logits')
        if len(input) > 1:
          head_logits = tp_ops.mask_logits(head_logits, input[2])
        head = tp_layers.to_action_head(head_logits, CategoricalPdType)
        heads.append(head)

    if nc.use_lstm:
      hs_new = tf.concat(hs_new, axis=1)
    y = tf.concat(y, axis=1)
    heads = tp_utils.pack_sequence_as_structure_like_gym_space(nc.ac_space,
                                                               heads)
    if nc.n_player == 1:
      heads = heads[0]
    # make value head
    vf = None
    if nc.use_value_head:
      assert nc.n_player == 2
      with tf.variable_scope('vf'):
        vf = tfc_layers.fully_connected(y, nc.spa_ch_dim * 4)
        vf = tfc_layers.fully_connected(vf, nc.spa_ch_dim * 2)
        vf = tfc_layers.fully_connected(vf, nc.n_v, activation_fn=None,
                                        normalizer_fn=None)
    # make loss
    loss = None
    if nc.use_loss_type in ['rl', 'rl_ppo', 'rl_vtrace']:
      assert nc.n_player == 2
      with tf.variable_scope('losses'):
        # regularization loss
        total_reg_loss = tf.losses.get_regularization_losses(scope=sc.name)
        # entropy loss
        entropy_loss = nest.map_structure_up_to(
          ac_spaces, lambda head: tf.reduce_mean(head.ent), heads)
        # ppo loss
        neglogp = nest.map_structure_up_to(
          ac_spaces, lambda head, ac: head.pd.neglogp(ac), heads, inputs.A)
        loss_endpoints = {}
        for k, v in enumerate(entropy_loss):
          loss_endpoints['ent_' + str(k)] = v
        if nc.use_loss_type == 'rl' or nc.use_loss_type == 'rl_ppo':
          pg_loss, value_loss = tp_losses.ppo_loss(
            neglogp=neglogp,
            oldneglogp=inputs.neglogp,
            vpred=vf,
            R=inputs.R,
            V=inputs.V,
            masks=None,
            reward_weights=nc.reward_weights,
            adv_normalize=True,
            sync_statistics=nc.sync_statistics
          )
        elif nc.use_loss_type == 'rl_vtrace':
          def _batch_to_TB(tsr):
            return tf.transpose(tf.reshape(
              tsr, shape=(nc.nrollout, nc.rollout_len)))

          lam = tf.convert_to_tensor(nc.lam, tf.float32)
          vpred_list = [_batch_to_TB(v)
                        for v in tf.split(vf, nc.n_v, axis=1)]
          reward_list = [_batch_to_TB(r)
                         for r in tf.split(inputs.r, nc.n_v, axis=1)]
          discounts = _batch_to_TB(inputs.discount)
          value_loss = []
          for values, rewards in zip(vpred_list, reward_list):
            value_loss.append(
              tp_losses.td_lambda(values, rewards, discounts, lam=lam))
          value_loss = tf.stack(value_loss)

          neglogp_list = [_batch_to_TB(neglogp)
                          for neglogp in nest.flatten(neglogp)]
          oldneglogp_list = [_batch_to_TB(oldneglogp)
                             for oldneglogp in nest.flatten(inputs.neglogp)]
          shaped_values = tf.matmul(vf, nc.reward_weights,
                                    transpose_b=True)
          shaped_rewards = tf.matmul(inputs.r, nc.reward_weights,
                                     transpose_b=True)
          values = tf.transpose(tf.reshape(
            shaped_values, shape=(nc.nrollout, nc.rollout_len)))
          rewards = tf.transpose(tf.reshape(
            shaped_rewards, shape=(nc.nrollout, nc.rollout_len)))
          pg_loss = tf.reduce_sum(
            [tp_losses.vtrace_loss(neglogp, oldneglogp, None, values,
                                   rewards, discounts, 1.0, 1.0)
             for oldneglogp, neglogp in zip(oldneglogp_list, neglogp_list)])
          upgo_loss = tp_losses.upgo_loss(tf.stack(neglogp_list, axis=-1),
                                          tf.stack(oldneglogp_list, axis=-1),
                                          None, vpred_list[0], reward_list[0],
                                          discounts)
          loss_endpoints['upgo_loss'] = upgo_loss
        loss_endpoints['pg_loss'] = pg_loss
        if len(value_loss.shape) == 0:
          loss_endpoints['value_loss'] = value_loss
        else:
          for i in range(value_loss.shape[0]):
            loss_endpoints['value_loss_' + str(i)] = value_loss[i]
        loss = ConvLstmLosses(
          total_reg_loss=total_reg_loss,
          pg_loss=pg_loss,
          value_loss=value_loss,
          entropy_loss=entropy_loss,
          loss_endpoints=loss_endpoints
        )
        # collect vars, endpoints, etc.
    trainable_vars = _make_vars(sc)
    endpoints = OrderedDict()  # TODO
  return ConvLstmOutputs(
    self_fed_heads=heads,
    outer_fed_heads=heads,
    S=hs_new,
    loss=loss,
    vars=trainable_vars,
    endpoints=endpoints,
    value_head=vf
  )


# APIs
net_build_fun = conv_lstm
net_config_cls = ConvLstmConfig
net_inputs_placeholders_fun = conv_lstm_inputs_placeholder
