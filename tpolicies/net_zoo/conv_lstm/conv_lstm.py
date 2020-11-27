from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.layers.python.layers import utils as lutils
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import nest

import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
import tpolicies.ops as tp_ops
import tpolicies.tp_utils as tp_utils
from tpolicies.utils.distributions import CategoricalPdType
from tpolicies.net_zoo.conv_lstm.data import ConvLstmConfig
from tpolicies.net_zoo.conv_lstm.data import ConvLstmTrainableVariables
from tpolicies.net_zoo.conv_lstm.data import ConvLstmInputs
from tpolicies.net_zoo.conv_lstm.data import ConvLstmOutputs
from tpolicies.net_zoo.conv_lstm.data import ConvLstmLosses


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
    inputs_hs = tf.reshape(inputs_hs, [nc.nrollout, nc.rollout_len, nc.hs_len])
    initial_hs = inputs_hs[:, 0, :]
    lstm_embed, hs_new = tp_layers.lstm(inputs_x_seq=x_seq,
                                        inputs_terminal_mask_seq=hsm_seq,
                                        inputs_state=initial_hs,
                                        nh=nc.nlstm,
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
  vf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'vf'))
  pf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'action'))
  return ConvLstmTrainableVariables(all_vars=all_vars, lstm_vars=lstm_vars, vf_vars=vf_vars, pf_vars=pf_vars)


def conv_lstm_inputs_placeholder(nc: ConvLstmConfig):
  """create the inputs placeholder for conv_lstm"""
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
  n_v = 1  # no. of value heads
  R = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'R')
  V = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'V')
  S = tf.placeholder(tf.float32, (nc.batch_size, nc.hs_len), 'hs')
  M = tf.placeholder(tf.float32, (nc.batch_size,), 'hsm')

  return ConvLstmInputs(
    X=X_ph,
    A=A_ph,
    neglogp=neglogp,
    R=R,
    V=V,
    S=S,
    M=M
  )


def conv_lstm(inputs: ConvLstmInputs,
              nc: ConvLstmConfig,
              scope=None) -> ConvLstmOutputs:
  """create the whole net for conv-lstm"""
  with tf.variable_scope(scope, default_name='conv_lstm') as sc:
    # NOTE: use name_scope, in case multiple parameter-sharing nets are built
    net_name_scope = tf.get_default_graph().get_name_scope()
    endpoints_collections = net_name_scope + '_endpoints'
    x = inputs.X
    # make body
    with tf.variable_scope('body'):
      # # 168
      # x = tfc_layers.conv2d(x, filters=32, kernel_size=[8, 8], strides=2, scope='conv0')
      # # 84
      # x = tfc_layers.conv2d(x, filters=64, kernel_size=[4, 4], strides=2, scope='conv1')
      # # 42
      # x = tfc_layers.max_pool2d(x, [2, 2], scope='pool1')
      # # 21
      # x = tfc_layers.conv2d(x, filters=64, kernel_size=[3, 3], scope='conv2')
      # x = tfc_layers.max_pool2d(x, [2, 2], scope='pool2')
      # # 10
      # x = tfc_layers.flatten(x)
      x = tfc_layers.conv2d(x, nc.spa_ch_dim, (2, 2), [3, 3], scope='conv0')
      x = tfc_layers.conv2d(x, nc.spa_ch_dim, [3, 3], scope='conv1')
      x = tfc_layers.max_pool2d(x, [2, 2], scope='pool1')
      x = tfc_layers.conv2d(x, nc.spa_ch_dim, [3, 3], scope='conv2')
      x = tfc_layers.max_pool2d(x, [2, 2], scope='pool2')
      x = tfc_layers.flatten(x)
      if nc.use_lstm:
        with tf.variable_scope('lstm_embed'):
          x, hs_new = _lstm_embed_block(inputs_x=x,
                                        inputs_hs=inputs.S,
                                        inputs_mask=inputs.M,
                                        nc=nc)
    # make action head
    with tf.variable_scope('action'):
      head = tp_layers.discrete_action_head(
        inputs=x,
        n_actions=nc.ac_space.n,
        pdtype_cls=CategoricalPdType
      )
    # make value head
    vf = None
    if nc.use_value_head:
      with tf.variable_scope('vf'):
        vf = tfc_layers.fully_connected(x, 64)
        vf = tfc_layers.fully_connected(vf, 1, activation_fn=None,
                                        normalizer_fn=None)
    # make loss
    loss = None
    if nc.use_loss_type == 'rl':
      # regularization loss
      total_reg_loss = tf.losses.get_regularization_losses(scope=sc.name)
      with tf.variable_scope('losses'):
        # ppo loss
        neglogp = head.pd.neglogp(inputs.A)
        pg_loss, value_loss = tp_losses.ppo_loss(
          neglogp=neglogp,
          oldneglogp=inputs.neglogp,
          vpred=vf,
          R=inputs.R,
          V=inputs.V,
          masks=None,
          reward_weights=None,
          adv_normalize=True,
          sync_statistics=nc.sync_statistics
        )
        # entropy loss
        entropy_loss = head.ent
        loss_endpoints = {'pg_loss': tf.reduce_mean(pg_loss),
                          'value_loss': tf.reduce_mean(value_loss),
                          'entropy_loss': tf.reduce_mean(entropy_loss)
                          }
        loss = ConvLstmLosses(
          total_reg_loss=total_reg_loss,
          pg_loss=pg_loss,
          value_loss=tf.reduce_mean(value_loss),
          entropy_loss=entropy_loss,
          loss_endpoints=loss_endpoints
        )
    # collect vars, endpoints, etc.
    trainable_vars = _make_vars(sc)
    endpoints = OrderedDict()  # TODO
  return ConvLstmOutputs(
    self_fed_heads=head,
    outer_fed_heads=head,
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
