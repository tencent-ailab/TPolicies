from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.layers.python.layers import utils as lutils
from tensorflow.contrib.framework import add_arg_scope
from gym import spaces

import tpolicies.layers as tp_layers
import tpolicies.ops as tp_ops
import tpolicies.tp_utils as tp_utils
from tpolicies.utils.distributions import DiagGaussianPdType
from tpolicies.net_zoo.gym_ddpg.data import DDPGConfig
from tpolicies.net_zoo.gym_ddpg.data import DDPGTrainableVariables
from tpolicies.net_zoo.gym_ddpg.data import DDPGInputs
from tpolicies.net_zoo.gym_ddpg.data import DDPGOutputs
from tpolicies.net_zoo.gym_ddpg.data import DDPGLosses


def _make_vars(scope) -> DDPGTrainableVariables:
  scope = scope if isinstance(scope, str) else scope.name + '/'
  all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
  lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'lstm_embed'))
  return DDPGTrainableVariables(all_vars=all_vars, lstm_vars=lstm_vars)


def ddpg_inputs_placeholder(nc: DDPGConfig):
  """create the inputs placeholder for gym_ddpg"""
  X_ph = tp_utils.placeholders_from_gym_space(
    nc.ob_space, batch_size=nc.batch_size, name='ob_ph')

  if nc.test:
    # when testing, there are no ground-truth actions
    A_ph = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
  else:
    A_ph = tp_utils.placeholders_from_gym_space(
      nc.ac_space, batch_size=nc.batch_size, name='ac_ph')

  n_v = nc.n_v
  r = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'r')
  discount = tf.placeholder(tf.float32, (nc.batch_size,), 'discount')
  S = tf.placeholder(tf.float32, (nc.batch_size, nc.hs_len), 'hs')
  M = tf.placeholder(tf.float32, (nc.batch_size,), 'hsm')

  return DDPGInputs(
    X=X_ph,
    A=A_ph,
    S=S,
    M=M,
    r=r,
    discount=discount,
  )


def gym_ddpg(inputs: DDPGInputs,
             nc: DDPGConfig,
             scope=None) -> DDPGOutputs:
  """create the whole net for gym-ddpg"""
  X = inputs.X
  ac_space = nc.ac_space
  assert isinstance(ac_space, spaces.Box) and len(ac_space.shape) == 1
  def build_ac():
    # make body
    with tf.variable_scope('body'):
      x = tfc_layers.fully_connected(X, nc.fc_ch_dim, scope='fc0')
      hs = None
      if nc.use_lstm:
        with tf.variable_scope('lstm_embed'):
          x, hs = tp_layers.lstm_embed_block(inputs_x=x, inputs_hs=inputs.S,
                                             inputs_mask=inputs.M, nc=nc)
      # make action head
      with tf.variable_scope('action', reuse=tf.AUTO_REUSE):
        size = ac_space.shape[0]
        mean = tfc_layers.fully_connected(x, size,
                                          activation_fn=tf.tanh,
                                          normalizer_fn=None,
                                          scope='mean',)
        mean = (ac_space.high + ac_space.low) * 0.5 + mean * (ac_space.high - ac_space.low) * 0.5
        logstd = tf.get_variable(name='logstd', shape=[1, size],
                                 initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        head = tp_layers.to_action_head(pdparam, DiagGaussianPdType)
        head = head._replace(sam=tf.clip_by_value(head.sam, ac_space.low, ac_space.high))
    # make value head
    self_vf = None
    outer_vf = None
    if nc.use_value_head:
      with tf.variable_scope('vf'):
        self_vf = tfc_layers.fully_connected(
          tf.concat([X, head.argmax], axis=-1), nc.fc_ch_dim)
        self_vf = tfc_layers.fully_connected(self_vf, nc.n_v,
                                             activation_fn=None,
                                             normalizer_fn=None)
      with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
        outer_vf = tfc_layers.fully_connected(
          tf.concat([X, inputs.A], axis=-1), nc.fc_ch_dim)
        outer_vf = tfc_layers.fully_connected(outer_vf, nc.n_v,
                                              activation_fn=None,
                                              normalizer_fn=None)
    return head, self_vf, outer_vf, hs, logstd

  with tf.variable_scope(scope, default_name='gym_ddpg') as sc:
    # NOTE: use name_scope, in case multiple parameter-sharing nets are built
    head, self_vf, outer_vf, hs, logstd = build_ac()
    # collect vars, endpoints, etc.
    trainable_vars = _make_vars(sc)

  if nc.use_target_net:
    with tf.variable_scope('target_model') as sc:
      _, target_vf, _, _, _ = build_ac()

  # make loss
  loss = None
  if nc.use_loss_type == 'rl':
    with tf.variable_scope('losses'):
      # regularization loss
      total_reg_loss = tf.losses.get_regularization_losses(scope=sc.name)
      # entropy loss
      entropy_loss = tf.reduce_mean(head.ent)
      loss_endpoints = {}
      loss_endpoints['ent'] = entropy_loss
      # pg loss
      def _batch_to_TB(tsr):
        return tf.transpose(tf.reshape(
          tsr, shape=(nc.nrollout, nc.rollout_len)))

      vpred_list = [_batch_to_TB(v)
                    for v in tf.split(outer_vf, nc.n_v, axis=1)]
      target_v_list = [_batch_to_TB(v)
                       for v in tf.split(target_vf, nc.n_v, axis=1)]
      reward_list = [_batch_to_TB(r)
                     for r in tf.split(inputs.r, nc.n_v, axis=1)]
      discounts = _batch_to_TB(inputs.discount)
      value_loss = []
      for values, rewards, target_v in zip(vpred_list, reward_list, target_v_list):
        target_q = rewards[:-1] + discounts[:-1] * target_v[1:]
        value_loss.append(tf.reduce_mean(
          0.5 * tf.square(tf.stop_gradient(target_q) - values[:-1])))
      value_loss = tf.stack(value_loss)

      if nc.reward_weights is not None:
        self_vf = tf.matmul(self_vf, nc.reward_weights,
                            transpose_b=True)
      pg_loss = - tf.reduce_mean(self_vf)
      loss_endpoints['pg_loss'] = pg_loss
      loss_endpoints['std'] = tf.exp(logstd)
      if len(value_loss.shape) == 0:
        loss_endpoints['value_loss'] = value_loss
      else:
        for i in range(value_loss.shape[0]):
          loss_endpoints['value_loss_' + str(i)] = value_loss[i]
      loss = DDPGLosses(
        total_reg_loss=total_reg_loss,
        pg_loss=pg_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        loss_endpoints=loss_endpoints
      )
  endpoints = OrderedDict()  # TODO
  return DDPGOutputs(
    self_fed_heads=head,
    outer_fed_heads=head,
    S=hs,
    loss=loss,
    vars=trainable_vars,
    endpoints=endpoints,
    self_fed_value_head=self_vf,
    outer_fed_value_head=outer_vf,
  )


# APIs
net_build_fun = gym_ddpg
net_config_cls = DDPGConfig
net_inputs_placeholders_fun = ddpg_inputs_placeholder
