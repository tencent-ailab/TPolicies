from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.framework import nest

import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
import tpolicies.tp_utils as tp_utils
from tpolicies.utils.distributions import make_pdtype
from tpolicies.net_zoo.soccer.data import ContNNConfig
from tpolicies.net_zoo.soccer.data import ContNNTrainableVariables
from tpolicies.net_zoo.soccer.data import ContNNInputs
from tpolicies.net_zoo.soccer.data import ContNNOutputs
from tpolicies.net_zoo.soccer.data import ContNNLosses
from tpolicies.utils.distributions import DiagGaussianPdType


def _make_vars(scope) -> ContNNTrainableVariables:
  scope = scope if isinstance(scope, str) else scope.name + '/'
  all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
  vf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'vf'))
  pf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'action'))
  return ContNNTrainableVariables(all_vars=all_vars, vf_vars=vf_vars, pf_vars=pf_vars)

def cont_nn_inputs_placeholder(nc: ContNNConfig):
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
  return ContNNInputs(
    X=X_ph,
    A=A_ph,
    neglogp=neglogp,
    R=R,
    V=V
  )


def cont_nn(inputs: ContNNInputs,
              nc: ContNNConfig,
              scope=None) -> ContNNOutputs:
  """create the whole net for conv-lstm"""
  with tf.variable_scope(scope, default_name='soccer') as sc:
    # NOTE: use name_scope, in case multiple parameter-sharing nets are built
    net_name_scope = tf.get_default_graph().get_name_scope()
    endpoints_collections = net_name_scope + '_endpoints'
    X = inputs.X
    if nc.n_player == 1:
      X = (X,)
      ac_spaces = (nc.ac_space,)
    else:
      ac_spaces = tuple(nc.ac_space.spaces)
    y = []
    heads = []
    for input, ac_space in zip(X, ac_spaces):
      with tf.variable_scope('body', reuse=tf.AUTO_REUSE):
        x = tfc_layers.fully_connected(input, nc.spa_ch_dim, activation_fn=tf.nn.relu, scope="fc1")
        x = tfc_layers.fully_connected(x, nc.spa_ch_dim, activation_fn=tf.nn.relu, scope="fc2")
        x = tfc_layers.fully_connected(x, nc.spa_ch_dim, activation_fn=tf.nn.relu, scope="fc3")
        x = tfc_layers.fully_connected(x, nc.spa_ch_dim, activation_fn=tf.nn.relu, scope="fc4")
        y.append(x)

      # make action head
      with tf.variable_scope('action', reuse=tf.AUTO_REUSE):
        pdtype = make_pdtype(ac_space)
        pdprams = tfc_layers.fully_connected(x, pdtype.param_shape()[0], #ac_space.shape[0],
                                             activation_fn=None,
                                             normalizer_fn=None,
                                             scope='pdprams')
        head = tp_layers.to_action_head(pdprams, DiagGaussianPdType)
        heads.append(head)

    y = tf.concat(y, axis=1)
    heads = tp_utils.pack_sequence_as_structure_like_gym_space(nc.ac_space,
                                                               heads)
    if nc.n_player == 1:
      heads = heads[0]                                                     
    # make value head
    vf = None
    if nc.use_value_head:
      with tf.variable_scope('vf'):
        vf = tfc_layers.fully_connected(y, nc.spa_ch_dim * 4)
        vf = tfc_layers.fully_connected(vf, nc.spa_ch_dim * 2)
        vf = tfc_layers.fully_connected(vf, nc.n_v, activation_fn=None,
                                        normalizer_fn=None)
    # make loss
    loss = None
    if nc.use_loss_type == 'rl':
      # regularization loss
      total_reg_loss = tf.losses.get_regularization_losses(scope=sc.name)
      with tf.variable_scope('losses'):
        # ppo loss
        assert nc.n_player == 1
        neglogp = head.pd.neglogp(inputs.A)
        ppo_loss, value_loss = tp_losses.ppo_loss(
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
        loss_endpoints = {}
        loss = ContNNLosses(
          total_reg_loss=total_reg_loss,
          pg_loss=ppo_loss,
          value_loss=value_loss,
          entropy_loss=entropy_loss,
          loss_endpoints=loss_endpoints
        )
        # collect vars, endpoints, etc.
    trainable_vars = _make_vars(sc)
    endpoints = OrderedDict()  # TODO
  return ContNNOutputs(
    self_fed_heads=heads,
    outer_fed_heads=heads,
    loss=loss,
    vars=trainable_vars,
    endpoints=endpoints,
    value_head=vf
  )


# APIs
net_build_fun = cont_nn
net_config_cls = ContNNConfig
net_inputs_placeholders_fun = cont_nn_inputs_placeholder
