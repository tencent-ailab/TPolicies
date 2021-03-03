import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from collections import OrderedDict

import tpolicies.tp_utils as tp_utils
import tpolicies.layers as tp_layers
import tpolicies.losses as tp_losses
from tpolicies.utils.distributions import make_pdtype
from tpolicies.utils.distributions import CategoricalPdType
from tpolicies.utils.distributions import DiagGaussianPdType
from tpolicies.net_zoo.gym_classic_control.data import MlpTrainableVariables
from tpolicies.net_zoo.gym_classic_control.data import MlpConfig
from tpolicies.net_zoo.gym_classic_control.data import MlpInputs, MlpOutputs
from tpolicies.net_zoo.gym_classic_control.data import MlpLosses


def _make_vars(scope) -> MlpTrainableVariables:
    scope = scope if isinstance(scope, str) else scope.name + '/'
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    vf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'vf'))
    pf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'action'))
    return MlpTrainableVariables(all_vars=all_vars, vf_vars=vf_vars, pf_vars=pf_vars)


def mlp_inputs_placeholder(nc: MlpConfig):
    """create the inputs placeholder for mlp"""
    X_ph = tp_utils.placeholders_from_gym_space(
        nc.ob_space, batch_size=nc.batch_size, name='ob_ph')

    if nc.test:
        # when testing, there are no ground-truth actions
        A_ph = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
    else:
        A_ph = tp_utils.placeholders_from_gym_space(
            nc.ac_space, batch_size=nc.batch_size, name='ac_ph')

    neglogp = tp_utils.map_gym_space_to_structure(
        func=lambda x_sp: tf.placeholder(shape=(nc.batch_size,),
                                         dtype=tf.float32,
                                         name='neglogp'),
        gym_sp=nc.ac_space
    )
    n_v = 1 # no. of value heads
    R = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'R')
    V = tf.placeholder(tf.float32, (nc.batch_size, n_v), 'V')

    return MlpInputs(
        X=X_ph,
        A=A_ph,
        neglogp=neglogp,
        R=R,
        V=V
    )

def MLP(inputs: MlpInputs,
        nc: MlpConfig,
        scope=None) -> MlpOutputs:
    """create the whole net for mlp"""
    with tf.variable_scope(scope, default_name='mlp') as sc:
        # NOTE: use name_scope, in case multiple parameter-sharing nets are built
        net_name_scope = tf.get_default_graph().get_name_scope()
        endpoints_collections = net_name_scope + '_endpoints'

        x = inputs.X

        # make body
        with tf.variable_scope('body'):
            x = tfc_layers.fully_connected(
                x, nc.spa_ch_dim, activation_fn=tf.nn.tanh, scope='fc1')
            x = tfc_layers.fully_connected(
                x, nc.spa_ch_dim, activation_fn=tf.nn.tanh, scope='fc2')

        # make action head
        with tf.variable_scope('act'):
            pdtype = make_pdtype(nc.ac_space)
            pi = tfc_layers.fully_connected(
                x, pdtype.param_shape()[0], activation_fn=tf.nn.tanh, normalizer_fn=None, scope='action')
            head = tp_layers.to_action_head(pi, CategoricalPdType) if nc.is_discrete else \
                   tp_layers.to_action_head(pi, DiagGaussianPdType)

        # make value head
        with tf.variable_scope('vf'):
            vf = tfc_layers.fully_connected(
                x, nc.spa_ch_dim, activation_fn=tf.nn.relu, scope='fc3')
            vf = tfc_layers.fully_connected(
                x, 1, activation_fn=None, normalizer_fn=None, scope='value')

        # make losses
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
                                  'entropy_loss': tf.reduce_mean(entropy_loss),
                                  'return': tf.reduce_mean(inputs.R)}
                loss = MlpLosses(
                    total_reg_loss=total_reg_loss,
                    pg_loss=pg_loss,
                    value_loss=value_loss,
                    entropy_loss=entropy_loss,
                    loss_endpoints=loss_endpoints
                )

        # collect vars, endpoints, etc.
        trainable_vars = _make_vars(sc)
        endpoints = OrderedDict()
    return MlpOutputs(
        self_fed_heads=head,
        outer_fed_heads=head,
        loss=loss,
        vars=trainable_vars,
        endpoints=endpoints,
        value_head=vf
    )


# APIs
net_build_fun = MLP
net_config_cls = MlpConfig
net_inputs_placeholders_fun = mlp_inputs_placeholder
