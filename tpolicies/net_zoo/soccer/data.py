from collections import namedtuple
import logging

from gym import spaces


ContNNLosses = namedtuple('ContNNLosses', [
  'total_reg_loss',
  'pg_loss',
  'value_loss',
  'entropy_loss',
  'loss_endpoints'  # OrderedDict
])

ContNNTrainableVariables = namedtuple('ContNNTrainableVariables', [
  'all_vars',
  'vf_vars',
  'pf_vars'
])

ContNNInputs = namedtuple('ContNNInputs', [
  'X',
  'A',
  'neglogp',
  'R',
  'V'
])

ContNNOutputs = namedtuple('ContNNOutputs', [
  'self_fed_heads',
  'outer_fed_heads',
  'loss',
  'vars',
  'endpoints',
  'value_head'
])

class ContNNConfig(object):
  def __init__(self, ob_space, ac_space, **kwargs):
    # logical settings
    self.test = False  # negate is_training
    self.batch_size = None
    # network architecture related
    self.centralV = False
    #self.use_lstm = True
    self.use_value_head = False  # for RL training/testing
    self.use_loss_type = 'none'  # {'rl' | 'none'}
    self.use_self_fed_heads = False
    # value head related
    self.n_v = 1
    # common embedding settings
    self.spa_ch_dim = 64
    self.n_player = 1
    # regularization settings
    self.weight_decay = None  # None means not use. use value, e.g., 0.0005
    # loss related settings
    self.merge_pi = False
    self.adv_normalize = True
    self.reward_weights_shape = None
    self.sync_statistics = None
    # endpoints collection stuff
    self.endpoints_verbosity = 10  # the higher, the more verbose
    self.endpoints_norm = True
    # finally, cache the spaces
    self.ob_space = ob_space
    self.ac_space = ac_space

    # allow partially overwriting
    for k, v in kwargs.items():
      if k not in self.__dict__:
        logging.info('unrecognized config k: {}, v: {}, ignored'.format(k, v))
      self.__dict__[k] = v

    if self.reward_weights_shape is None:
      self.reward_weights_shape = [1, self.n_v]