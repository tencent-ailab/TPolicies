from collections import namedtuple
import logging

from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete


ConvLstmLosses = namedtuple('ConvLstmLosses', [
  'total_reg_loss',
  'pg_loss',  # policy gradient loss
  'value_loss',
  'entropy_loss',
  'loss_endpoints'  # OrderedDict
])

ConvLstmTrainableVariables = namedtuple('ConvLstmTrainableVariables', [
  'all_vars',
  'lstm_vars',
  'vf_vars',
  'pf_vars'
])

ConvLstmInputs = namedtuple('ConvLstmInputs', [
  'X',  # ob_space-like structure of Tensors
  'A',  # ac_space-like structure of Tensors
  'neglogp', # neglopg of (X, A)
  'R',  # rl return at (X, A)
  'V',  # rl value at X
  'S',  # rnn hidden state
  'M'  # rnn hidden state mask
])

ConvLstmOutputs = namedtuple('ConvLstmOutputs', [
  'self_fed_heads',
  'outer_fed_heads',
  'S',
  'loss',
  'vars',
  'endpoints',
  'value_head'
])


class ConvLstmConfig(object):
  def __init__(self, ob_space: GymBox, ac_space: GymDiscrete, **kwargs):
    # logical settings
    self.test = False  # negate is_training
    self.batch_size = None
    # network architecture related
    self.use_lstm = True
    self.use_value_head = False  # for RL training/testing
    self.use_loss_type = 'none'  # {'rl' | 'none'}
    self.use_self_fed_heads = False
    # value head related
    self.n_v = 1
    # common embedding settings
    self.spa_ch_dim = 64
    # lstm settings
    self.nrollout = None
    self.rollout_len = 4
    self.hs_len = 64
    self.nlstm = 32
    self.forget_bias = 0.0  # usually it's 1.0
    self.lstm_dropout_rate = 0.5
    self.lstm_layer_norm = True
    # regularization settings
    self.weight_decay = None  # None means not use. use value, e.g., 0.0005
    # loss related settings
    self.merge_pi = True
    self.adv_normalize = True
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

    # consistency checking
    if self.use_lstm:
      assert self.batch_size is not None, 'lstm requires a specific batch_size.'
      self.nrollout = self.batch_size // self.rollout_len

    assert self.n_v == 1, 'no. value head must be 1.'