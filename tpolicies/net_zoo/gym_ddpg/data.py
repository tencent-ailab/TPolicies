from collections import namedtuple
import logging

from gym import spaces


DDPGLosses = namedtuple('DDPGLosses', [
  'total_reg_loss',
  'pg_loss',
  'value_loss',
  'entropy_loss',
  'loss_endpoints'  # OrderedDict
])

DDPGTrainableVariables = namedtuple('DDPGTrainableVariables', [
  'all_vars',
  'lstm_vars'
])

DDPGInputs = namedtuple('DDPGInputs', [
  'X',  # ob_space-like structure of Tensors
  'A',  # ac_space-like structure of Tensors
  'S',  # rnn hidden state
  'M',  # rnn hidden state mask
  'r',
  'discount',
])

DDPGOutputs = namedtuple('DDPGOutputs', [
  'self_fed_heads',
  'outer_fed_heads',
  'S',
  'loss',
  'vars',
  'endpoints',
  'self_fed_value_head',
  'outer_fed_value_head',
])


class DDPGConfig(object):
  def __init__(self, ob_space, ac_space, **kwargs):
    # logical settings
    self.test = False  # negate is_training
    self.batch_size = None
    # network architecture related
    self.centralV = False
    self.use_lstm = True
    self.use_value_head = False  # for RL training/testing
    self.use_loss_type = 'none'  # {'rl' | 'none'}
    self.use_self_fed_heads = True
    self.use_target_net = False
    # value head related
    self.n_v = 1
    self.lam = None
    # common embedding settings
    self.fc_ch_dim = 32
    # lstm settings
    self.n_player = 2
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
    self.merge_pi = False
    self.adv_normalize = True
    self.reward_weights_shape = None
    self.reward_weights = None
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

    # consistency checking
    assert self.batch_size is not None, 'lstm requires a specific batch_size.'
    self.nrollout = self.batch_size // self.rollout_len