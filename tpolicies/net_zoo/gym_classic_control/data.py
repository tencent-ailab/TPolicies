import logging
from typing import Union
from collections import namedtuple

from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete


MlpTrainableVariables = namedtuple('MlpTrainableVariables',[
    'all_vars',
    'vf_vars',
    'pf_vars'
])


MlpInputs = namedtuple('MlpInputs', [
    'X',    # ob_space-like structure of Tensors
    'A',    # ac_space-like structure of Tensors
    'neglogp',  # neglogp of (X, A)
    'R',    # rl return at (X, A)
    'V',    # rl value at X
])


MlpOutputs = namedtuple('MlpOutputs',[
    'self_fed_heads',
    'outer_fed_heads',
    'loss',
    'vars',
    'endpoints',
    'value_head'
])


MlpLosses = namedtuple('MlpLosses', [
    'total_reg_loss',
    'pg_loss',
    'value_loss',
    'entropy_loss',
    'loss_endpoints'
])


class MlpConfig(object):
    def __init__(self, ob_space: GymBox, ac_space: Union[GymDiscrete, GymBox], **kwargs):
        if isinstance(ac_space, GymDiscrete):
            self.is_discrete = True
        elif isinstance(ac_space, GymBox):
            self.is_discrete = False
        else:
            raise ValueError("Not supported action type {}".format(type(ac_space)))

        # logical settings
        self.test = False   # negate is_training
        self.batch_size = None
        # network architecture related
        self.use_value_head = False  # for RL training/testing
        self.use_loss_type = 'none'  # {'rl' | 'none'}
        self.use_self_fed_heads = False
        # value head related
        self.n_v = 1
        # common embedding settings
        self.spa_ch_dim = 12
        # regularization settings
        self.weight_decay = None  # None means not use. use value, e.g., 0.0005
        # loss related settings
        self.sync_statistics = None
        # finally, cache the spaces
        self.ob_space = ob_space
        self.ac_space = ac_space

        # allow partially overwriting
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logging.info('unrecognized config k: {}, v: {}, ignored'.format(k, v))
            self.__dict__[k] = v

