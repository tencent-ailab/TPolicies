"""DMNet V5 data structures"""
from collections import namedtuple
import logging

from gym.spaces import Dict as GymDict

from timitate.lib6.pb2mask_converter import PB2MaskConverter
from timitate.utils.const import IDEAL_BASE_POS_DICT
from timitate.utils.const import MAP_ORI_SIZE_DICT
from timitate.lib6.z_actions import z_ability2index
from timitate.lib6.z_actions import z_buff2index
from timitate.lib6.z_actions import z_name_map
from timitate.lib6.z_actions import z_action2ability
from timitate.utils.utils import type2index
from timitate.lib6.zstat_utils import BUILD_ORDER_OBJECT_CANDIDATES
from timitate.lib6.zstat_utils import SKILL_OBJECT_CANDIDATES
from timitate.lib6.zstat_utils import EFFECT_ABILITY_CANDIDATES
from timitate.lib6.zstat_utils import RESEARCH_OBJECT_CANDIDATES

MNetV6Embed = namedtuple('MNetV6Embed', [
  'units_embed',
  'spa_embed',
  'vec_embed',
  'int_embed',
  'zstat_embed',
  'lstm_embed',
])

MNetV6VecEmbed = namedtuple('MNetV6VecEmbed', [
  'vec_embed',
  'ab_mask_embed'
])

MNetV6SpaEmbed = namedtuple('MNetV6SpaEmbed', [
  'map_skip',  # a list
  'spa_vec_embed',
])

MNetV6UnitEmbed = namedtuple('MNetV6UnitEmbed', [
  'units_embed',  # [bs, unit_num, dim]
  'embedded_unit',  # [bs, dim]
])

MNetV6EmbedScope = namedtuple('MNetV6EmbedScope', [
  'unit_embed_sc',
  'buff_embed_sc',
  'ab_embed_sc',
  'noop_num_embed_sc',
  'order_embed_sc',
  'u_type_embed_sc',
  'base_embed_sc'
])

MNetV6Consts = namedtuple('MNetV6Consts', [
  'arg_mask',
  'mask_base_poses',
  'base_poses',
  'base'
])

MNetV6d5Consts = namedtuple('MNetV6Consts', [
  'arg_mask',
  'mask_base_poses',
  'base_poses',
  'base',
  'select_type_func_mask',
  'tar_u_type_func_mask'
])

MNetV6Losses = namedtuple('MNetV6Losses', [
  'total_reg_loss',
  'total_il_loss',
  'pg_loss',
  'value_loss',
  'entropy_loss',
  'distill_loss',
  'loss_endpoints'  # OrderedDict
])

MNetV6TrainableVariables = namedtuple('MNetV6TrainableVariables', [
  'all_vars',  # list
  'lstm_vars',  # list
])

MNetV6Inputs = namedtuple('MNetV6Inputs', [
  'X',  # ob_space-like structure of Tensors
  'A',  # ac_space-like structure of Tensors
  'neglogp', # neglopg of (X, A)
  'R',  # rl return at (X, A)
  'V',  # rl value at X
  'S',  # rnn hidden state
  'M',  # rnn hidden state mask
  'flatparam', # logtis from teacher at (X, A)
  'r',
  'discount',
])

MNetV6Outputs = namedtuple('MNetV6Outputs', [
  'self_fed_heads',  # MNetV6ActionHeads
  'outer_fed_heads',  # MNetV6ActionHeads
  'embed',  # MNetV6Embed
  'S',  #  rnn hidden state
  'loss',  # MNetV6Losses
  'vars',  # MNetV6TrainableVariables
  'endpoints',  # OrderedDict
  'value_head',  # value head
])


class MNetV6Config(object):
  def __init__(self, ob_space: GymDict, ac_space: GymDict, **kwargs):
    # logical settings
    self.test = False  # negate is_training
    self.fix_all_embed = False
    self.batch_size = None
    self.use_base_mask = False
    # network architecture related
    self.use_lstm = True
    self.use_value_head = False  # for RL training/testing
    self.use_self_fed_heads = True
    self.use_loss_type = 'none'  # {'il' | 'rl' | 'none'}
    self.embed_for_action_heads = 'int'  # {'int' | 'lstm'}
    self.il_multi_label_loss = False
    self.use_astar_glu = False
    self.use_astar_func_embed = False
    self.use_filter_mask = False
    # sc2 constants
    self.arg_mask = PB2MaskConverter().get_arg_mask()
    # TODO(pengsun): compatible with old version TImitate; remove it later
    # --------------------------
    tmp_pb2mask_cvt = PB2MaskConverter()
    self.select_type_func_mask = (
      None if not hasattr(tmp_pb2mask_cvt, 'get_selection_type_func_mask')
      else tmp_pb2mask_cvt.get_selection_type_func_mask()
    )
    self.tar_u_type_func_mask = (
      None if not hasattr(tmp_pb2mask_cvt, 'get_tar_u_type_func_mask')
      else tmp_pb2mask_cvt.get_tar_u_type_func_mask()
    )
    # --------------------------
    self.z_ability2index = z_ability2index
    self.z_name_map = z_name_map
    self.z_action2ability = z_action2ability
    self.z_buff2index = z_buff2index
    self.ideal_base_pos_dict = IDEAL_BASE_POS_DICT
    self.map_ori_size_dict = MAP_ORI_SIZE_DICT
    # inferred embedding settings
    self.buff_embed_size = 2 + len(self.z_buff2index)
    self.order_embed_size = 2 + len(self.z_ability2index)
    self.u_type_embed_size = 2 + len(type2index().keys())
    # common embedding settings
    self.enc_dim = 256
    self.spa_ch_dim = 64
    self.ff_dim = 256
    # transformer settings
    self.trans_version = 'v1'
    self.trans_n_blk = 3
    self.trans_dropout_rate = 0.0
    # spatial embedding settings
    self.spa_n_blk = 3
    # vector embedding settings
    self.vec_embed_version = 'v2'
    # last action embedding settings
    self.last_act_embed_version = 'v1'
    # zstat settings
    self.zstat_embed_version = 'v1'
    self.zstat_index_base_wavelen = 10000.0
    # lstm settings
    self.nrollout = None
    self.rollout_len = 1
    self.hs_len = 65
    self.nlstm = 32
    self.forget_bias = 1.0  # usually it's 1.0
    self.lstm_duration = 4
    self.lstm_dropout_rate = 0.5
    self.lstm_cell_type = 'lstm'
    self.lstm_layer_norm = True
    # pointer net settings
    self.num_dec_blocks = 1
    # image related settings
    img_sp = ob_space.spaces['X_IMAGE']
    assert len(img_sp.shape) == 3
    self.map_max_row_col = (img_sp.shape[0], img_sp.shape[1])
    # sc2 map settings
    self.max_map_num = ob_space.spaces['MAP_INDICATOR'].shape[0]
    # action heads related
    self.ab_n_blk = 3  # no. of blocks for ability head
    self.ab_n_skip = 2  # no. of skips for ability head
    self.pos_logits_mode = '1x1'  # {[1x1] | 3x3up2}
    self.pos_n_blk = 1  # no. of blocks for pos head
    self.pos_n_skip = 4  # no. of skips for pos head
    self.temperature = 1.0
    self.ab_dim = ac_space.spaces['A_AB'].n
    self.noop_dim = ac_space.spaces['A_NOOP_NUM'].n
    self.tar_unit_dim = ac_space.spaces['A_CMD_UNIT'].n
    self.shift_dim = ac_space.spaces['A_SHIFT'].n
    self.tar_loc_dim = ac_space.spaces['A_CMD_POS'].n
    self.max_bases_num = 18
    self.select_dim = self.tar_unit_dim
    self.gather_batch = False
    # value head related
    self.n_v = 1
    self.lam = None  # lambda for td-lambda. Maybe used by some rl loss
    # loss related settings
    self.merge_pi = True
    self.adv_normalize = True
    self.reward_weights_shape = None
    self.distillation = False
    self.sync_statistics = None
    # The following two can be overwritten by tf.placeholder if needed
    self.clip_range = 0.1
    self.reward_weights = None
    # Notice(lxhan): ac_space does not reveal the width and height; it directly
    # uses the discretized size width*height. We assume the output
    # image should be a square image
    self.output_img_size = (int(pow(self.tar_loc_dim, 0.5)),) * 2
    # regularization settings
    self.weight_decay = None  # None means not use. use value, e.g., 0.0005
    # endpoints collection stuff
    self.endpoints_verbosity = 10  # the higher, the more verbose
    self.endpoints_norm = True
    # arg_scope (initialization, regularizer, etc.)
    self.arg_scope_type = 'mnet_v5_type_a'
    # finally, cache the spaces
    self.ob_space = ob_space
    self.ac_space = ac_space
    self.zstat_embed_version = 'v3'
    self.value_net_version = 'v2'
    self.n_v = 6

    # allow partially overwriting
    for k, v in kwargs.items():
      if k not in self.__dict__:
        logging.info('unrecognized config k: {}, v: {}, ignored'.format(k, v))
      self.__dict__[k] = v

    if self.reward_weights_shape is None:
      self.reward_weights_shape = [1, self.n_v]

    # consistency check
    if self.use_lstm:
      assert self.batch_size is not None, 'lstm requires a specific batch_size.'
      self.nrollout = self.batch_size // self.rollout_len
      assert (self.rollout_len % self.lstm_duration == 0 or
              self.rollout_len == 1)
      if self.lstm_cell_type == 'k_lstm':
        assert self.hs_len == 2 * self.nlstm + 1
      elif self.lstm_cell_type == 'lstm':
        assert self.hs_len == 2 * self.nlstm
      else:
        raise NotImplemented('Unknown lstm_cell_type {}'.format(
          self.lstm_cell_type))
    if self.use_filter_mask and self.use_self_fed_heads:
      logging.warning('filter_mask requires outer_fed_heads or '
                      'if the policy only outputs lstm hidden state.')

    if not self.test:
      assert self.use_self_fed_heads is False, (
      'when training, must use outer fed heads (use_self_fed_heads = False)'
      )

    self.effects_mask = None
    self.upgrades_mask = None
    self.uc_split_indices = [len(BUILD_ORDER_OBJECT_CANDIDATES),
                             len(SKILL_OBJECT_CANDIDATES) +
                             len(EFFECT_ABILITY_CANDIDATES),
                             len(RESEARCH_OBJECT_CANDIDATES)]


MNetV5Inputs = namedtuple('MNetV5Inputs', [
  'X',  # ob_space-like structure of Tensors
  'A',  # ac_space-like structure of Tensors
  'neglogp', # neglopg of (X, A)
  'R',  # rl return at (X, A)
  'V',  # rl value at X
  'S',  # rnn hidden state
  'M',  # rnn hidden state mask
  'logits', # logtis from teacher at (X, A)
  'r',
  'discount',
])
MNetV5TrainableVariables = namedtuple('MNetV5TrainableVariables', [
  'all_vars',  # list
  'lstm_vars',  # list
])
MNetV5EmbedScope = namedtuple('MNetV5EmbedScope', [
  'unit_embed_sc',
  'buff_embed_sc',
  'ab_embed_sc',
  'noop_num_embed_sc',
  'order_embed_sc',
  'u_type_embed_sc',
  'base_embed_sc'
])
MNetV5Consts = namedtuple('MNetV5Consts', [
  'arg_mask',
  'mask_base_poses',
  'base_poses',
  'base'
])