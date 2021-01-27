"""MNet V6"""
from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import nest
import numpy as np
from timitate.utils.utils import CoorSys

import tpolicies.ops as tp_ops
import tpolicies.losses as tp_losses
import tpolicies.layers as tp_layers
import tpolicies.tp_utils as tp_utils
from tpolicies.utils.sequence_ops import multistep_forward_view
from tpolicies.ops import fetch_op
from tpolicies.utils.distributions import CategoricalPdType
from tpolicies.net_zoo.mnet_v6.utils import _make_mnet_v6_arg_scope_a, \
  _const_select_type_mask, _const_tar_u_type_mask, _const_arg_mask, \
  _const_mask_base_poses, _const_base_poses, _units_embed_block, \
  _scatter_units_block, _spa_embed_block_v2, _vec_embed_block_v2, \
  _vec_embed_block_v2d1, _vec_embed_block_v3, _vec_embed_block_v3d1, \
  _last_action_embed_block_mnet_v6, _last_action_embed_block_mnet_v6_v2, \
  _zstat_embed, _pre_discrete_action_res_block, \
  _pre_discrete_action_fc_block, _pre_ptr_action_res_block, \
  _pre_loc_action_astar_like_block_v1, _astar_v_oppo_vec_embed_block, \
  _light_lstm_value_block_v2, _light_lstm_value_block_v4, \
  _light_trans_value_block_v1, _transformer_block, _transformer_block_v2, \
  _transformer_block_v3, _transformer_block_v4, _transformer_block_v5, \
  _action_mask_weights, _make_shared_embed_scopes, _make_inputs_palceholders, \
  _make_mnet_v5_arg_scope_a
from tpolicies.net_zoo.mnet_v6.utils import _make_mnet_v6_vars
from tpolicies.net_zoo.mnet_v6.utils import _make_mnet_v6_endpoints_dict
from tpolicies.net_zoo.mnet_v6.data import MNetV6Embed, MNetV6Config, \
  MNetV6d5Consts
from tpolicies.net_zoo.mnet_v6.data import MNetV6VecEmbed
from tpolicies.net_zoo.mnet_v6.data import MNetV6SpaEmbed
from tpolicies.net_zoo.mnet_v6.data import MNetV6UnitEmbed
from tpolicies.net_zoo.mnet_v6.data import MNetV6Consts
from tpolicies.net_zoo.mnet_v6.data import MNetV6EmbedScope
from tpolicies.net_zoo.mnet_v6.data import MNetV6Inputs
from tpolicies.net_zoo.mnet_v6.data import MNetV6Outputs
from tpolicies.net_zoo.mnet_v6.data import MNetV6Losses


def mnet_v6d6_inputs_placeholders(nc: MNetV6Config):
  return _make_inputs_palceholders(nc)


def mnet_v6d6(inputs: MNetV6Inputs,
              nc: MNetV6Config,
              scope=None) -> MNetV6Outputs:
  """create the whole net for mnet_v6d6

  based on mnet_v6d3 and mnet_v6d5, add arguments for
    1. using AStar-like glu
    2. using AStar-like func embed
  other differences from v6d3 are same as v6d5 """
  with tf.variable_scope(scope, default_name='mnet_v6d6') as sc:
    # NOTE: use name_scope, in case multiple parameter-sharing nets are built
    net_name_scope = tf.get_default_graph().get_name_scope()
    endpoints_collections = net_name_scope + '_endpoints'
    arg_scope_funcs = {
      'mnet_v5_type_a': _make_mnet_v5_arg_scope_a,  # backwards compatibility
      'mnet_v6_type_a': _make_mnet_v6_arg_scope_a,
    }
    if nc.arg_scope_type not in arg_scope_funcs:
      raise ValueError('Unknown arg_scope type {}'.format(nc.arg_scope_type))
    make_arg_scope_func = arg_scope_funcs[nc.arg_scope_type]
    with arg_scope(make_arg_scope_func(nc, endpoints_collections)):
      # make embeddings
      embed_sc = _make_shared_embed_scopes()
      consts = mnet_v6d5_consts(nc, scope='consts')
      coord_sys = CoorSys(r_max=nc.map_max_row_col[0],
                          c_max=nc.map_max_row_col[1])
      embed, hs_new = mnet_v6d6_embed(inputs, embed_sc, consts, nc, coord_sys,
                                      scope='embed')

      # make heads
      with tf.variable_scope('heads', reuse=tf.AUTO_REUSE) as heads_sc:
        pass
      self_fed_heads, outer_fed_heads = None, None
      if nc.use_self_fed_heads:
        self_fed_heads, structured_mw = mnet_v6d6_heads(
          inputs=inputs,
          inputs_embed=embed,
          embed_sc=embed_sc,
          consts=consts,
          coord_sys=coord_sys,
          nc=nc,
          scope=heads_sc)
      else:
        flag = (inputs.A['A_AB'] is not None and
                inputs.A['A_NOOP_NUM'] is not None and
                inputs.A['A_SHIFT'] is not None and
                inputs.A['A_SELECT'] is not None)
        assert flag, ('creating outer_fed_heads, '
                      'but outer fed heads are None ...')
        outer_fed_heads, structured_mw = mnet_v6d6_heads(
          inputs=inputs,
          inputs_embed=embed,
          embed_sc=embed_sc,
          consts=consts,
          coord_sys=coord_sys,
          nc=nc,
          scope=heads_sc)
      value_head = None
      if nc.use_value_head:
        value_head = mnet_v6d6_value(inputs_embed=embed,
                                     inputs_obs=inputs.X,
                                     nc=nc,
                                     scope='vf')

      # make losses
      with tf.device("/cpu:0"):
        loss = mnet_v6d6_loss(inputs, outer_fed_heads, value_head, consts, nc,
                              structured_mw=structured_mw,
                              net_level_scope=sc.name, scope='losses')

    # done net building, other things for network manipulation
    trainable_vars = _make_mnet_v6_vars(sc)
    endpoints = _make_mnet_v6_endpoints_dict(nc, endpoints_collections,
                                             name_scope=net_name_scope)
  return MNetV6Outputs(self_fed_heads, outer_fed_heads, embed, hs_new,
                       loss, trainable_vars, endpoints, value_head)


def mnet_v6d6_heads(inputs: MNetV6Inputs,
                  inputs_embed: MNetV6Embed,
                  embed_sc: MNetV6EmbedScope,
                  consts: MNetV6Consts,
                  coord_sys,
                  nc: MNetV6Config,
                  scope=None):
  # shorter names
  inputs_obs, inputs_act = inputs.X, getattr(inputs, 'A', None)
  embed = inputs_embed

  with tf.variable_scope(scope, default_name='mnet_v6d6_heads'):
    # use or create scalar_context
    if embed.vec_embed.ab_mask_embed is None:
      scalar_context = tfc_layers.fully_connected(
        tp_ops.to_float32(inputs_obs['MASK_AB']),
        64
      )
    else:
      scalar_context = embed.vec_embed.ab_mask_embed
    # update scalar_context
    scalar_context = tf.concat([scalar_context, embed.zstat_embed], axis=-1)

    # make ability action head: level 1
    with tf.variable_scope('ability'):
      # create embeddings for the action heads
      if nc.embed_for_action_heads == 'int':
        emb_for_heads = embed.int_embed
      elif nc.embed_for_action_heads == 'lstm':
        emb_for_heads = embed.lstm_embed
      else:
        raise NotImplementedError(
          'Unknown nc.embed_for_action_heads {}'.format(
            nc.embed_for_action_heads)
        )

      # NOTE: comparable to v5, use layer_norm
      o = _pre_discrete_action_res_block(emb_for_heads,
                                         nc.enc_dim,
                                         n_blk=nc.ab_n_blk,
                                         n_skip=nc.ab_n_skip)
      if nc.use_astar_glu:
        ab_head = tp_layers.discrete_action_head_v2(
          inputs=o,
          n_actions=nc.ab_dim,
          pdtype_cls=CategoricalPdType,
          context=scalar_context,
          mask=inputs_obs['MASK_AB'],  # fine to pass again for hard masking
          temperature=nc.temperature,
          scope='action_head'
        )
      else:
        ab_head = tp_layers.discrete_action_head(
          inputs=o,
          n_actions=nc.ab_dim,
          enc_dim=nc.enc_dim,
          pdtype_cls=CategoricalPdType,
          mask=inputs_obs['MASK_AB'],
          embed_scope=None,
          temperature=nc.temperature,
          scope='action_head'
        )

    # make noop action head: auto-reg level 2
    ab_taken = (inputs_act['A_AB'] if inputs_act is not None else ab_head.sam)
    mw = _action_mask_weights(inputs_ab=ab_taken,
                              inputs_arg_mask=consts.arg_mask,
                              weights_include_ab=True)
    structured_mw = tp_utils.pack_sequence_as_structure_like_gym_space(
      nc.ac_space, mw)
    ab_taken_embed = tp_layers.linear_embed(ab_taken,
                                            vocab_size=nc.ab_dim,
                                            enc_size=nc.enc_dim,
                                            scope=embed_sc.ab_embed_sc)

    if nc.use_astar_glu:
      # create regressive embeddings gated on scalar_context
      reg_embed = tp_layers.glu(emb_for_heads, scalar_context, 1024)
      reg_embed += tp_layers.glu(ab_taken_embed, scalar_context, 1024)
    else:
      reg_embed = tfc_layers.fully_connected(emb_for_heads, 1024)
      reg_embed += tfc_layers.fully_connected(ab_taken_embed, 1024)

    # smoothing discrete head for noop
    with tf.variable_scope('noop_num'):
      # NOTE: comparable to v5, use bottleneck
      noop_logits = _pre_discrete_action_fc_block(inputs=reg_embed,
                                                  n_actions=nc.noop_dim,
                                                  enc_dim=nc.enc_dim,
                                                  n_blk=2)
      noop_head = tp_layers.to_action_head(noop_logits,
                                           CategoricalPdType)

    # make shift action head: auto-reg level 3
    noop_taken = (inputs_act['A_NOOP_NUM'] if inputs_act is not None
                  else noop_head.sam)
    noop_taken_embed = tp_layers.linear_embed(noop_taken,
                                              vocab_size=nc.noop_dim,
                                              enc_size=nc.enc_dim,
                                              scope=embed_sc.noop_num_embed_sc)
    # reg_embed = tf.concat([reg_embed, noop_taken_embed], axis=-1)
    reg_embed += tfc_layers.fully_connected(noop_taken_embed, 1024)
    with tf.variable_scope('shift'):
      o = _pre_discrete_action_res_block(reg_embed, nc.enc_dim, n_blk=1,
                                         n_skip=2)
      sft_head = tp_layers.discrete_action_head(
        inputs=o,
        n_actions=nc.shift_dim,
        enc_dim=nc.enc_dim,
        pdtype_cls=CategoricalPdType,
        embed_scope=None,
        temperature=nc.temperature,
        scope='shift_head'
      )

    # make selection action head: auto-reg level 4
    sft_taken = (inputs_act['A_SHIFT'] if inputs_act is not None
                 else sft_head.sam)
    # sft_taken_embed = tp_ops.to_float32(tf.expand_dims(sft_taken, axis=-1))
    # reg_embed = tf.concat([reg_embed, sft_taken_embed], axis=-1)
    sft_taken_embed = tp_layers.linear_embed(sft_taken,
                                             vocab_size=2,
                                             enc_size=1024,
                                             scope="sft_embed")
    reg_embed += sft_taken_embed

    # create func embed
    if nc.use_astar_func_embed:
      with tf.variable_scope('func_embed',
                             reuse=tf.AUTO_REUSE) as func_embed_sc:
        pass
      # selection func_embed per AStar
      select_func_embed = tf.nn.embedding_lookup(consts.select_type_func_mask,
                                                 ab_taken)
      select_func_embed = tfc_layers.fully_connected(
        tf.cast(select_func_embed, tf.float32),
        nc.enc_dim,
        activation_fn=tf.nn.relu,
        scope=func_embed_sc
      )
      # target unit func_embed per AStar
      tar_u_func_embed = tf.nn.embedding_lookup(
        consts.tar_u_type_func_mask,
        ab_taken
      )
      tar_u_func_embed = tfc_layers.fully_connected(
        tf.cast(tar_u_func_embed, tf.float32),
        nc.enc_dim,
        activation_fn=tf.nn.relu,
        scope=func_embed_sc
      )

    with tf.variable_scope('select'):
      if nc.use_filter_mask:
        s_mask = inputs_obs['MASK_SELECTION']
      else:
        s_mask = fetch_op(inputs_obs['MASK_SELECTION'], ab_taken)
      s_keys = tfc_layers.fully_connected(embed.units_embed.units_embed,
                                          32, activation_fn=None,
                                          scope='selection_raw_keys')
      # make ground-truth selection labels (if any)
      selection_labels = (inputs_act['A_SELECT'] if inputs_act is not None
                          else None)
      # get the head and the updated s_embed
      if nc.use_astar_func_embed:
        s_head, reg_embed = tp_layers.sequential_selection_head_v2(
          inputs=reg_embed,
          inputs_select_mask=s_mask,
          input_keys=s_keys,
          input_selections=selection_labels,
          input_func_embed=select_func_embed,
          max_num=64,
          temperature=nc.temperature,
          scope='selection_head'
        )
      else:
        s_head, reg_embed = tp_layers.sequential_selection_head(
          inputs=reg_embed,
          inputs_select_mask=s_mask,
          input_keys=s_keys,
          input_selections=selection_labels,
          max_num=64,
          temperature=nc.temperature,
          scope='selection_head'
        )
      # reg_embed = tf.concat([reg_embed, s_embed], axis=-1)

    # make cmd_u action head: auto-reg level 5
    gathered_reg_embed = reg_embed
    gathered_units_embed = embed.units_embed.units_embed
    gathered_map_skip = embed.spa_embed.map_skip
    with tf.variable_scope("cmd_u"):
      # NOTE: comparable with v5
      ind = None
      if nc.gather_batch:
        mask = structured_mw['A_CMD_UNIT']
        ind = tf.cast(tf.where(mask), tf.int32)[:, 0]
        gathered_reg_embed = tf.gather(reg_embed, ind)
        if nc.use_filter_mask:
          inputs_ptr_mask = tf.gather(inputs_obs['MASK_CMD_UNIT'], ind)
        else:
          inputs_ptr_mask = tf.gather_nd(
            inputs_obs['MASK_CMD_UNIT'],
            tf.stack([ind, tf.gather(ab_taken, ind)], axis=1)
          )
        gathered_units_embed = tf.gather(embed.units_embed.units_embed, ind)
        if nc.use_astar_func_embed:
          tar_u_func_embed = tf.gather(tar_u_func_embed, ind)
      else:
        if nc.use_filter_mask:
          inputs_ptr_mask = inputs_obs['MASK_CMD_UNIT']
        else:
          inputs_ptr_mask = fetch_op(inputs_obs['MASK_CMD_UNIT'], ab_taken)
      cmd_u_inputs = _pre_ptr_action_res_block(gathered_reg_embed, nc.enc_dim,
                                               n_blk=1, n_skip=2)
      if nc.use_astar_func_embed:
        cmd_u_head = tp_layers.ptr_action_head_v2(
          inputs_query=cmd_u_inputs,
          inputs_ptr_mask=inputs_ptr_mask,
          inputs_entity_embed=gathered_units_embed,
          inputs_func_embed=tar_u_func_embed,
          ptr_out_dim=nc.tar_unit_dim,
          pdtype_cls=CategoricalPdType,
          temperature=nc.temperature,
          scatter_ind=ind,
          scatter_bs=nc.batch_size,
          scope='cmd_u_head'
        )
      else:
        cmd_u_head = tp_layers.ptr_action_head(
          inputs_query=cmd_u_inputs,
          inputs_ptr_mask=inputs_ptr_mask,
          inputs_entity_embed=gathered_units_embed,
          ptr_out_dim=nc.tar_unit_dim,
          num_dec_blocks=1,
          ff_dim=nc.enc_dim,
          enc_dim=nc.enc_dim,
          pdtype_cls=CategoricalPdType,
          temperature=nc.temperature,
          scatter_ind=ind,
          scatter_bs=nc.batch_size,
          scope='cmd_u_head'
        )

    # cmd_pos: auto-reg level 5
    ch_dim = nc.spa_ch_dim
    with tf.variable_scope("pos"):
      # common pos embedding
      ind = None
      if nc.gather_batch:
        mask = structured_mw['A_CMD_POS']
        ind = tf.cast(tf.where(mask), tf.int32)[:, 0]
        gathered_reg_embed = tf.gather(reg_embed, ind)
        gathered_map_skip = [tf.gather(map_skip, ind)
                             for map_skip in embed.spa_embed.map_skip]
        if nc.use_filter_mask:
          loc_masks = tf.gather(inputs_obs['MASK_CMD_POS'], ind)
        else:
          loc_masks = tf.gather_nd(inputs_obs['MASK_CMD_POS'],
                                   tf.stack([ind, tf.gather(ab_taken, ind)], axis=1))
      else:
        if nc.use_filter_mask:
          loc_masks = inputs_obs['MASK_CMD_POS']
        else:
          loc_masks = fetch_op(inputs_obs['MASK_CMD_POS'], ab_taken)
      # pos embedding with shared variables
      with tf.variable_scope('cmd_pos'):
        # TODO: Astar-like pos head
        pos_inputs = _pre_loc_action_astar_like_block_v1(
          gathered_reg_embed,
          gathered_map_skip[-1],
          n_blk=nc.pos_n_blk,
          n_skip=nc.pos_n_skip
        )
        pos_head = tp_layers.loc_action_head(
          inputs=pos_inputs,
          mask=loc_masks,
          pdtype_cls=CategoricalPdType,
          temperature=nc.temperature,
          logits_mode=nc.pos_logits_mode,
          scatter_ind=ind,
          scatter_bs=nc.batch_size,
          scope='pos_head'
        )
  return tp_utils.pack_sequence_as_structure_like_gym_space(nc.ac_space, [
    ab_head, noop_head, sft_head, s_head, cmd_u_head, pos_head,
  ]), structured_mw


def mnet_v6d6_value(inputs_embed: MNetV6Embed,
                    inputs_obs,
                    nc: MNetV6Config,
                    scope=None):
  int_embed = inputs_embed.int_embed
  with tf.variable_scope(scope, default_name='mnet_v6d6_vf'):
    oppo_int_embed = _astar_v_oppo_vec_embed_block(inputs_obs, nc.enc_dim,
                                                   nc.use_score_in_value)
    if nc.value_net_version == 'v2':
      return _light_lstm_value_block_v2(
        int_embed=int_embed,
        z_bo=inputs_obs['Z_BUILD_ORDER'],
        z_boc=inputs_obs['Z_BUILD_ORDER_COORD'],
        z_bobt=inputs_obs['Z_BUILD_ORDER_BT'],
        z_bocbt=inputs_obs['Z_BUILD_ORDER_COORD_BT'],
        z_bu=inputs_obs['Z_UNIT_COUNT'],
        c_bo=inputs_obs['IMM_BUILD_ORDER'],
        c_boc=inputs_obs['IMM_BUILD_ORDER_COORD'],
        c_bobt=inputs_obs['IMM_BUILD_ORDER_BT'],
        c_bocbt=inputs_obs['IMM_BUILD_ORDER_COORD_BT'],
        c_bu=inputs_obs['IMM_UNIT_COUNT'],
        upgrades=inputs_obs['X_VEC_UPGRADE'],
        oppo_int_embed=oppo_int_embed,
        oppo_c_bo=inputs_obs['OPPO_IMM_BUILD_ORDER'],
        oppo_c_boc=inputs_obs['OPPO_IMM_BUILD_ORDER_COORD'],
        oppo_c_bobt=inputs_obs['OPPO_IMM_BUILD_ORDER_BT'],
        oppo_c_bocbt=inputs_obs['OPPO_IMM_BUILD_ORDER_COORD_BT'],
        oppo_c_bu=inputs_obs['OPPO_IMM_UNIT_COUNT'],
        nc=nc
      )
    elif nc.value_net_version == 'v4':
      return _light_lstm_value_block_v4(
        int_embed=int_embed,
        score=inputs_obs['X_VEC_SCORE'],
        z_bo=inputs_obs['Z_BUILD_ORDER'],
        z_boc=inputs_obs['Z_BUILD_ORDER_COORD'],
        z_bu=inputs_obs['Z_UNIT_COUNT'],
        c_bo=inputs_obs['IMM_BUILD_ORDER'],
        c_boc=inputs_obs['IMM_BUILD_ORDER_COORD'],
        c_bu=inputs_obs['IMM_UNIT_COUNT'],
        upgrades=inputs_obs['X_VEC_UPGRADE'],
        oppo_int_embed=oppo_int_embed,
        oppo_c_bo=inputs_obs['OPPO_IMM_BUILD_ORDER'],
        oppo_c_boc=inputs_obs['OPPO_IMM_BUILD_ORDER_COORD'],
        oppo_c_bu=inputs_obs['OPPO_IMM_UNIT_COUNT'],
        nc=nc
      )
    elif nc.value_net_version == 'trans_v1':
      return _light_trans_value_block_v1(
        int_embed=int_embed,
        z_bo=inputs_obs['Z_BUILD_ORDER'],
        z_boc=inputs_obs['Z_BUILD_ORDER_COORD'],
        z_bu=inputs_obs['Z_UNIT_COUNT'],
        c_bo=inputs_obs['IMM_BUILD_ORDER'],
        c_boc=inputs_obs['IMM_BUILD_ORDER_COORD'],
        c_bu=inputs_obs['IMM_UNIT_COUNT'],
        upgrades=inputs_obs['X_VEC_UPGRADE'],
        oppo_int_embed=oppo_int_embed,
        oppo_c_bo=inputs_obs['OPPO_IMM_BUILD_ORDER'],
        oppo_c_boc=inputs_obs['OPPO_IMM_BUILD_ORDER_COORD'],
        oppo_c_bu=inputs_obs['OPPO_IMM_UNIT_COUNT'],
        nc=nc
      )
    else:
      raise ModuleNotFoundError('Unkown value net version for mnet_v6d6')


def mnet_v6d6_embed(inputs: MNetV6Inputs,
                    embed_sc: MNetV6EmbedScope,
                    consts: MNetV6Consts,
                    nc: MNetV6Config,
                    coord_sys,
                    scope=None):
  inputs_obs = inputs.X
  with tf.variable_scope(scope, default_name='mnet_v6_embed'):
    # unit list embeddings
    # low-level unit embeddings
    u_embed, u_coor, u_is_selected_mask, u_was_tar_mask = _units_embed_block(
      inputs=inputs_obs,
      embed_sc=embed_sc,
      nc=nc
    )
    # higher-level unit embeddings
    if nc.trans_version == 'v1':
      enhanced_units_embed, embedded_unit = _transformer_block(
        units_embed=u_embed, units_mask=inputs_obs['MASK_LEN'], nc=nc)
    elif nc.trans_version == 'v2':
      enhanced_units_embed, embedded_unit = _transformer_block_v2(
        units_embed=u_embed, units_mask=inputs_obs['MASK_LEN'], nc=nc)
    elif nc.trans_version == 'v3':
      enhanced_units_embed, embedded_unit = _transformer_block_v3(
        units_embed=u_embed, units_mask=inputs_obs['MASK_LEN'], nc=nc)
    elif nc.trans_version == 'v4':
      enhanced_units_embed, embedded_unit = _transformer_block_v4(
        units_embed=u_embed, units_mask=inputs_obs['MASK_LEN'],
        enc_dim=nc.enc_dim, out_fc_dim=nc.enc_dim, nc=nc)
    elif nc.trans_version == 'v5':
      enhanced_units_embed, embedded_unit = _transformer_block_v5(
        units_embed=u_embed, units_mask=inputs_obs['MASK_LEN'],
        enc_dim=nc.enc_dim, out_fc_dim=nc.enc_dim, nc=nc)
    else:
      raise NotImplementedError
    # scatter units to img
    # (bs, 600, dim)
    lowdim_units_embed = tfc_layers.conv1d(enhanced_units_embed, 32, 1)
    # (bs, 600, 32)
    scattered_embed = _scatter_units_block(
      inputs_units_embed=lowdim_units_embed,
      inputs_xy=u_coor,
      coord_sys=coord_sys,
      nc=nc
    )
    # (bs, 128, 128, 32)

    # joint unit-map spatial embeddings
    map_skip, spa_vec_embed = _spa_embed_block_v2(
      inputs_img=inputs_obs['X_IMAGE'],
      inputs_additonal_img=scattered_embed,
      nc=nc)

    # global feature embeddings
    ab_mask_embed = None  # aka "available_actions"
    # vector embeddings
    if nc.vec_embed_version == 'v2':
      vec_embed = _vec_embed_block_v2(inputs=inputs_obs, enc_dim=nc.enc_dim)
    elif nc.vec_embed_version == 'v2d1' or nc.vec_embed_version == 'v2.1':
      vec_embed, ab_mask_embed = _vec_embed_block_v2d1(inputs=inputs_obs,
                                                       enc_dim=nc.enc_dim)
    elif nc.vec_embed_version == 'v3':
      vec_embed = _vec_embed_block_v3(inputs=inputs_obs, enc_dim=nc.enc_dim)
    elif nc.vec_embed_version == 'v3d1' or nc.vec_embed_version == 'v3.1':
      vec_embed, ab_mask_embed = _vec_embed_block_v3d1(inputs=inputs_obs,
                                                       enc_dim=nc.enc_dim)
    else:
      raise NotImplementedError('unknown vec_embed_version: {}'.format(
        nc.vec_embed_version
      ))
    # last actions embeddings
    if nc.last_act_embed_version == 'v1':
      last_actions_embed = _last_action_embed_block_mnet_v6(
        inputs=inputs_obs,
        inputs_arg_mask=consts.arg_mask,
        ab_embed_sc=embed_sc.ab_embed_sc,
        nc=nc
      )
    elif nc.last_act_embed_version == 'v2':
      last_actions_embed = _last_action_embed_block_mnet_v6_v2(
        inputs=inputs_obs,
        inputs_arg_mask=consts.arg_mask,
        ab_embed_sc=embed_sc.ab_embed_sc,
        nc=nc
      )
    else:
      raise NotImplementedError('unknown last_act_embed_version: {}'.format(
        nc.last_act_embed_version
      ))
    # zstat embeddings
    zstat_embed = _zstat_embed(inputs_obs, nc)

    # integrate the features
    int_embed = tf.concat([embedded_unit, spa_vec_embed, vec_embed,
                           last_actions_embed, zstat_embed], axis=-1)

    # lstm embeddings
    hs_new = None
    lstm_embed = None
    if nc.use_lstm:
      lstm_embed, hs_new = tp_layers.lstm_embed_block(
        inputs_x=int_embed,
        inputs_hs=inputs.S,
        inputs_mask=inputs.M,
        nc=nc)
      int_embed = tf.concat([int_embed, lstm_embed], axis=-1)
    # used for burn-in
    if nc.fix_all_embed:
      int_embed = tf.stop_gradient(int_embed)
      enhanced_units_embed = tf.stop_gradient(enhanced_units_embed)
      embedded_unit = tf.stop_gradient(embedded_unit)
      spa_vec_embed = tf.stop_gradient(spa_vec_embed)
      vec_embed = tf.stop_gradient(vec_embed)
      if ab_mask_embed is not None:
        ab_mask_embed = tf.stop_gradient(ab_mask_embed)
      zstat_embed = tf.stop_gradient(zstat_embed)
      map_skip = [tf.stop_gradient(m) for m in map_skip]
      if nc.use_lstm:
        lstm_embed = tf.stop_gradient(lstm_embed)

    return MNetV6Embed(
      units_embed=MNetV6UnitEmbed(units_embed=enhanced_units_embed,
                                  embedded_unit=embedded_unit),
      spa_embed=MNetV6SpaEmbed(map_skip=map_skip,
                               spa_vec_embed=spa_vec_embed),
      vec_embed=MNetV6VecEmbed(vec_embed=vec_embed,
                               ab_mask_embed=ab_mask_embed),
      int_embed=int_embed,
      zstat_embed=zstat_embed,
      lstm_embed=lstm_embed,
    ), hs_new


def mnet_v6d6_loss(inputs: MNetV6Inputs,
                   outer_fed_heads,
                   value_head,
                   consts: MNetV6Consts,
                   nc: MNetV6Config,
                   net_level_scope: str,
                   structured_mw=None,
                   scope=None):
  # regularization loss. Only `variable`s are involved, so it is safe to
  # collect them using regular expression, e.g., 'mnet_v5.*', regardless
  # of the current name_scope (e.g., 'mnet_v5_1', 'mnet_v5_2', ...)
  total_reg_loss = tf.losses.get_regularization_loss(
    scope='{}.*'.format(net_level_scope))

  total_il_loss = None
  pg_loss = None
  value_loss = None
  entropy_loss = None
  distill_loss = None
  loss_endpoints = {}
  example_ac_sp = tp_utils.map_gym_space_to_structure(lambda x: None,
                                                      nc.ac_space)
  with tf.variable_scope(scope, default_name='mnet_v6_losses'):
    if nc.use_loss_type in ['il', 'rl', 'rl_ppo', 'rl_ppo2', 'rl_vtrace']:
      # head masks and structure template
      if structured_mw is None:
        mw = _action_mask_weights(inputs_ab=inputs.A['A_AB'],
                                  inputs_arg_mask=consts.arg_mask,
                                  weights_include_ab=True)
        structured_mw = tp_utils.pack_sequence_as_structure_like_gym_space(
          nc.ac_space, mw)
      outer_fed_head_pds = nest.map_structure_up_to(
        example_ac_sp, lambda head: head.pd, outer_fed_heads)

      if nc.use_loss_type == 'il':
        # build imitation learning loss the cross entropy
        total_il_loss, head_xe_loss = tp_losses.multi_head_neglogp_loss(
          inputs_action_pds=outer_fed_head_pds,
          inputs_action_labels=inputs.A,
          inputs_mask_weights=structured_mw,
          set_loss=nc.il_multi_label_loss,
        )
        assert type(head_xe_loss) == OrderedDict
        loss_endpoints = head_xe_loss
      elif nc.use_loss_type in ['rl', 'rl_ppo', 'rl_ppo2', 'rl_vtrace']:
        # build rl losses

        # the entropy regularizer
        entropy_loss = nest.map_structure_up_to(
          example_ac_sp, lambda head, mask: tf.reduce_mean(head.ent * mask),
          outer_fed_heads, structured_mw)

        # distillation loss, i.e., the teacher-student KL regularizer
        distill_loss = None
        ab_distill_loss = None
        if nc.distillation:
          outer_fed_head_pds = nest.map_structure_up_to(
            example_ac_sp, lambda head: head.pd, outer_fed_heads)
          distill_loss = tp_losses.distill_loss(
            student_pds=outer_fed_head_pds,
            teacher_logits=inputs.flatparam,
            masks=structured_mw)
          ab_pd = outer_fed_head_pds['A_AB']
          teacher_logit = inputs.flatparam['A_AB']
          # TODO: this is from definition of position encoding, remove it?
          first_4mins_mask = tf.cast(
            inputs.X['X_VEC_GAME_PROG'][:, -1] >= np.cos(
              60 * 4 * np.power(10000, -62 / 64)),
            tf.float32)
          first_4mins_mask *= tf.cast(
            (tf.reduce_sum(inputs.X['Z_BUILD_ORDER'], axis=[1, 2]) > 0),
            tf.float32)
          ab_distill_loss = tp_losses.distill_loss(ab_pd, teacher_logit,
                                                   first_4mins_mask)

        # the main policy gradient loss
        outer_fed_head_neglogp = nest.map_structure_up_to(
          example_ac_sp,
          lambda head, ac: head.pd.neglogp(ac),
          outer_fed_heads,
          inputs.A
        )
        loss_endpoints = {}
        if nc.use_loss_type == 'rl' or nc.use_loss_type == 'rl_ppo':
          # PPO loss
          pg_loss, value_loss = tp_losses.ppo_loss(
            outer_fed_head_neglogp,
            inputs.neglogp,
            value_head,
            inputs.R,
            inputs.V,
            masks=structured_mw,
            reward_weights=nc.reward_weights,
            merge_pi=nc.merge_pi,
            adv_normalize=nc.adv_normalize,
            clip_range=nc.clip_range,
            sync_statistics=nc.sync_statistics,
          )
        elif nc.use_loss_type in ['rl_ppo2', 'rl_vtrace']:
          # Note: we need convert the shape (batch_size, ...) to the shape
          # (T, B, ...) where T=nc.rollout_len, B=nc.nrollout, batch_size=B*T
          # When computing ppo2-loss and value-loss, only T-1 time steps are
          # used due to the value bootstrap at the tail. When doing so, the
          # [:-1] indexing, leading to (T - 1, B, ...) tensor slice, makes life
          # much easier

          def _batch_to_TB(tsr):
            return tf.transpose(tf.reshape(
              tsr, shape=(nc.nrollout, nc.rollout_len)))

          # make the len=n_action_heads lists for action-head stuff
          # for tensor entry, shape (batch_size, ...) -> shape (T, B, ...)
          neglogp_list = [_batch_to_TB(neglogp)
                          for neglogp in nest.flatten(outer_fed_head_neglogp)]
          oldneglogp_list = [_batch_to_TB(oldneglogp)
                             for oldneglogp in nest.flatten(inputs.neglogp)]
          mask_list = [_batch_to_TB(mw) for mw in nest.flatten(structured_mw)]
          # make the len=n_v lists for value-head stuff
          # for tensor entry, shape (batch_size, ...) -> shape (T, B, ...)
          # as aforementioned
          vpred_list = [_batch_to_TB(v)
                        for v in tf.split(value_head, nc.n_v, axis=1)]
          reward_list = [_batch_to_TB(r)
                         for r in tf.split(inputs.r, nc.n_v, axis=1)]
          discounts = _batch_to_TB(inputs.discount)
          # upgo_loss only use the win_loss, i.e, v[0]
          upgo_loss = tp_losses.upgo_loss(tf.stack(neglogp_list, axis=-1),
                                          tf.stack(oldneglogp_list, axis=-1),
                                          tf.stack(mask_list, axis=-1),
                                          vpred_list[0], reward_list[0],
                                          discounts)
          loss_endpoints['upgo_loss'] = upgo_loss

          if nc.use_loss_type == 'rl_ppo2':
            # PPO2 loss
            # reward_weights size should be consistent with n_v
            reward_weights = tf.squeeze(tf.convert_to_tensor(
              nc.reward_weights, tf.float32))
            assert reward_weights.shape.as_list()[0] == len(reward_list), (
              'For ppo2 loss, reward_weight size must be the same with number of'
              ' value head: each reward_weight element must correspond to one '
              'value-head exactly.'
            )

            # lambda for td-lambda or lambda-return
            assert nc.lam is not None, ('building rl_ppo2, but lam for '
                                        'lambda-return is None.')
            lam = tf.convert_to_tensor(nc.lam, tf.float32)

            # for each value-head, compute the corresponding policy gradient loss
            # and the value loss
            pg_loss, value_loss = [], []
            for vpred, reward in zip(vpred_list, reward_list):
              # compute the lambda-Return `R` in shape (T - 1, B)
              # [:-1] means discarding the last one,
              # [1:] means an off-one alignment.
              # back_prop=False means R = tf.stop_gradient(R)
              with tf.device("/cpu:0"):
                R = multistep_forward_view(reward[:-1], discounts[:-1], vpred[1:],
                                           lambda_=lam, back_prop=False)
              # compute the ppo2 loss using this value-head for each of the
              # n_action_heads action-head; then reduce them
              # [:-1] means discarding the last one and using only T - 1 time
              # steps
              _ploss = [
                tp_losses.ppo2_loss(neglogp[:-1],
                                    oldneglogp[:-1],
                                    tf.stop_gradient(vpred)[:-1],
                                    R,  # has been stop_gradient above
                                    mask[:-1],
                                    adv_normalize=nc.adv_normalize,
                                    clip_range=nc.clip_range,
                                    sync_statistics=nc.sync_statistics)
                for neglogp, oldneglogp, mask in zip(
                  neglogp_list, oldneglogp_list, mask_list)
              ]
              pg_loss.append(tf.reduce_sum(_ploss))
              # compute the value loss for this value-head
              value_loss.append(tf.reduce_mean(0.5 * tf.square(R - vpred[:-1])))
            # element-wise times reward_weight and the pg_loss for that value-head
            pg_loss = tf.stack(pg_loss) * reward_weights  # shape (n_v,)
            # make the final pg_loss, value_loss in desired format
            pg_loss = tf.reduce_sum(pg_loss)
            value_loss = tf.stack(value_loss)
          else:
            # vtrace loss
            # lambda for td-lambda or lambda-return
            assert nc.lam is not None, ('building rl_vtrace, but lam for '
                                        'td-lambda is None.')
            lam = tf.convert_to_tensor(nc.lam, tf.float32)
            value_loss = []
            for values, rewards in zip(vpred_list, reward_list):
              value_loss.append(
                tp_losses.td_lambda(values, rewards, discounts, lam=lam))
            shaped_values = tf.matmul(value_head, nc.reward_weights,
                                      transpose_b=True)
            shaped_rewards = tf.matmul(inputs.r, nc.reward_weights,
                                       transpose_b=True)
            values = tf.transpose(tf.reshape(
              shaped_values, shape=(nc.nrollout, nc.rollout_len)))
            rewards = tf.transpose(tf.reshape(
              shaped_rewards, shape=(nc.nrollout, nc.rollout_len)))
            pg_loss = tf.reduce_sum(
              [tp_losses.vtrace_loss(neglogp, oldneglogp, mask, values,
                                     rewards, discounts, 1.0, 1.0)
               for oldneglogp, neglogp, mask in zip(oldneglogp_list,
                                                    neglogp_list, mask_list)])
            value_loss = tf.stack(value_loss)

        # TODO: maybe more rl endpoints
        # policy gradient loss must be scalar
        loss_endpoints['pg_loss'] = pg_loss
        #  value loss can be scalar or vector
        if len(value_loss.shape) == 0:
          loss_endpoints['value_loss'] = value_loss
        else:
          for i in range(value_loss.shape[0]):
            loss_endpoints['value_loss_' + str(i)] = value_loss[i]
        for k, v in entropy_loss.items():
          loss_endpoints['ent_' + k] = v
        if nc.distillation:
          for k, v in distill_loss.items():
            loss_endpoints['distill_' + k] = v
          loss_endpoints['distill_ab_bf4mins'] = ab_distill_loss
    else:
      print('use_loss_type: {}. Nothing done.'.format(nc.use_loss_type))
      pass

    return MNetV6Losses(
      total_reg_loss=total_reg_loss,
      total_il_loss=total_il_loss,
      pg_loss=pg_loss,
      value_loss=value_loss,
      entropy_loss=entropy_loss,
      distill_loss=distill_loss,
      loss_endpoints=loss_endpoints
    )


# APIs
net_build_fun = mnet_v6d6
net_config_cls = MNetV6Config
net_inputs_placeholders_fun = mnet_v6d6_inputs_placeholders


def mnet_v6d5_consts(nc: MNetV6Config, scope=None) -> MNetV6d5Consts:
  with tf.variable_scope(scope, default_name='mnet_v5_consts'):
    arg_mask = _const_arg_mask(nc)
    mask_base_poses = _const_mask_base_poses(nc)
    base_poses, base = _const_base_poses(nc)
    select_type_func_mask = _const_select_type_mask(nc)
    tar_u_type_func_mask = _const_tar_u_type_mask(nc)
  return MNetV6d5Consts(arg_mask,
                        mask_base_poses,
                        base_poses, base,
                        select_type_func_mask,
                        tar_u_type_func_mask)