from collections import OrderedDict
from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.layers.python.layers import utils as lutils
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope

from tpolicies import ops as tp_ops, layers as tp_layers, tp_utils as tp_utils
from tpolicies.layers import lstm_embed_block
from tpolicies.net_zoo.mnet_v6.data import MNetV6Config, MNetV5EmbedScope, \
  MNetV5Inputs, MNetV5TrainableVariables
from tpolicies.net_zoo.mnet_v6.data import MNetV6TrainableVariables
from tpolicies.utils.distributions import make_pdtype


@add_arg_scope
def _const_select_type_mask(nc: MNetV6Config, outputs_collections=None):
  return lutils.collect_named_outputs(outputs_collections, 'const_select_type_mask',
                                      tf.constant(nc.select_type_func_mask))


@add_arg_scope
def _const_tar_u_type_mask(nc: MNetV6Config, outputs_collections=None):
  return lutils.collect_named_outputs(outputs_collections, 'const_tar_u_type_mask',
                                      tf.constant(nc.tar_u_type_func_mask))


def _make_mnet_v6_arg_scope_a(nc: MNetV6Config, endpoints_collections: str):
  ep_fns = []
  if nc.endpoints_verbosity >= 2:
    ep_fns += [
      _units_embed_block,
      _transformer_block,
      _transformer_block_v2,
      _transformer_block_v3,
      _transformer_block_v4,
      _scatter_units_block,
      _spa_embed_block_v2,
      _vec_embed_block_v2,
      _vec_embed_block_v2d1,
      _vec_embed_block_v3,
      _vec_embed_block_v3d1,
      _last_action_embed_block_mnet_v6,
      _last_action_embed_block_mnet_v6_v2,
      _zstat_embed,
      lstm_embed_block,
    ]
  if nc.endpoints_verbosity >= 4:
    ep_fns += [
      _const_arg_mask,
      _const_mask_base_poses,
      _const_base_poses,
      _pre_discrete_action_res_block,
      _pre_discrete_action_fc_block,
      _pre_ptr_action_res_block,
      _pre_loc_action_astar_like_block_v1,
    ]
  if nc.endpoints_verbosity >= 6:
    ep_fns += [
      tp_layers.linear_embed,
      tfc_layers.fully_connected,
      tfc_layers.conv1d,
      tfc_layers.conv2d,
      tfc_layers.max_pool2d,
    ]
  weights_regularizer = (None if nc.weight_decay is None
                         else l2_regularizer(nc.weight_decay))

  with arg_scope(ep_fns,
                 outputs_collections=endpoints_collections):
    with arg_scope([tfc_layers.conv1d,
                    tfc_layers.conv2d,
                    tfc_layers.fully_connected,
                    tp_layers.linear_embed,
                    tp_layers.lstm,
                    tp_layers.k_lstm],
                   weights_regularizer=weights_regularizer):
      with arg_scope([tp_layers.linear_embed],
                     weights_initializer=xavier_initializer()):
        with arg_scope(
            [tfc_layers.conv2d],
            weights_initializer=variance_scaling_initializer()) as arg_sc:
          return arg_sc


def _make_mnet_v6_vars(scope) -> MNetV6TrainableVariables:
  scope = scope if isinstance(scope, str) else scope.name + '/'
  all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
  lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'lstm_embed'))
  return MNetV6TrainableVariables(all_vars=all_vars, lstm_vars=lstm_vars)


def _make_mnet_v6_endpoints_dict(nc: MNetV6Config, endpoints_collections: str,
                                 name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask', alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese', alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses', alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base', alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embeddings
  _safe_collect(ep_key='ab_emb', alias='shared/ab_embed',
                scope='{}.*{}.*'.format(name_scope, 'ab_embed'))
  _safe_collect(ep_key='noop_emb', alias='shared/noop_num_embed',
                scope='{}.*{}.*'.format(name_scope, 'noop_num_embed'))
  _safe_collect(ep_key='buff_emb', alias='shared/buff_embed',
                scope='{}.*{}.*'.format(name_scope, 'buff_embed'))
  _safe_collect(ep_key='order_emb', alias='shared/order_embed',
                scope='{}.*{}.*'.format(name_scope, 'order_embed'))
  _safe_collect(ep_key='u_type_emb', alias='shared/u_type_embed',
                scope='{}.*{}.*'.format(name_scope, 'u_type_embed'))

  # units embeddings related
  _safe_collect(ep_key='units_emb',
                alias='transformer_encoder/emb',
                scope='{}.*{}'.format(name_scope, 'transformer_encoder'))
  _safe_collect(ep_key='units_avg_emb',
                alias='transformer_encoder/avg_emb',
                scope='{}.*{}'.format(name_scope, 'transformer_encoder'))

  # joint unit-map embeddings related
  _safe_collect(ep_key='spa_emb_skip0',
                alias='spa_embed/map_skip0',
                scope='{}.*{}.*'.format(name_scope, 'spa_embed'))
  _safe_collect(ep_key='spa_emb_skip1',
                alias='spa_embed/map_skip1',
                scope='{}.*{}.*'.format(name_scope, 'spa_embed'))
  _safe_collect(ep_key='spa_emb_skip2',
                alias='spa_embed/map_skip2',
                scope='{}.*{}.*'.format(name_scope, 'spa_embed'))
  _safe_collect(ep_key='spa_emb_vec',
                alias='spa_embed/vec',
                scope='{}.*{}.*'.format(name_scope, 'spa_embed'))

  # vector embeddings related
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))

  # last action embeddings related
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))
  # integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))
  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed/out',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))
  # pre-layer for the heads
  # lvl 0
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 1
  _safe_collect(ep_key='head_noop_pre',
                alias='pre_noop_action_block',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 2
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 3, select
  _safe_collect(ep_key='head_sel_keys_pre', alias='selection_raw_keys',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))

  return ep


@add_arg_scope
def _const_arg_mask(nc: MNetV6Config, outputs_collections=None):
  return lutils.collect_named_outputs(outputs_collections, 'const_arg_mask',
                                      tf.constant(nc.arg_mask))


@add_arg_scope
def _const_mask_base_poses(nc: MNetV6Config, outputs_collections=None):
  mbp = tf.constant(
    value=[[True] * len(nc.ideal_base_pos_dict[key]) +
           [not nc.use_base_mask] * (
               nc.max_bases_num - len(nc.ideal_base_pos_dict[key]))
           for key in nc.ideal_base_pos_dict] +
          [[not nc.use_base_mask] * nc.max_bases_num] * (
              nc.max_map_num - len(nc.ideal_base_pos_dict)),
    shape=(nc.max_map_num, nc.max_bases_num),
    dtype=tf.bool)
  return lutils.collect_named_outputs(outputs_collections,
                                      'const_mask_base_posese', mbp)


@add_arg_scope
def _const_base_poses(nc: MNetV6Config, outputs_collections=None):
  modified_ideal_base_pos_dict = deepcopy(nc.ideal_base_pos_dict)
  for key in modified_ideal_base_pos_dict:
    new_coors = []
    for coor in modified_ideal_base_pos_dict[key]:
      new_coor_x = coor[0] * nc.map_max_row_col[0] / float(
        nc.map_ori_size_dict[key][0])
      new_coor_y = coor[1] * nc.map_max_row_col[1] / float(
        nc.map_ori_size_dict[key][1])
      new_coors.append([new_coor_x, new_coor_y])
    modified_ideal_base_pos_dict[key] = new_coors

  # float one
  base_poses = [modified_ideal_base_pos_dict[key] +
                [[0.0, 0.0]] * (
                    nc.max_bases_num - len(modified_ideal_base_pos_dict[key]))
                for key in modified_ideal_base_pos_dict] + \
               [[[0.0, 0.0]] * nc.max_bases_num * (
                   nc.max_map_num - len(modified_ideal_base_pos_dict))]
  t_base_poses = tf.constant(
    value=base_poses,
    shape=(nc.max_map_num, nc.max_bases_num, 2),
    dtype=tf.float32)
  t_base_poses = tp_ops.to_int32(t_base_poses)

  # binarized one
  bi_base_poses = base_poses
  for i in range(nc.max_map_num):
    for j in range(nc.max_bases_num):
      coor = bi_base_poses[i][j]
      # len(bi_x) = 9, len(bi_y) = 9
      bi_x = [int(i) for i in '{0:09b}'.format(int(coor[0] * 2))]
      bi_y = [int(i) for i in '{0:09b}'.format(int(coor[1] * 2))]
      bi_base_poses[i][j] = [1 if k == i else 0 for k in
                             range(nc.max_map_num)] + bi_x + bi_y
  t_base = tf.constant(
    value=bi_base_poses,
    shape=(nc.max_map_num, nc.max_bases_num, nc.max_map_num + 18),
    dtype=tf.float32)
  return (
    lutils.collect_named_outputs(outputs_collections, 'const_base_poses',
                                 t_base_poses),
    lutils.collect_named_outputs(outputs_collections, 'const_base', t_base)
  )


@add_arg_scope
def _units_embed_block(inputs,  # ob_space-like
                       embed_sc: MNetV5EmbedScope,
                       nc: MNetV6Config,
                       outputs_collections=None):
  with tf.variable_scope('units_embed') as sc:
    x_u_dense_enc = tfc_layers.fully_connected(inputs['X_UNIT_FEAT'], nc.enc_dim,
                                               scope='u_dense_fc1')
    x_u_dense_enc = tfc_layers.fully_connected(x_u_dense_enc, nc.enc_dim,
                                               scope='u_dense_fc2')

    with arg_scope([tp_layers.linear_embed],
                   enc_size=nc.enc_dim):
      x_u_buff_enc = tp_layers.linear_embed(
        inputs=tp_ops.to_int32(inputs['X_UNIT_BUFF']),
        vocab_size=nc.buff_embed_size,
        scope=embed_sc.buff_embed_sc
      )
      x_u_order_enc = tp_layers.linear_embed(
        inputs=tp_ops.to_int32(inputs['X_UNIT_ORDER']),
        vocab_size=nc.order_embed_size,
        scope=embed_sc.order_embed_sc
      )
      x_u_type_enc = tp_layers.linear_embed(
        inputs=tp_ops.to_int32(inputs['X_UNIT_TYPE']),
        vocab_size=nc.u_type_embed_size,
        scope=embed_sc.u_type_embed_sc
      )
    x_units_enc = tf.concat(
      [x_u_dense_enc, x_u_buff_enc, x_u_order_enc, x_u_type_enc], axis=-1)
    # TODO: trans net here if use
    x_units_enc = tfc_layers.fully_connected(x_units_enc, nc.enc_dim, scope='fc3')
    # Caution! We must mask out invalid units to have zero embeddings, because
    # the scatter op later
    x_units_enc = tp_ops.mask_embed(embed=x_units_enc, mask=inputs['MASK_LEN'])

    # Caution! We make a compromise that the last two should be is_selected
    # and was_target
    coor = [tp_ops.to_int32(inputs['X_UNIT_COOR'][:, :, 0]),
            tp_ops.to_int32(inputs['X_UNIT_COOR'][:, :, 1])]
    # [bs, 600] * 2
    is_selected_mask = tp_ops.to_bool(inputs['X_UNIT_FEAT'][:, :, -2])
    was_tar_mask = tp_ops.to_bool(inputs['X_UNIT_FEAT'][:, :, -1])
    return (
      lutils.collect_named_outputs(outputs_collections, sc.name, x_units_enc),
      coor, is_selected_mask, was_tar_mask
    )


@add_arg_scope
def _selected_embed_block(inputs_units_embed, mask, outputs_collections=None):
  """
  :param inputs_units_embed: [bs, 600, some dim]
  :param mask: [bs, 600]
  :return:
  """
  with tf.variable_scope('selected_embed') as sc:
    selected_embed = tp_ops.mask_embed(embed=inputs_units_embed, mask=mask)
    selected_embed_max_pool = tf.reduce_max(selected_embed, axis=1)
    selected_embed_mean_pool = tf.reduce_mean(selected_embed, axis=1)
    return lutils.collect_named_outputs(
      outputs_collections,
      sc.name,
      tf.concat([selected_embed_max_pool, selected_embed_mean_pool], axis=1)
    )


@add_arg_scope
def _scatter_units_block(inputs_units_embed, inputs_xy, coord_sys, nc,
                         outputs_collections=None):
  """Scatter units

  Args:
    units_embed: [bs, 600, dim]
    inputs_xy:
    coord_sys:
    nc:
    outputs_collections:

  Returns
    A scatter
  """
  with tf.variable_scope('scatter_units') as sc:
    nbatch = tf.shape(inputs_units_embed)[0]
    batch_idx = tf.tile(tf.expand_dims(tf.range(nbatch), 1), [1, nc.tar_unit_dim])
    # [bs, 600, 3]
    idx = tf.stack((batch_idx,) + coord_sys.xy_to_rc(inputs_xy[0], inputs_xy[1]),
                   axis=-1)
    # values will be summed up for repeated idx in scatter_nd
    # TODO: must check empty units' units_embed has been masked to be zero vectors
    scattered_img = tf.scatter_nd(indices=idx,
                                  updates=inputs_units_embed,
                                  shape=[nbatch,
                                         coord_sys.map_r_max,
                                         coord_sys.map_c_max,
                                         inputs_units_embed.get_shape()[2]])
    # TODO: otherwise you will need the following codes
    # set trans_to_spatial[:, 0, 0, :] = 0 in case padding in units' features
    # zz_mask = tf.zeros(shape=[tf.shape(scattered_img)[0],
    #                           1,
    #                           tf.shape(scattered_img)[-1]])
    # zz_mask = tf.concat([zz_mask,
    #                      tf.ones(shape=[tf.shape(scattered_img)[0],
    #                                     tf.shape(scattered_img)[1]-1,
    #                                     tf.shape(scattered_img)[-1]])],
    #                     axis=1)
    # zz_mask = tf.expand_dims(zz_mask, axis=2)
    # zz_mask = tf.concat([zz_mask,
    #                      tf.ones(shape=[tf.shape(scattered_img)[0],
    #                                     tf.shape(scattered_img)[1],
    #                                     tf.shape(scattered_img)[2]-1,
    #                                     tf.shape(scattered_img)[-1]])],
    #                     axis=2)
    # zz_mask = tf.stop_gradient(zz_mask)
    # scattered_img = tf.multiply(scattered_img, zz_mask)
    return lutils.collect_named_outputs(outputs_collections, sc.name,
                                        scattered_img)


@add_arg_scope
def _spa_embed_block_v2(inputs_img, inputs_additonal_img, nc,
                        outputs_collections=None):

  def _downsampled_convs(x_):
    # (bs, H, W, dim)
    x_ = tfc_layers.conv2d(x_, 32, [1, 1])
    # (bs, H, W, 32)
    x_ = tfc_layers.conv2d(x_, 32, [4, 4], stride=2)
    # (bs, H/2, W/2, 32)
    x_ = tfc_layers.conv2d(x_, 64, [3, 3], stride=2)
    # (bs, H/4, W/4, 64)
    x_ = tfc_layers.conv2d(x_, 128, [3, 3], stride=2)
    # (bs, H/8, W/8, 128)
    return x_

  def _my_res_bottleneck_v1(x_):
    """my simple bottleneck v1 impl. No normalization intentionally."""
    skip = x_
    # (bs, HH, WW, 128)
    x_ = tfc_layers.conv2d(x_, 32, [1, 1])
    # (bs, HH, WW, 32)
    x_ = tfc_layers.conv2d(x_, 32, [3, 3])
    # (bs, HH, WW, 32)
    x_ = tfc_layers.conv2d(x_, 128, [3, 3], activation_fn=None)
    # (bs, HH, WW, 128)
    x_ = tf.nn.relu(skip + x_)
    return x_

  with tf.variable_scope('spa_embed') as sc:
    x = tf.concat([inputs_img, inputs_additonal_img], axis=3)  # NHWC presumed
    # (bs, H, W, dim)
    x = _downsampled_convs(x)
    # (bs, H/8, W/8, 128)

    # res blocks
    map_skip = []
    for i in range(nc.spa_n_blk):
      x = _my_res_bottleneck_v1(x)
      map_skip.append(x)  # NOTE(pengsun): intentional after-as-skip

    # (bs, H/8, W/8, 128)
    x = tfc_layers.flatten(x)
    x = tfc_layers.fully_connected(x, nc.enc_dim)
    # (bs, enc_dim)

    for i, ms in enumerate(map_skip):
      lutils.collect_named_outputs(
        outputs_collections,
        sc.original_name_scope + 'map_skip' + str(i),
        ms
      )
    return map_skip, lutils.collect_named_outputs(
      outputs_collections, sc.original_name_scope + 'vec', x
    )


def _multi_dilated_conv(x_enc, rates=(3, 6, 12, 24), c_dim=16, k_size=3):
  xx_rates = []
  for rate in rates:
    xx = tfc_layers.conv2d(x_enc, c_dim, [k_size, k_size], rate=rate,
                           scope=('rate%d_conv1' % rate))
    xx = tfc_layers.conv2d(xx, c_dim, [1, 1], scope=('rate%d_conv2' % rate))
    xx_rates.append(xx)
  x_out = tf.add_n(xx_rates, name='added_out')
  return x_out


@add_arg_scope
def _gather_units_block(inputs_spa_embed, inputs_units_coor, mask, coord_sys,
                        scope=None, outputs_collections=None):
  with tf.variable_scope(scope, default_name='gather_units') as sc:
    nbatch = tf.shape(inputs_spa_embed)[0]
    units_num = tf.shape(inputs_units_coor[0])[-1]
    batch_idx = tf.tile(tf.expand_dims(tf.range(nbatch), 1), [1, units_num])
    # [bs, 600, 3]
    idx = tf.stack((batch_idx,) + coord_sys.xy_to_rc(
      inputs_units_coor[0], inputs_units_coor[1]), axis=-1)
    units_embed = tf.gather_nd(inputs_spa_embed, idx)
    # mask
    units_embed = tp_ops.mask_embed(embed=units_embed, mask=mask)
  return lutils.collect_named_outputs(outputs_collections, sc.name, units_embed)


@add_arg_scope
def _u2d_spa_embed_block(inputs_u2d, out_dim, ch_dim, outputs_collections=None):
  x = inputs_u2d
  with tf.variable_scope('u2d_spa_embed') as sc:
    # (bs, H, W, C)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3], normalizer_fn=tp_layers.inst_ln)
    # (bs, H, W, ch_dim)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3])
    # (bs, H, W, ch_dim)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3])
    # (bs, H, W, ch_dim)
    x = tfc_layers.conv2d(x, out_dim, [3, 3], normalizer_fn=tp_layers.inst_ln)
    # (bs, H, W, out_dim)
  return lutils.collect_named_outputs(outputs_collections, sc.name + '_out', x)


@add_arg_scope
def _map_spa_embed_block(inputs_map, out_dim, ch_dim, outputs_collections=None):
  x = inputs_map
  with tf.variable_scope('map_spa_embed') as sc:
    # (bs, H, W, C)
    x = tfc_layers.conv2d(x, 2 * ch_dim, [3, 3], stride=2,
                          normalizer_fn=tp_layers.inst_ln)
    # (bs, H/2, W/2, 2*ch_dim)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3], stride=2)
    # (bs, H/4, W/4, ch_dim)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3], stride=2)
    # (bs, H/8, W/8, ch_dim)
    lutils.collect_named_outputs(outputs_collections, sc.name + '_small', x)
    x = tp_layers.res_sum_bottleneck_blocks_v2(x, n_blk=2, n_skip=1,
                                               ch_dim=ch_dim,
                                               bottleneck_ch_dim=32,
                                               k_size=3)
    lutils.collect_named_outputs(outputs_collections, sc.name + '_res', x)
    # (bs, H/8, W/8, ch_dim)
    x = tfc_layers.conv2d_transpose(x, ch_dim, [3, 3], stride=2)
    x = tfc_layers.conv2d_transpose(x, ch_dim, [3, 3], stride=2)
    x = tfc_layers.conv2d_transpose(x, ch_dim, [3, 3], stride=2)
    lutils.collect_named_outputs(outputs_collections, sc.name + '_resize', x)
    # (bs, H, W, ch_dim)
    x = tfc_layers.conv2d(x, out_dim, [1, 1])
    # (bs, H, W, out_dim)
  return lutils.collect_named_outputs(outputs_collections, sc.name + '_out', x)


@add_arg_scope
def _map_spa_embed_block_v2(inputs_map, out_dim, ch_dim,
                            outputs_collections=None):
  output_size = [tf.shape(inputs_map)[1], tf.shape(inputs_map)[2]]
  x = inputs_map
  with tf.variable_scope('map_spa_embed_v2') as sc:
    # (bs, H, W, C)
    x = tfc_layers.conv2d(x, 2 * ch_dim, [3, 3], stride=2,
                          normalizer_fn=tp_layers.inst_ln)
    # (bs, H/2, W/2, 2*ch_dim)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3], stride=2)
    # (bs, H/4, W/4, ch_dim)
    x = tfc_layers.conv2d(x, ch_dim, [3, 3], stride=2)
    lutils.collect_named_outputs(outputs_collections, sc.name + '_small', x)
    # (bs, H/8, W/8, ch_dim)
    x = tp_layers.res_sum_bottleneck_blocks_v2(x, n_blk=2, n_skip=1,
                                               ch_dim=ch_dim,
                                               bottleneck_ch_dim=32,
                                               k_size=3)
    lutils.collect_named_outputs(outputs_collections, sc.name + '_res', x)
    # (bs, H/8, W/8, ch_dim)
    x = tf.image.resize_bilinear(x, output_size)
    lutils.collect_named_outputs(outputs_collections, sc.name + '_resize', x)
    # (bs, H, W, ch_dim)
    x = tfc_layers.conv2d(x, out_dim, [1, 1])
    # (bs, H, W, out_dim)
  return lutils.collect_named_outputs(outputs_collections, sc.name + '_out', x)


@add_arg_scope
def _vec_embed_block(inputs, enc_dim, outputs_collections=None):
  with tf.variable_scope("vec_enc") as sc:
    x_stat = tfc_layers.fully_connected(inputs['X_VEC_PLAYER_STAT'], 64,
                                        scope='x_stat')
    x_upgrad = tfc_layers.fully_connected(inputs['X_VEC_UPGRADE'], 128,
                                          scope='x_upgrad')
    x_game_prog = tfc_layers.fully_connected(inputs['X_VEC_GAME_PROG'], 64,
                                             scope='x_game_prog')
    x_s_ucnt = tfc_layers.fully_connected(inputs['X_VEC_S_UNIT_CNT'], 128,
                                          scope='x_s_ucnt')
    x_e_ucnt = tfc_layers.fully_connected(inputs['X_VEC_E_UNIT_CNT'], 128,
                                          scope='x_e_ucnt')
    x_af_ucnt = tfc_layers.fully_connected(inputs['X_VEC_AF_UNIT_CNT'], 64,
                                           scope='x_af_ucnt')
    x_vec_prog = tfc_layers.fully_connected(inputs['X_VEC_PROG'], 64,
                                            scope='x_vec_prog')
    x_map = tfc_layers.fully_connected(inputs['MAP_INDICATOR'], 64,
                                       scope='x_map')
    x_mmr = tfc_layers.fully_connected(inputs['MMR'], 64, scope='x_mmr')

    vec_embed = tf.concat([x_stat, x_upgrad, x_game_prog,
                           x_s_ucnt, x_e_ucnt, x_af_ucnt,
                           x_vec_prog, x_map, x_mmr], axis=-1)
    vec_embed = tfc_layers.fully_connected(vec_embed, 512, scope='x_fc1')
    vec_embed = tfc_layers.fully_connected(vec_embed, 2 * enc_dim,
                                           scope='x_fc2')
    return lutils.collect_named_outputs(outputs_collections, sc.name, vec_embed)


@add_arg_scope
def _vec_embed_block_v2(inputs, enc_dim, outputs_collections=None):
  vec_embed, _dummy = _vec_embed_block_v2d1(inputs, enc_dim)
  return vec_embed


@add_arg_scope
def _vec_embed_block_v2d1(inputs, enc_dim, outputs_collections=None):
  with tf.variable_scope("vec_enc") as sc:
    x_stat = tfc_layers.fully_connected(inputs['X_VEC_PLAYER_STAT'], 64,
                                        scope='x_stat')
    x_upgrad = tfc_layers.fully_connected(inputs['X_VEC_UPGRADE'], 128,
                                          scope='x_upgrad')
    x_game_prog = tfc_layers.fully_connected(inputs['X_VEC_GAME_PROG'], 64,
                                             scope='x_game_prog')
    x_s_ucnt = tfc_layers.fully_connected(inputs['X_VEC_S_UNIT_CNT'], 128,
                                          scope='x_s_ucnt')
    x_e_ucnt = tfc_layers.fully_connected(inputs['X_VEC_E_UNIT_CNT'], 128,
                                          scope='x_e_ucnt')
    x_af_ucnt = tfc_layers.fully_connected(inputs['X_VEC_AF_UNIT_CNT'], 64,
                                           scope='x_af_ucnt')
    x_vec_prog = tfc_layers.fully_connected(inputs['X_VEC_PROG'], 64,
                                            scope='x_vec_prog')
    x_map = tfc_layers.fully_connected(inputs['MAP_INDICATOR'], 64,
                                       scope='x_map')
    x_mmr = tfc_layers.fully_connected(inputs['MMR'], 64, scope='x_mmr')
    x_ab_mask = tfc_layers.fully_connected(tf.cast(inputs['MASK_AB'], tf.float32), 64)

    vec_embed = tf.concat([x_stat, x_upgrad, x_game_prog,
                           x_s_ucnt, x_e_ucnt, x_af_ucnt,
                           x_vec_prog, x_map, x_mmr, x_ab_mask], axis=-1)
    vec_embed = tfc_layers.fully_connected(vec_embed, 512, scope='x_fc1')
    vec_embed = tfc_layers.fully_connected(vec_embed, 2 * enc_dim,
                                           scope='x_fc2')
    return (
      lutils.collect_named_outputs(outputs_collections, sc.name, vec_embed),
      x_ab_mask
    )


@add_arg_scope
def _vec_embed_block_v3(inputs, enc_dim, outputs_collections=None):
  """vector embeddings v3

  In the opposite of v1/v2, this block
  * has no MAP_INDICATOR
  * has no extra fc layers
  * has bigger fc dim for `x_stat`
  The architecture is more similar to AStar.
  """
  vec_embed, _dummy = _vec_embed_block_v3d1(inputs, enc_dim)
  return vec_embed


@add_arg_scope
def _vec_embed_block_v3d1(inputs, enc_dim, outputs_collections=None):
  """vector embeddings v3.1

  The same with v3, but returns an extra embeddings
  """
  with tf.variable_scope("vec_enc") as sc:
    x_stat = tfc_layers.fully_connected(inputs['X_VEC_PLAYER_STAT'], 128,
                                        scope='x_stat')
    x_upgrad = tfc_layers.fully_connected(inputs['X_VEC_UPGRADE'], 128,
                                          scope='x_upgrad')
    x_game_prog = tfc_layers.fully_connected(inputs['X_VEC_GAME_PROG'], 64,
                                             scope='x_game_prog')
    x_s_ucnt = tfc_layers.fully_connected(inputs['X_VEC_S_UNIT_CNT'], 128,
                                          scope='x_s_ucnt')
    x_e_ucnt = tfc_layers.fully_connected(inputs['X_VEC_E_UNIT_CNT'], 128,
                                          scope='x_e_ucnt')
    x_af_ucnt = tfc_layers.fully_connected(inputs['X_VEC_AF_UNIT_CNT'], 64,
                                           scope='x_af_ucnt')
    x_vec_prog = tfc_layers.fully_connected(inputs['X_VEC_PROG'], 64,
                                            scope='x_vec_prog')
    x_mmr = tfc_layers.fully_connected(inputs['MMR'], 64, scope='x_mmr')
    x_ab_mask = tfc_layers.fully_connected(
      tf.cast(inputs['MASK_AB'], tf.float32),
      64,
      scope='x_available_action'
    )

    vec_embed = tf.concat([x_stat, x_upgrad, x_game_prog,
                           x_s_ucnt, x_e_ucnt, x_af_ucnt,
                           x_vec_prog, x_mmr, x_ab_mask], axis=-1)
    return (
      lutils.collect_named_outputs(outputs_collections, sc.name, vec_embed),
      x_ab_mask
    )


@add_arg_scope
def _last_action_embed_block(inputs,
                             inputs_arg_mask,
                             inputs_base_embed,
                             ab_embed_sc,
                             nc: MNetV6Config,
                             outputs_collections=None):
  """
  '0:ab', '1:noop', '2:ss', '3:ms', '4:sft', '5:cmd_u',
  '6:pos', '7:creep_pos', '8:nydus_pos', '9:base', '10:unload'
  :param action: tuple of [bs, some dim]
  :param units_embed:
  :return:
  """

  def _make_spa_embed(img):
    x = tfc_layers.conv2d(img, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [88, 100]
    x = tfc_layers.conv2d(x, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [44, 50]
    x = tfc_layers.conv2d(x, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [22, 25]
    x = tfc_layers.conv2d(x, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [11, 12]
    x = tfc_layers.flatten(x)
    return x

  with tf.variable_scope('last_action_embed') as sc:
    ab_embed = tp_layers.linear_embed(inputs=inputs['A_AB'], vocab_size=nc.ab_dim,
                                      enc_size=nc.enc_dim, scope=ab_embed_sc)

    arg_mask = tf.nn.embedding_lookup(inputs_arg_mask, inputs['A_AB'])
    arg_mask = tp_ops.to_float32(arg_mask)
    arg_mask = tf.expand_dims(arg_mask, axis=-1)

    noop_embed = tf.one_hot(inputs['A_NOOP_NUM'], depth=nc.noop_dim,
                            dtype=tf.float32)
    noop_embed = tf.multiply(noop_embed, arg_mask[:, 0, :])

    sft_embed = tp_ops.to_float32(tf.expand_dims(inputs['A_SHIFT'], axis=1))
    sft_embed = tf.multiply(sft_embed, arg_mask[:, 3, :])

    r, c = nc.map_max_row_col
    pos_embed = tf.multiply(tf.one_hot(inputs['A_CMD_POS'], depth=r * c),
                            arg_mask[:, 5, :])
    pos_embed = tf.reshape(pos_embed, [-1, r, c])
    creep_pos_embed = tf.multiply(tf.one_hot(inputs['A_CMD_CREEP_POS'],
                                             depth=r * c),
                                  arg_mask[:, 6, :])
    creep_pos_embed = tf.reshape(creep_pos_embed, [-1, r, c])
    nydus_pos_embed = tf.multiply(tf.one_hot(inputs['A_CMD_NYDUS_POS'],
                                             depth=r * c),
                                  arg_mask[:, 7, :])
    nydus_pos_embed = tf.reshape(nydus_pos_embed, [-1, r, c])

    # [bs, map_num], [map_num, base_num * dim]
    cur_dict = tf.matmul(
      inputs['MAP_INDICATOR'],
      tf.reshape(inputs_base_embed, shape=[nc.max_map_num, -1])
    )
    # [bs, base_num * dim]
    cur_dict = tf.reshape(
      cur_dict,
      shape=(-1, nc.max_bases_num, inputs_base_embed.shape[-1])
    )
    # [bs, base_num, dim]

    # base_embed = tf.nn.embedding_lookup(cur_dict, action[9])
    base_embed = tp_ops.fetch_op(cur_dict, inputs['A_CMD_BASE'])
    base_embed = tf.multiply(base_embed, arg_mask[:, 8, :])

    spa_embed = tf.stack([pos_embed, creep_pos_embed, nydus_pos_embed], axis=-1)
    spa_embed = _make_spa_embed(spa_embed)
    spa_embed = tfc_layers.fully_connected(spa_embed, nc.enc_dim)

    int_act_embed = tf.concat(
      [ab_embed, noop_embed, sft_embed, base_embed, spa_embed], axis=-1)
    int_act_embed = tfc_layers.fully_connected(int_act_embed, nc.enc_dim)
    return lutils.collect_named_outputs(outputs_collections, sc.name,
                                        int_act_embed)


@add_arg_scope
def _last_action_embed_block_mnet_v6(inputs,
                                     inputs_arg_mask,
                                     ab_embed_sc,
                                     nc: MNetV6Config,
                                     outputs_collections=None):
  """
  '0:ab', '1:noop', '2:sft', '3:select', '5:cmd_u', '6:pos'
  :param action: tuple of [bs, some dim]
  :param units_embed:
  :return:
  """

  def _make_spa_embed(img):
    x = tfc_layers.conv2d(img, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [88, 100]
    x = tfc_layers.conv2d(x, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [44, 50]
    x = tfc_layers.conv2d(x, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [22, 25]
    x = tfc_layers.conv2d(x, 4, [3, 3])
    x = tfc_layers.max_pool2d(x, [2, 2])  # [11, 12]
    x = tfc_layers.flatten(x)
    return x

  with tf.variable_scope('last_action_embed') as sc:
    ab_embed = tp_layers.linear_embed(inputs=inputs['A_AB'], vocab_size=nc.ab_dim,
                                      enc_size=nc.enc_dim, scope=ab_embed_sc)

    arg_mask = tf.nn.embedding_lookup(inputs_arg_mask, inputs['A_AB'])
    arg_mask = tp_ops.to_float32(arg_mask)
    arg_mask = tf.expand_dims(arg_mask, axis=-1)

    noop_embed = tf.one_hot(inputs['A_NOOP_NUM'], depth=nc.noop_dim,
                            dtype=tf.float32)
    noop_embed = tf.multiply(noop_embed, arg_mask[:, 0, :])

    sft_embed = tp_ops.to_float32(tf.expand_dims(inputs['A_SHIFT'], axis=1))
    sft_embed = tf.multiply(sft_embed, arg_mask[:, 1, :])

    r, c = nc.map_max_row_col
    pos_embed = tf.multiply(tf.one_hot(inputs['A_CMD_POS'], depth=r * c),
                            arg_mask[:, 4, :])
    pos_embed = tf.reshape(pos_embed, [-1, r, c, 1])

    spa_embed = _make_spa_embed(pos_embed)
    spa_embed = tfc_layers.fully_connected(spa_embed, nc.enc_dim)

    int_act_embed = tf.concat(
      [ab_embed, noop_embed, sft_embed, spa_embed], axis=-1)
    int_act_embed = tfc_layers.fully_connected(int_act_embed, nc.enc_dim)
    return lutils.collect_named_outputs(outputs_collections, sc.name,
                                        int_act_embed)


@add_arg_scope
def _last_action_embed_block_mnet_v6_v2(inputs,
                                        inputs_arg_mask,
                                        ab_embed_sc,
                                        nc: MNetV6Config,
                                        outputs_collections=None):
  """last action embeddings for mnet_v6, version 2

  In the opposite of v1, it
  * removes last-`cmd_pos` which can dominate other sibling features
  * embeds (by an fc) ab, noop, sft separately and then concatenates them.
  This architecture is more similar to AStar.
  """
  with tf.variable_scope('last_action_embed') as sc:
    ab_embed = tf.one_hot(inputs['A_AB'], depth=nc.ab_dim, dtype=tf.float32)
    ab_embed = tfc_layers.fully_connected(ab_embed, 128, scope='last_ab_embed')

    noop_embed = tf.one_hot(inputs['A_NOOP_NUM'], depth=nc.noop_dim,
                            dtype=tf.float32)
    noop_embed = tfc_layers.fully_connected(noop_embed, 64,
                                            scope='last_noop_embed')

    sft_embed = tf.one_hot(inputs['A_SHIFT'], depth=nc.shift_dim,
                           dtype=tf.float32)
    sft_embed = tfc_layers.fully_connected(sft_embed, 128,
                                           scope='last_sft_embed')

    int_act_embed = tf.concat([ab_embed, noop_embed, sft_embed], axis=-1)

    return lutils.collect_named_outputs(outputs_collections, sc.name,
                                        int_act_embed)


@add_arg_scope
def _integrate_embed(inputs_units_embed,
                     inputs_units_sel_embed,
                     inputs_spa_vec_embed,
                     inputs_vec_embed,
                     inputs_last_actions_embed,
                     nc,
                     outputs_collections=None):
  with tf.variable_scope('inte_embed') as sc:
    units_embed_max_pool = tf.reduce_max(inputs_units_embed, axis=1)
    units_embed_mean_pool = tf.reduce_mean(inputs_units_embed, axis=1)
    m_units_embed = tf.concat(
      [units_embed_max_pool, units_embed_mean_pool, inputs_units_sel_embed],
      axis=-1)
    m_units_embed = tfc_layers.fully_connected(m_units_embed, nc.enc_dim)

    inte_embed = tf.concat(
      [m_units_embed, inputs_spa_vec_embed, inputs_vec_embed,
       inputs_last_actions_embed], axis=-1)
    return lutils.collect_named_outputs(outputs_collections, sc.name,
                                        inte_embed)


@add_arg_scope
def _integrate_embed_v2(inputs_u_embed,
                        inputs_spa_embed,
                        inputs_vec_embed,
                        inputs_last_actions_embed,
                        nc,
                        outputs_collections=None):
  with tf.variable_scope('inte_embed_v2') as sc:
    # (bs, 600, dim_u)
    u_embed = tf.reduce_mean(inputs_u_embed, axis=1)
    # (bs, dim_u)
    u_embed = tfc_layers.fully_connected(u_embed, nc.enc_dim)
    # (bs, enc_dim)

    # (bs, H, W, dim_s)
    spa_embed = tf.reduce_mean(inputs_spa_embed, axis=[1, 2])
    # (bs, dim_s)

    inte_embed = tf.concat(
      [u_embed, spa_embed, inputs_vec_embed, inputs_last_actions_embed],
      axis=-1
    )
    return lutils.collect_named_outputs(outputs_collections, sc.name,
                                        inte_embed)


def pos_encode(seq_len, d_model, base_wavelen=None):
  """Cosine-sine position encoding

  Args:
    seq_len: int, sequence length
    d_model: int, the dimension, must be (d_model%2 == 0)
    base_wavelen: float, base wave length for the cosine-sine encoding

  Returns:
    A constant Tensor in shape (seq_len, d_model), representing the position
     encoding.
  """
  pos = np.arange(0, seq_len)
  pos = pos.reshape(seq_len, 1)  # (seq_len, 1)
  # a frequency related term as the divisor
  assert d_model % 2 == 0, 'd_model must be even.'
  base_wavelen = base_wavelen or 10000.0
  div_term = np.power(base_wavelen, -np.arange(0, d_model, 2) / d_model)
  div_term = div_term.reshape(1, d_model // 2)  # (1, d_model/2)
  # the concrete postion encoding in desired shape
  pe = np.zeros(shape=(seq_len, d_model), dtype=np.float32)
  pe[:, 0::2] = np.sin(pos * div_term)
  pe[:, 1::2] = np.cos(pos * div_term)
  return tf.constant(pe)


@add_arg_scope
def _zstat_connections(inputs_zstat_bo,
                       inputs_zstat_bo_coor,
                       inputs_zstat_bu,
                       nc, use_ln=False):
  z_bo = tfc_layers.fully_connected(inputs_zstat_bo, 32, scope='bo_fc')
  z_bo_coor = tfc_layers.fully_connected(inputs_zstat_bo_coor, 32,
                                         scope='boc_fc')
  z_bu = inputs_zstat_bu

  z_bo = tf.concat([z_bo, z_bo_coor], axis=-1)
  z_bo_embed = tfc_layers.fully_connected(z_bo, 32, scope='boe')

  z_mask = tf.zeros_like(z_bo_embed)
  hs = tf.zeros([z_bo.shape[0].value, z_bo.shape[1].value, 64])
  z_bo_b2s = tf.unstack(z_bo_embed, axis=1)
  z_mask = tf.unstack(z_mask, axis=1)
  with tf.variable_scope("z_enc"):
    z_bo_lstm_embeds, _ = tp_layers.lstm(inputs_x_seq=z_bo_b2s,
                                         inputs_terminal_mask_seq=z_mask,
                                         inputs_state=hs[:, 0, :], nh=32,
                                         scope='z_lstm',
                                         use_layer_norm=use_ln)
    if nc.zstat_embed_version == 'v1':
      z_bo_lstm_embed = z_bo_lstm_embeds[-1]  # only use the last lstm output
    elif nc.zstat_embed_version == 'v2':
      idx = tf.cast(tf.reduce_sum(inputs_zstat_bo, [1, 2]), tf.int32)
      idxs = tf.stack([tf.maximum(idx-1, 0), tf.range(nc.batch_size)], axis=1)
      z_bo_lstm_embed = tf.gather_nd(z_bo_lstm_embeds, idxs)
    else:
      raise KeyError('Unknown zstat_embed_version in nc!')
    z_bu_embed = tfc_layers.fully_connected(z_bu, nc.enc_dim, scope='z_bu_fc1')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim,
                                            scope='z_bu_fc2')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, 32, scope='z_bu_fc3')

  return z_bo_lstm_embeds, z_bo_lstm_embed, z_bu_embed


@add_arg_scope
def _zstat_connections_v2(inputs_zstat_bo,
                          inputs_zstat_bo_coor,
                          inputs_zstat_bobt,
                          inputs_zstat_bobt_coor,
                          inputs_zstat_bu,
                          nc, use_ln=False):
  # simple inputs connects
  z_bo = tfc_layers.fully_connected(inputs_zstat_bo, 32, scope='bo_fc')
  z_bo_coor = tfc_layers.fully_connected(inputs_zstat_bo_coor, 32,
                                         scope='boc_fc')
  z_bobt = tfc_layers.fully_connected(inputs_zstat_bobt, 32, scope='bobt_fc')
  z_bobt_coor = tfc_layers.fully_connected(inputs_zstat_bobt_coor, 32,
                                         scope='bocbt_fc')
  z_bu = inputs_zstat_bu

  # simple embeddings
  z_bo = tf.concat([z_bo, z_bo_coor], axis=-1)
  z_bo_embed = tfc_layers.fully_connected(z_bo, 32, scope='boe')
  z_bobt = tf.concat([z_bobt, z_bobt_coor], axis=-1)
  z_bobt_embed = tfc_layers.fully_connected(z_bobt, 32, scope='bobte')

  # bo lstm use
  z_bo_mask = tf.zeros_like(z_bo_embed)
  bo_hs = tf.zeros([z_bo.shape[0].value, z_bo.shape[1].value, 64])
  z_bo_b2s = tf.unstack(z_bo_embed, axis=1)
  z_bo_mask = tf.unstack(z_bo_mask, axis=1)

  # bobt lstm use
  z_bobt_mask = tf.zeros_like(z_bobt_embed)
  bobt_hs = tf.zeros([z_bobt.shape[0].value, z_bobt.shape[1].value, 64])
  z_bobt_b2s = tf.unstack(z_bobt_embed, axis=1)
  z_bobt_mask = tf.unstack(z_bobt_mask, axis=1)

  # lstm part
  with tf.variable_scope("z_enc"):
    z_bo_lstm_embeds, _ = tp_layers.lstm(inputs_x_seq=z_bo_b2s,
                                         inputs_terminal_mask_seq=z_bo_mask,
                                         inputs_state=bo_hs[:, 0, :], nh=32,
                                         scope='z_lstm_bo',
                                         use_layer_norm=use_ln)

    z_bobt_lstm_embeds, _ = tp_layers.lstm(inputs_x_seq=z_bobt_b2s,
                                           inputs_terminal_mask_seq=z_bobt_mask,
                                           inputs_state=bobt_hs[:, 0, :], nh=32,
                                           scope='z_lstm_bobt',
                                           use_layer_norm=use_ln)

    idx = tf.cast(tf.reduce_sum(inputs_zstat_bo, [1, 2]), tf.int32)
    idxs = tf.stack([tf.maximum(idx-1, 0), tf.range(nc.batch_size)], axis=1)
    z_bo_lstm_embed = tf.gather_nd(z_bo_lstm_embeds, idxs)

    idx = tf.cast(tf.reduce_sum(inputs_zstat_bobt, [1, 2]), tf.int32)
    idxs = tf.stack([tf.maximum(idx-1, 0), tf.range(nc.batch_size)], axis=1)
    z_bobt_lstm_embed = tf.gather_nd(z_bobt_lstm_embeds, idxs)

    z_bu_embed = tfc_layers.fully_connected(z_bu, nc.enc_dim, scope='z_bu_fc1')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim, scope='z_bu_fc2')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, 32, scope='z_bu_fc3')

  return z_bo_lstm_embeds, z_bo_lstm_embed, z_bobt_lstm_embeds, z_bobt_lstm_embed, z_bu_embed


@add_arg_scope
def _zstat_connections_v3(inputs_zstat_bo,
                          inputs_zstat_bo_coor,
                          inputs_zstat_bobt,
                          inputs_zstat_bobt_coor,
                          inputs_zstat_bu,
                          nc):
  """
  :param inputs_zstat_bo: [bs, 50, 62]
  :param inputs_zstat_bo_coor: [bs, 50, 18]
  :param inputs_zstat_bobt: [bs, 20, 46]
  :param inputs_zstat_bobt_coor: [bs, 20, 18]
  :param inputs_zstat_bu: [bs, 80]
  :param nc:
  :param use_ln:
  :return:
  """
  # simple inputs connects
  z_bo = tfc_layers.fully_connected(inputs_zstat_bo, 32, scope='bo_fc')
  z_bo_coor = tfc_layers.fully_connected(inputs_zstat_bo_coor, 32,
                                         scope='boc_fc')
  z_bobt = tfc_layers.fully_connected(inputs_zstat_bobt, 32, scope='bobt_fc')
  z_bobt_coor = tfc_layers.fully_connected(inputs_zstat_bobt_coor, 32,
                                         scope='bocbt_fc')
  z_bu = inputs_zstat_bu

  # simple embeddings
  z_bo = tf.concat([z_bo, z_bo_coor], axis=-1)
  z_bo_embed = tfc_layers.fully_connected(z_bo, 32, scope='boe')
  z_bobt = tf.concat([z_bobt, z_bobt_coor], axis=-1)
  z_bobt_embed = tfc_layers.fully_connected(z_bobt, 32, scope='bobte')

  with tf.variable_scope("z_enc"):
    # transformer part
    # according to AStar: The first 20 constructed entities are converted to a 2D tensor
    # of size [20, num_entity_types], concatenated with indices and the binary encodings
    # (as in the Entity Encoder) of where entities were constructed (if applicable). The
    # concatenation is passed through a transformer similar to the one in the entity
    # encoder, but with keys, queries, and values of 8 and with a MLP hidden size of 32
    z_bo_mask = tf.cast(tf.reduce_sum(inputs_zstat_bo, axis=-1), dtype=tf.bool)
    z_bo_embed = tfc_layers.fully_connected(z_bo_embed, 16, activation_fn=None)  # per AStar
    z_bo_embed = tp_ops.mask_embed(z_bo_embed, z_bo_mask)
    _, z_bo_trans_embed = _transformer_block_v4(
      z_bo_embed, z_bo_mask, enc_dim=16, out_fc_dim=32, nc=nc, scope='z_trans_bo')  # per AStar

    # transformer part
    z_bobt_mask = tf.cast(tf.reduce_sum(inputs_zstat_bobt, axis=-1), dtype=tf.bool)
    z_bobt_embed = tfc_layers.fully_connected(z_bobt_embed, 16, activation_fn=None)  # per AStar
    z_bobt_embed = tp_ops.mask_embed(z_bobt_embed, z_bobt_mask)
    _, z_bobt_trans_embed = _transformer_block_v4(
      z_bobt_embed, z_bobt_mask, enc_dim=16, out_fc_dim=32, nc=nc, scope='z_trans_bobt')

    z_bu_embed = tfc_layers.fully_connected(z_bu, nc.enc_dim, scope='z_bu_fc1')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim, scope='z_bu_fc2')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, 32, scope='z_bu_fc3')

  return z_bo_trans_embed, z_bobt_trans_embed, z_bu_embed


@add_arg_scope
def _zstat_connections_v3d1(inputs_zstat_bo,
                            inputs_zstat_bo_coor,
                            inputs_zstat_bu,
                            nc):
  """
  Args
    inputs_zstat_bo: [bs, T, 62]
    inputs_zstat_bo_coor: [bs, T, 18]
    inputs_zstat_bu: [bs, 80]
    nc:
    use_ln:

  Returns
    The bo(Build Order) embedding
    The bu(Build Unit) embedding
  """

  # simple inputs connects
  z_bo = tfc_layers.fully_connected(inputs_zstat_bo, 32, scope='bo_fc')
  z_bo_coor = tfc_layers.fully_connected(inputs_zstat_bo_coor, 32,
                                         scope='boc_fc')
  z_bu = inputs_zstat_bu

  # simple embeddings
  z_bo = tf.concat([z_bo, z_bo_coor], axis=-1)
  z_bo_embed = tfc_layers.fully_connected(z_bo, 32, scope='boe')
  # TODO(pengsun): concat with indices

  with tf.variable_scope("z_enc"):
    # transformer part
    # according to AStar: The first 20 constructed entities are converted to a 2D tensor
    # of size [20, num_entity_types], concatenated with indices and the binary encodings
    # (as in the Entity Encoder) of where entities were constructed (if applicable). The
    # concatenation is passed through a transformer similar to the one in the entity
    # encoder, but with keys, queries, and values of 8 and with a MLP hidden size of 32
    z_bo_mask = tf.cast(tf.reduce_sum(inputs_zstat_bo, axis=-1), dtype=tf.bool)
    z_bo_embed = tfc_layers.fully_connected(z_bo_embed, 16, activation_fn=None)  # per AStar
    z_bo_embed = tp_ops.mask_embed(z_bo_embed, z_bo_mask)
    _, z_bo_trans_embed = _transformer_block_v4(
      z_bo_embed, z_bo_mask, enc_dim=16, out_fc_dim=32, nc=nc,
      scope='z_trans_bo'
    )  # per AStar

    z_bu_embed = tfc_layers.fully_connected(z_bu, nc.enc_dim,
                                            scope='z_bu_fc1')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim,
                                            scope='z_bu_fc2')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, 32, scope='z_bu_fc3')

  return z_bo_trans_embed, z_bu_embed


@add_arg_scope
def _zstat_connections_v4(inputs_zstat_bo,
                          inputs_zstat_bo_coor,
                          inputs_zstat_bobt,
                          inputs_zstat_bobt_coor,
                          inputs_zstat_bu,
                          nc, use_ln=False):
  # simple inputs connects
  z_bo = tfc_layers.fully_connected(inputs_zstat_bo, 32, scope='bo_fc')
  z_bo_coor = tfc_layers.fully_connected(inputs_zstat_bo_coor, 32,
                                         scope='boc_fc')
  z_bobt = tfc_layers.fully_connected(inputs_zstat_bobt, 32, scope='bobt_fc')
  z_bobt_coor = tfc_layers.fully_connected(inputs_zstat_bobt_coor, 32,
                                         scope='bocbt_fc')

  # simple embeddings
  z_bo = tf.concat([z_bo, z_bo_coor], axis=-1)
  z_bo_embed = tfc_layers.fully_connected(z_bo, 32, scope='boe')
  z_bobt = tf.concat([z_bobt, z_bobt_coor], axis=-1)
  z_bobt_embed = tfc_layers.fully_connected(z_bobt, 32, scope='bobte')

  # bo lstm use
  z_bo_mask = tf.zeros_like(z_bo_embed)
  bo_hs = tf.zeros([z_bo.shape[0].value, z_bo.shape[1].value, 64])
  z_bo_b2s = tf.unstack(z_bo_embed, axis=1)
  z_bo_mask = tf.unstack(z_bo_mask, axis=1)

  # bobt lstm use
  z_bobt_mask = tf.zeros_like(z_bobt_embed)
  bobt_hs = tf.zeros([z_bobt.shape[0].value, z_bobt.shape[1].value, 64])
  z_bobt_b2s = tf.unstack(z_bobt_embed, axis=1)
  z_bobt_mask = tf.unstack(z_bobt_mask, axis=1)

  # lstm part
  with tf.variable_scope("z_enc"):
    z_bo_lstm_embeds, _ = tp_layers.lstm(inputs_x_seq=z_bo_b2s,
                                         inputs_terminal_mask_seq=z_bo_mask,
                                         inputs_state=bo_hs[:, 0, :], nh=32,
                                         scope='z_lstm_bo',
                                         use_layer_norm=use_ln)

    z_bobt_lstm_embeds, _ = tp_layers.lstm(inputs_x_seq=z_bobt_b2s,
                                           inputs_terminal_mask_seq=z_bobt_mask,
                                           inputs_state=bobt_hs[:, 0, :], nh=32,
                                           scope='z_lstm_bobt',
                                           use_layer_norm=use_ln)

    idx = tf.cast(tf.reduce_sum(inputs_zstat_bo, [1, 2]), tf.int32)
    idxs = tf.stack([tf.maximum(idx-1, 0), tf.range(nc.batch_size)], axis=1)
    z_bo_lstm_embed = tf.gather_nd(z_bo_lstm_embeds, idxs)

    idx = tf.cast(tf.reduce_sum(inputs_zstat_bobt, [1, 2]), tf.int32)
    idxs = tf.stack([tf.maximum(idx-1, 0), tf.range(nc.batch_size)], axis=1)
    z_bobt_lstm_embed = tf.gather_nd(z_bobt_lstm_embeds, idxs)

    # cumulative stats: buildings/units, effects, upgrades
    z_bu = tf.cast(tf.greater(inputs_zstat_bu, 0), tf.float32)
    z_bu_splits = tf.split(z_bu, nc.uc_split_indices, axis=-1)
    z_bu_bu_embed = tfc_layers.fully_connected(z_bu_splits[0], 32, activation_fn=tf.nn.relu)
    z_bu_eff_embed = tfc_layers.fully_connected(z_bu_splits[1], 32, activation_fn=tf.nn.relu)
    z_bu_up_embed = tfc_layers.fully_connected(z_bu_splits[2], 32, activation_fn=tf.nn.relu)
    z_bu_embed = tf.concat([z_bu_bu_embed, z_bu_eff_embed, z_bu_up_embed], axis=-1)

    # z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim, scope='z_bu_fc1')
    # z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim, scope='z_bu_fc2')
    # z_bu_embed = tfc_layers.fully_connected(z_bu_embed, 32, scope='z_bu_fc3')

    # the lstm outputs are not passed through relu
    z_bo_lstm_embed = tfc_layers.fully_connected(z_bo_lstm_embed, 32, activation_fn=tf.nn.relu)
    z_bobt_lstm_embed = tfc_layers.fully_connected(z_bobt_lstm_embed, 32, activation_fn=tf.nn.relu)

  return z_bo_lstm_embed, z_bobt_lstm_embed, z_bu_embed


@add_arg_scope
def _zstat_connections_v5(inputs_zstat_bo,
                          inputs_zstat_bo_coor,
                          inputs_zstat_bu,
                          nc, use_ln=False):
  """ Same as _zstat_connections_v2, except that bobt is removed """
  # simple inputs connects
  z_bo = tfc_layers.fully_connected(inputs_zstat_bo, 32, scope='bo_fc')
  z_bo_coor = tfc_layers.fully_connected(inputs_zstat_bo_coor, 32,
                                         scope='boc_fc')
  z_bu = inputs_zstat_bu

  # simple embeddings
  z_bo = tf.concat([z_bo, z_bo_coor], axis=-1)
  z_bo_embed = tfc_layers.fully_connected(z_bo, 32, scope='boe')

  # bo lstm use
  z_bo_mask = tf.zeros_like(z_bo_embed)
  bo_hs = tf.zeros([z_bo.shape[0].value, z_bo.shape[1].value, 64])
  z_bo_b2s = tf.unstack(z_bo_embed, axis=1)
  z_bo_mask = tf.unstack(z_bo_mask, axis=1)

  # lstm part
  with tf.variable_scope("z_enc"):
    z_bo_lstm_embeds, _ = tp_layers.lstm(inputs_x_seq=z_bo_b2s,
                                         inputs_terminal_mask_seq=z_bo_mask,
                                         inputs_state=bo_hs[:, 0, :], nh=32,
                                         scope='z_lstm_bo',
                                         use_layer_norm=use_ln)

    idx = tf.cast(tf.reduce_sum(inputs_zstat_bo, [1, 2]), tf.int32)
    idxs = tf.stack([tf.maximum(idx-1, 0), tf.range(nc.batch_size)], axis=1)
    z_bo_lstm_embed = tf.gather_nd(z_bo_lstm_embeds, idxs)

    z_bu_embed = tfc_layers.fully_connected(z_bu, nc.enc_dim, scope='z_bu_fc1')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, nc.enc_dim, scope='z_bu_fc2')
    z_bu_embed = tfc_layers.fully_connected(z_bu_embed, 32, scope='z_bu_fc3')

  return z_bo_lstm_embeds, z_bo_lstm_embed, z_bu_embed


@add_arg_scope
def _zstat_bo_connections(inputs_zstat_bo,
                          inputs_zstat_bo_coor,
                          nc,
                          out_dim=32):
  """zstat build order connections.

  Use a transformer block to do the embedding like AStar.

  Args:
    inputs_zstat_bo: build order, (bs, T, dim_bo)
    inputs_zstat_bo_coor: the corresponding coordinates, (bs, T, dim_coord)
    nc: net_config
    out_dim: output dimension, int

  Returns:
    A tensor representing the embedding, (bs, out_dim)
  """
  # mask out those all-zero entries in inputs_ztat_bo, (bs, T)
  bo_mask = tf.cast(tf.reduce_sum(inputs_zstat_bo, axis=-1), dtype=tf.bool)

  # create the position encoding
  T = inputs_zstat_bo.shape.as_list()[1]
  pe = pos_encode(T, d_model=16, base_wavelen=nc.zstat_index_base_wavelen)
  # (T, d_model)
  pe = tf.tile(tf.expand_dims(pe, 0), multiples=[nc.batch_size, 1, 1])
  # (bs, T, d_model)

  # concat bo, coord and position encoding, (bs, T, summed_dim)
  bo_embed = tf.concat([inputs_zstat_bo, inputs_zstat_bo_coor, pe], axis=2)
  # pre-transformer layer, (bs, T, 8)
  bo_embed = tfc_layers.fully_connected(bo_embed, 8, scope='boe')
  _, bo_trans_embed = _transformer_block_v4(
    bo_embed,
    bo_mask,
    nc,
    n_blk=3,  # not specified in AStar paper/supp; use 3 as units embeddings
    enc_dim=8,  # must be the same as bo_embed due to the res-sum connection
    out_fc_dim=out_dim,
    scope='z_trans_bo'
  )
  return bo_trans_embed


@add_arg_scope
def _zstat_bu_connections(inputs_zstat_bu, nc, out_dim=64):
  """zstat build units connections.

  AStar splits inputs_zstat_bu into 3 sub-vectors, i.e., the units/buildings,
  effects and upgrades. Then a outdim 32 for each, which amounts to 96 in total.
  In our impl here, we don't do the splits. Instead, a single fc layer for all.
  Thus we recommend use a out_dim > 32 in lump.

  Args:
    inputs_zstat_bu: zstat build units, (bs, bu_dim)
    nc: net config
    out_dim: int, output dim for the whole inputs_zstat_bu vector

  Returns:
    A tensor reprenting the bu embedding, (bs, out_dim)
  """
  # Do the "preprocessed into a boolean vector of whether or not statistic is
  # present" as in AStar Supp. Due to the non-negativity of inputs_zstat_bu,
  # tf.sign is a Step/Heaviside function in effect.
  bu_embd = tf.sign(inputs_zstat_bu)
  bu_embed = tfc_layers.fully_connected(bu_embd, out_dim, scope='z_bue')
  return bu_embed


@add_arg_scope
def _zstat_bu_connections_v2(inputs_zstat_bu, nc, out_dim=64):
  """zstat build units connections, version 2

  AStar splits inputs_zstat_bu into 3 sub-vectors, i.e., the units/buildings,
  effects and upgrades. Then an out_dim (=32) fc for each, which amounts to
  3*out_dim (=96) in total.

  Args:
    inputs_zstat_bu: zstat build units, (bs, bu_dim)
    nc: net config
    out_dim: int, output dim for each inputs_zstat_bu sub-vector

  Returns:
    A tensor reprenting the bu embedding, (bs, 3*out_dim)
  """
  # Do the "preprocessed into a boolean vector of whether or not statistic is
  # present" as in AStar Supp. Due to the non-negativity of inputs_zstat_bu,
  # tf.sign is a Step/Heaviside function in effect.
  bu_embd = tf.sign(inputs_zstat_bu)
  # split int 3 sub-vectors by a quick-and-dirty hard coding.
  bu_embd_splits = tf.split(bu_embd, nc.uc_split_indices, axis=-1)
  # do the three sub-vectors embedding separately
  buildings_units_embed = tfc_layers.fully_connected(
    bu_embd_splits[0], out_dim, scope='z_buildings_units_e')
  effects_embed = tfc_layers.fully_connected(bu_embd_splits[1], out_dim,
                                             scope='z_effects_e')
  upgrades_embed = tfc_layers.fully_connected(bu_embd_splits[2], out_dim,
                                              scope='z_upgrades_e')
  return tf.concat([buildings_units_embed, effects_embed, upgrades_embed],
                   axis=-1)


@add_arg_scope
def _zstat_embed(inputs_obs, nc, outputs_collections=None):
  if nc.zstat_embed_version == 'v1':
    zstat_embed = _zstat_embed_block(
      inputs_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      inputs_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      inputs_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v2':
    zstat_embed = _zstat_embed_block_v2(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      imm_zstat_bo=inputs_obs['IMM_BUILD_ORDER'],
      imm_zstat_bo_coor=inputs_obs['IMM_BUILD_ORDER_COORD'],
      imm_zstat_bu=inputs_obs['IMM_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v3':
    # lstm embed
    zstat_embed = _zstat_embed_block_v3(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bobt=inputs_obs['Z_BUILD_ORDER_BT'],
      tar_zstat_bobt_coor=inputs_obs['Z_BUILD_ORDER_COORD_BT'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      imm_zstat_bo=inputs_obs['IMM_BUILD_ORDER'],
      imm_zstat_bo_coor=inputs_obs['IMM_BUILD_ORDER_COORD'],
      imm_zstat_bobt=inputs_obs['IMM_BUILD_ORDER_BT'],
      imm_zstat_bobt_coor=inputs_obs['IMM_BUILD_ORDER_COORD_BT'],
      imm_zstat_bu=inputs_obs['IMM_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v4':
    # trans embed
    zstat_embed = _zstat_embed_block_v4(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bobt=inputs_obs['Z_BUILD_ORDER_BT'],
      tar_zstat_bobt_coor=inputs_obs['Z_BUILD_ORDER_COORD_BT'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      imm_zstat_bo=inputs_obs['IMM_BUILD_ORDER'],
      imm_zstat_bo_coor=inputs_obs['IMM_BUILD_ORDER_COORD'],
      imm_zstat_bobt=inputs_obs['IMM_BUILD_ORDER_BT'],
      imm_zstat_bobt_coor=inputs_obs['IMM_BUILD_ORDER_COORD_BT'],
      imm_zstat_bu=inputs_obs['IMM_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v4d1':
    # trans embed, version 4.1
    zstat_embed = _zstat_embed_block_v4d1(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v5':
    # lstm embed but with binary unit count/cumulative stats as AStar;
    # shared connections for tar_z & imm_z, lstm output with
    # one more linear layer with relu
    zstat_embed = _zstat_embed_block_v5(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bobt=inputs_obs['Z_BUILD_ORDER_BT'],
      tar_zstat_bobt_coor=inputs_obs['Z_BUILD_ORDER_COORD_BT'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      imm_zstat_bo=inputs_obs['IMM_BUILD_ORDER'],
      imm_zstat_bo_coor=inputs_obs['IMM_BUILD_ORDER_COORD'],
      imm_zstat_bobt=inputs_obs['IMM_BUILD_ORDER_BT'],
      imm_zstat_bobt_coor=inputs_obs['IMM_BUILD_ORDER_COORD_BT'],
      imm_zstat_bu=inputs_obs['IMM_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v6':
    # same as v5 except that imm zstat is removed, suggested by qiaobo
    # this is reasonable since in IL, imm zstat goes exactly following
    # target zstat, while in RL imm zstat differs from target zstat.
    zstat_embed = _zstat_embed_block_v6(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bobt=inputs_obs['Z_BUILD_ORDER_BT'],
      tar_zstat_bobt_coor=inputs_obs['Z_BUILD_ORDER_COORD_BT'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  elif nc.zstat_embed_version == 'v7':
    # same as v3 except that imm zstat and bobt are removed
    zstat_embed = _zstat_embed_block_v7(
      tar_zstat_bo=inputs_obs['Z_BUILD_ORDER'],
      tar_zstat_bo_coor=inputs_obs['Z_BUILD_ORDER_COORD'],
      tar_zstat_bu=inputs_obs['Z_UNIT_COUNT'],
      nc=nc,
      outputs_collections=outputs_collections,
    )
  else:
    raise KeyError('Unknown zstat_embed_version in nc!')
  return zstat_embed


@add_arg_scope
def _zstat_embed_block(inputs_zstat_bo,
                       inputs_zstat_bo_coor,
                       inputs_zstat_bu,
                       nc,
                       outputs_collections=None):
  with tf.variable_scope('zstat_embed') as sc:
    _, z_bo_embed, z_bu_embed = _zstat_connections(
      inputs_zstat_bo, inputs_zstat_bo_coor, inputs_zstat_bu, nc)
    res = tf.concat([z_bo_embed, z_bu_embed], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _zstat_embed_block_v2(tar_zstat_bo,
                          tar_zstat_bo_coor,
                          tar_zstat_bu,
                          imm_zstat_bo,
                          imm_zstat_bo_coor,
                          imm_zstat_bu,
                          nc,
                          outputs_collections=None):

  with tf.variable_scope('zstat_embed') as sc:
    with tf.variable_scope('tar'):
      _, tar_z_bo_embed, tar_z_bu_embed = _zstat_connections(
        tar_zstat_bo, tar_zstat_bo_coor, tar_zstat_bu, nc)
      tar_res = tf.concat([tar_z_bo_embed, tar_z_bu_embed], axis=-1)
    with tf.variable_scope('imm'):
      _, imm_z_bo_embed, imm_z_bu_embed = _zstat_connections(
        imm_zstat_bo, imm_zstat_bo_coor, imm_zstat_bu, nc)
      imm_res = tf.concat([imm_z_bo_embed, imm_z_bu_embed], axis=-1)
    res = tf.concat([tar_res, imm_res], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _zstat_embed_block_v3(tar_zstat_bo,
                          tar_zstat_bo_coor,
                          tar_zstat_bobt,
                          tar_zstat_bobt_coor,
                          tar_zstat_bu,
                          imm_zstat_bo,
                          imm_zstat_bo_coor,
                          imm_zstat_bobt,
                          imm_zstat_bobt_coor,
                          imm_zstat_bu,
                          nc,
                          outputs_collections=None):

  with tf.variable_scope('zstat_embed') as sc:
    with tf.variable_scope('tar'):
      _, tar_z_bo_embed, _, tar_z_bobt_embed, tar_z_bu_embed = _zstat_connections_v2(
        tar_zstat_bo, tar_zstat_bo_coor, tar_zstat_bobt, tar_zstat_bobt_coor, tar_zstat_bu,
        nc, use_ln=True)
      tar_res = tf.concat([tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed], axis=-1)
    with tf.variable_scope('imm'):
      _, imm_z_bo_embed, _, imm_z_bobt_embed, imm_z_bu_embed = _zstat_connections_v2(
        imm_zstat_bo, imm_zstat_bo_coor, imm_zstat_bobt, imm_zstat_bobt_coor, imm_zstat_bu,
        nc, use_ln=True)
      imm_res = tf.concat([imm_z_bo_embed, imm_z_bobt_embed, imm_z_bu_embed], axis=-1)
    res = tf.concat([tar_res, imm_res], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _zstat_embed_block_v4(tar_zstat_bo,
                          tar_zstat_bo_coor,
                          tar_zstat_bobt,
                          tar_zstat_bobt_coor,
                          tar_zstat_bu,
                          imm_zstat_bo,
                          imm_zstat_bo_coor,
                          imm_zstat_bobt,
                          imm_zstat_bobt_coor,
                          imm_zstat_bu,
                          nc,
                          outputs_collections=None):
  """transformer embedings for build orders"""

  with tf.variable_scope('zstat_embed') as sc:
    with tf.variable_scope('tar'):
      tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed = _zstat_connections_v3(
        tar_zstat_bo, tar_zstat_bo_coor, tar_zstat_bobt, tar_zstat_bobt_coor, tar_zstat_bu, nc)
      tar_res = tf.concat([tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed], axis=-1)
    with tf.variable_scope('imm'):
      imm_z_bo_embed, imm_z_bobt_embed, imm_z_bu_embed = _zstat_connections_v3(
        imm_zstat_bo, imm_zstat_bo_coor, imm_zstat_bobt, imm_zstat_bobt_coor, imm_zstat_bu, nc)
      imm_res = tf.concat([imm_z_bo_embed, imm_z_bobt_embed, imm_z_bu_embed], axis=-1)
    res = tf.concat([tar_res, imm_res], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _zstat_embed_block_v4d1(tar_zstat_bo,
                            tar_zstat_bo_coor,
                            tar_zstat_bu,
                            nc,
                            outputs_collections=None):
  """transformer embeddings for build orders, version 4.1

  No imm_zstat embeddings; Improved bo, bu connections."""

  with tf.variable_scope('zstat_embed') as sc:
    with tf.variable_scope('tar'):
      tar_z_bo_embed = _zstat_bo_connections(
        tar_zstat_bo, tar_zstat_bo_coor, nc, out_dim=32
      )
      tar_z_bu_embed = _zstat_bu_connections_v2(tar_zstat_bu, nc, out_dim=32)
      # TODO(pengsun): should return bo (beginning_build_order) and bu
      #  (cumulative_statistics) separately
      tar_res = tf.concat([tar_z_bo_embed, tar_z_bu_embed], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        tar_res)


@add_arg_scope
def _zstat_embed_block_v5(tar_zstat_bo,
                          tar_zstat_bo_coor,
                          tar_zstat_bobt,
                          tar_zstat_bobt_coor,
                          tar_zstat_bu,
                          imm_zstat_bo,
                          imm_zstat_bo_coor,
                          imm_zstat_bobt,
                          imm_zstat_bobt_coor,
                          imm_zstat_bu,
                          nc,
                          outputs_collections=None):

  with tf.variable_scope('zstat_embed') as sc:
    # target zstat
    with tf.variable_scope('shared_conn'):
      tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed = _zstat_connections_v4(
        tar_zstat_bo, tar_zstat_bo_coor, tar_zstat_bobt, tar_zstat_bobt_coor, tar_zstat_bu,
        nc, use_ln=True)
      tar_res = tf.concat([tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed], axis=-1)
    # immediate zstat, shared parameter
    with tf.variable_scope('shared_conn', reuse=tf.AUTO_REUSE):
      imm_z_bo_embed, imm_z_bobt_embed, imm_z_bu_embed = _zstat_connections_v4(
        imm_zstat_bo, imm_zstat_bo_coor, imm_zstat_bobt, imm_zstat_bobt_coor, imm_zstat_bu,
        nc, use_ln=True)
      imm_res = tf.concat([imm_z_bo_embed, imm_z_bobt_embed, imm_z_bu_embed], axis=-1)
    res = tf.concat([tar_res, imm_res], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _zstat_embed_block_v6(tar_zstat_bo,
                          tar_zstat_bo_coor,
                          tar_zstat_bobt,
                          tar_zstat_bobt_coor,
                          tar_zstat_bu,
                          nc,
                          outputs_collections=None):
  """
  Same as v5 except that imm zstat is removed, suggested by qiaobo
  :param tar_zstat_bo:
  :param tar_zstat_bo_coor:
  :param tar_zstat_bobt:
  :param tar_zstat_bobt_coor:
  :param tar_zstat_bu:
  :param nc:
  :param outputs_collections:
  :return:
  """
  with tf.variable_scope('zstat_embed') as sc:
    # target zstat
    tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed = _zstat_connections_v4(
      tar_zstat_bo, tar_zstat_bo_coor, tar_zstat_bobt, tar_zstat_bobt_coor, tar_zstat_bu,
      nc, use_ln=True)
    res = tf.concat([tar_z_bo_embed, tar_z_bobt_embed, tar_z_bu_embed], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _zstat_embed_block_v7(tar_zstat_bo,
                          tar_zstat_bo_coor,
                          tar_zstat_bu,
                          nc,
                          outputs_collections=None):
  """
  Same as v3 except that imm zstat and bobt are removed
  :param tar_zstat_bo:
  :param tar_zstat_bo_coor:
  :param tar_zstat_bu:
  :param nc:
  :param outputs_collections:
  :return:
  """
  with tf.variable_scope('zstat_embed') as sc:
    # target zstat
    _, tar_z_bo_embed, tar_z_bu_embed = _zstat_connections_v5(
      tar_zstat_bo, tar_zstat_bo_coor, tar_zstat_bu, nc, use_ln=True)
    res = tf.concat([tar_z_bo_embed, tar_z_bu_embed], axis=-1)
    return lutils.collect_named_outputs(outputs_collections,
                                        sc.original_name_scope + 'out',
                                        res)


@add_arg_scope
def _pre_discrete_action_block(inputs, enc_dim, n_blk=4,
                               outputs_collections=None):
  o = tp_layers.dense_sum_blocks(inputs, n=n_blk, enc_dim=enc_dim)
  return lutils.collect_named_outputs(outputs_collections,
                                      'pre_discrete_action', o)


@add_arg_scope
def _pre_discrete_action_res_block(inputs, enc_dim, n_blk=2, n_skip=2,
                                   outputs_collections=None):
  tmp = tfc_layers.fully_connected(inputs, enc_dim,
                                   normalizer_fn=None,
                                   activation_fn=None)
  o = tp_layers.res_sum_blocks_v2(tmp, n_blk=n_blk, n_skip=n_skip,
                                  enc_dim=enc_dim, layer_norm=True)
  o = tp_layers.ln(o, activation_fn=tf.nn.relu)
  return lutils.collect_named_outputs(outputs_collections,
                                      'pre_discrete_action_res', o)


@add_arg_scope
def _pre_discrete_action_fc_block(inputs, n_actions, enc_dim, n_blk=2,
                                  outputs_collections=None):
  for i in range(n_blk):
    inputs = tfc_layers.fully_connected(inputs, enc_dim)
  o = tfc_layers.fully_connected(inputs, n_actions)
  return lutils.collect_named_outputs(outputs_collections,
                                      'pre_noop_action_block', o)


@add_arg_scope
def _pre_action_temporal_smoothing_block(inputs, n_actions, enc_dim,
                                         outputs_collections=None):
  """An ad-hoc temporal smoothing choice from new_actions4.mnet"""
  head_h = tfc_layers.fully_connected(inputs, n_actions)
  head_h = tp_layers.dense_sum_conv_blocks(
    inputs=tf.expand_dims(head_h, axis=-1), n=3, ch_dim=enc_dim, k_size=4,
    mode='1d')
  head_h = lutils.collect_named_outputs(outputs_collections,
                                        'pre_action_temporal_smoothing', head_h)
  # (bs, n_actions, enc_dim)
  head_logits = tfc_layers.conv1d(head_h, 1, 1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='logits')
  # (bs, n_actions, 1)
  #head_logits = tf.unstack(head_logits, axis=-1)[0]
  head_logits = tf.squeeze(head_logits, axis=2)
  # (bs, n_actions)
  return head_logits


@add_arg_scope
def _pre_action_res_temporal_smoothing_block(inputs, n_actions, enc_dim,
                                             n_blk=2, n_skip=2,
                                             outputs_collections=None):
  """An ad-hoc temporal smoothing choice from new_actions4.mnet, with res"""
  head_h = tfc_layers.fully_connected(inputs, n_actions)
  # (bs, n_actions)
  head_h = tp_layers.res_sum_conv_blocks(
    inputs=tf.expand_dims(head_h, axis=-1), n_blk=n_blk, n_skip=n_skip,
    ch_dim=enc_dim, k_size=4, mode='1d')
  head_h = lutils.collect_named_outputs(outputs_collections,
                                        'pre_action_res_temporal_smoothing',
                                        head_h)
  # (bs, n_actions, enc_dim)
  head_logits = tfc_layers.conv1d(head_h, 1, 1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='logits')
  # (bs, n_actions, 1)
  #head_logits = tf.unstack(head_logits, axis=-1)[0]
  head_logits = tf.squeeze(head_logits, axis=2)
  # (bs, n_actions)
  return head_logits


@add_arg_scope
def _pre_action_res_bottleneck_temporal_smoothing_block(
    inputs,
    n_actions,
    enc_dim,
    n_blk=2,
    n_skip=2,
    outputs_collections=None):
  """An ad-hoc temporal smoothing choice from new_actions4.mnet, with res
   bottleneck"""
  head_h = tfc_layers.fully_connected(inputs, n_actions)
  # (bs, n_actions)
  head_h = tf.expand_dims(head_h, axis=-1)
  # (bs, n_actions, 1)
  head_h = tfc_layers.conv1d(head_h, enc_dim, 1, activation_fn=None,
                             normalizer_fn=None)
  # (bs, n_actions, enc_dim)
  head_h = tp_layers.res_sum_bottleneck_blocks_v2(
    inputs=head_h,
    n_blk=n_blk,
    n_skip=n_skip,
    ch_dim=enc_dim,
    bottleneck_ch_dim=int(enc_dim/4),
    k_size=4,
    mode='1d',
    layer_norm_type='inst_ln',
  )
  head_h = tp_layers.ln(head_h, activation_fn=tf.nn.relu)
  head_h = lutils.collect_named_outputs(outputs_collections,
                                        'pre_action_res_temporal_smoothing',
                                        head_h)
  # (bs, n_actions, enc_dim)
  head_logits = tfc_layers.conv1d(head_h, 1, 1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='logits')
  # (bs, n_actions, 1)
  head_logits = tf.squeeze(head_logits, axis=2)
  # (bs, n_actions)
  return head_logits


@add_arg_scope
def _pre_ptr_action_block(inputs, enc_dim, outputs_collections=None):
  o = tp_layers.dense_sum_blocks(inputs=inputs, n=4, enc_dim=enc_dim)
  o = lutils.collect_named_outputs(outputs_collections, 'pre_ptr_action', o)
  o = tf.expand_dims(o, axis=1)
  return o


@add_arg_scope
def _pre_ptr_action_res_block(inputs, enc_dim, n_blk=2, n_skip=2,
                              outputs_collections=None):
  tmp = tfc_layers.fully_connected(inputs, enc_dim,
                                   normalizer_fn=None,
                                   activation_fn=None)
  o = tp_layers.res_sum_blocks_v2(inputs=tmp, n_blk=n_blk, n_skip=n_skip,
                                  enc_dim=enc_dim, layer_norm=True)
  o = tp_layers.ln(o, activation_fn=tf.nn.relu)
  lutils.collect_named_outputs(outputs_collections, 'pre_ptr_action_res', o)
  o = tf.expand_dims(o, axis=1)
  return o


@add_arg_scope
def _pre_multinomial_action_block(inputs_reg_embed, inputs_entity_embed,
                                  enc_dim, outputs_collections=None):
  query_h = tp_layers.dense_sum_blocks(inputs=inputs_reg_embed, n=4,
                                       enc_dim=enc_dim,
                                       scope='q_densesum_blk')
  query_h = tf.expand_dims(query_h, axis=1)
  query_h = tf.tile(query_h, multiples=[
    1, tf.shape(inputs_entity_embed)[1], 1])
  head_h = tf.concat([inputs_entity_embed, query_h], axis=-1)
  head_h = tp_layers.dense_sum_blocks(inputs=head_h, n=4, enc_dim=enc_dim,
                                      scope='eq_densesum_blk')
  return lutils.collect_named_outputs(outputs_collections,
                                      'pre_multinomial_action', head_h)


@add_arg_scope
def _pre_multinomial_action_res_block(inputs_reg_embed,
                                      inputs_entity_embed, enc_dim,
                                      n_blk_before=2, n_skip_before=2,
                                      n_blk_after=2, n_skip_after=2,
                                      outputs_collections=None):
  tmp = tfc_layers.fully_connected(inputs_reg_embed, enc_dim,
                                   normalizer_fn=None,
                                   activation_fn=None)
  query_h = tp_layers.res_sum_blocks_v2(inputs=tmp,
                                        n_blk=n_blk_before,
                                        n_skip=n_skip_before,
                                        enc_dim=enc_dim,
                                        layer_norm=True,
                                        scope='q_ressum_blk')
  query_h = tp_layers.ln(query_h, activation_fn=tf.nn.relu)
  query_h = tf.expand_dims(query_h, axis=1)
  query_h = tf.tile(query_h, multiples=[
    1, tf.shape(inputs_entity_embed)[1], 1])

  head_h = tf.concat([inputs_entity_embed, query_h], axis=-1)
  head_h = tfc_layers.fully_connected(head_h, enc_dim,
                                      normalizer_fn=None,
                                      activation_fn=None)
  head_h = tp_layers.res_sum_blocks_v2(inputs=head_h,
                                       n_blk=n_blk_after,
                                       n_skip=n_skip_after,
                                       enc_dim=enc_dim,
                                       layer_norm=True,
                                       scope='eq_ressum_blk')
  head_h = tp_layers.ln(head_h, activation_fn=tf.nn.relu)
  return lutils.collect_named_outputs(outputs_collections,
                                      'pre_multinomial_action_res', head_h)


@add_arg_scope
def _pre_loc_action_block(inputs, ch_dim, outputs_collections=None):
  embed = tp_layers.dense_sum_conv_blocks(inputs=inputs, n=4, ch_dim=ch_dim,
                                          k_size=1,
                                          scope='densesum_2d_k_1')
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action',
                                      embed)


@add_arg_scope
def _pre_loc_action_res_block(inputs, ch_dim, outputs_collections=None):
  # Note: intentional kernel size = 3
  embed = tp_layers.res_sum_conv_blocks(inputs=inputs, n_blk=2, n_skip=2,
                                        ch_dim=ch_dim, k_size=3,
                                        scope='ressum_conv2d_blks')
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action_res',
                                      embed)


@add_arg_scope
def _pre_loc_action_astar_like_block_v1(reg_embed, spa_embed, ch_dim=128,
                                        n_blk=1, n_skip=4, layer_norm=True,
                                        outputs_collections=None):
  reg_embed = tf.reshape(reg_embed, shape=[-1, 16, 16, 4])
  concatenated_spa_embed = tf.concat([reg_embed, spa_embed], axis=-1)
  x = tfc_layers.conv2d(tf.nn.relu(concatenated_spa_embed), ch_dim, [1, 1])
  x = tp_layers.res_sum_conv_blocks(x, n_blk=n_blk, n_skip=n_skip,
                                    ch_dim=ch_dim,
                                    k_size=3, layer_norm=layer_norm)
  x = tfc_layers.conv2d_transpose(x, ch_dim, [3, 3], stride=2)
  x = tfc_layers.conv2d_transpose(x, int(ch_dim/2), [3, 3], stride=2)
  x = tfc_layers.conv2d_transpose(x, int(ch_dim/8), [3, 3], stride=2)
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action', x)


@add_arg_scope
def _pre_loc_action_res_bottleneck_block(inputs, ch_dim,
                                         outputs_collections=None):
  embed = tfc_layers.conv2d(inputs, ch_dim, [1, 1],
                            normalizer_fn=None,
                            activation_fn=None)
  # Note: intentional kernel size = 3; use bottleneck **v2** impl
  embed = tp_layers.res_sum_bottleneck_blocks_v2(
    inputs=embed, n_blk=2, n_skip=2, ch_dim=ch_dim,
    bottleneck_ch_dim=int(ch_dim/4), k_size=3,
    layer_norm_type='inst_ln', scope='ressum_conv2d_blks')
  embed = tp_layers.inst_ln(embed, activation_fn=tf.nn.relu)
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action_res',
                                      embed)


@add_arg_scope
def _pre_loc_action_res_bottleneck_block_v2(inputs, ch_dim,
                                            outputs_collections=None):
  embed = tfc_layers.conv2d(inputs, ch_dim, [1, 1],
                            normalizer_fn=None,
                            activation_fn=None)
  # Note: intentional kernel size = 3; use bottleneck **v3** impl
  embed = tp_layers.res_sum_bottleneck_blocks_v3(
    inputs=embed, n_blk=2, n_skip=2, ch_dim=ch_dim,
    bottleneck_ch_dim=int(ch_dim/4), k_size=3,
    layer_norm_type='inst_ln', scope='ressum_conv2d_blks')
  embed = tp_layers.inst_ln(embed, activation_fn=tf.nn.relu)
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action_res',
                                      embed)


@add_arg_scope
def _pre_loc_action_res_bottleneck_block_v3(inputs, ch_dim,
                                            outputs_collections=None):
  embed = tfc_layers.conv2d(inputs, ch_dim, [1, 1],
                            normalizer_fn=None,
                            activation_fn=None)
  # Note: intentional kernel size = 3; use bottleneck **v2** impl;
  # Note: intentional layer_norm_type='ln' to tackle the expanded feature
  # channel
  embed = tp_layers.res_sum_bottleneck_blocks_v2(
    inputs=embed, n_blk=2, n_skip=2, ch_dim=ch_dim,
    bottleneck_ch_dim=int(ch_dim/4), k_size=3,
    layer_norm_type='ln', scope='ressum_conv2d_blks')
  embed = tp_layers.ln(embed, begin_norm_axis=1, activation_fn=tf.nn.relu)
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action_res',
                                      embed)


@add_arg_scope
def _pre_loc_action_res_bottleneck_block_vold(inputs, ch_dim,
                                              outputs_collections=None):
  # Note: intentional kernel size = 3
  embed = tp_layers.res_sum_bottleneck_blocks(
    inputs=inputs, n_blk=2, n_skip=2, ch_dim=ch_dim,
    bottleneck_ch_dim=int(ch_dim/4), k_size=3,
    layer_norm=True, layer_norm_type='inst_ln',
    scope='ressum_conv2d_blks')
  return lutils.collect_named_outputs(outputs_collections, 'pre_loc_action_res',
                                      embed)


@add_arg_scope
def _pos_spa_embed_block(inputs_embed, inputs_spa_embed, ch_dim,
                         outputs_collections=None):
  m_embed = tp_layers.dense_sum_blocks(inputs=inputs_embed, n=2,
                                       enc_dim=ch_dim)
  m_embed = tf.expand_dims(tf.expand_dims(m_embed, axis=1), axis=1)
  m_embed = tf.tile(m_embed, multiples=[
    1, tf.shape(inputs_spa_embed)[1], tf.shape(inputs_spa_embed)[2], 1])
  m_spa_embed = tf.concat([inputs_spa_embed, m_embed], axis=-1)
  return lutils.collect_named_outputs(outputs_collections, 'pos_spa_embed',
                                      m_spa_embed)


@add_arg_scope
def _pos_spa_embed_res_block(inputs_embed, inputs_spa_embed, ch_dim, n_blk=1,
                             n_skip=2, outputs_collections=None):
  tmp = tfc_layers.fully_connected(inputs_embed, ch_dim,
                                   normalizer_fn=None,
                                   activation_fn=None)
  m_embed = tp_layers.res_sum_blocks_v2(inputs=tmp, n_blk=n_blk,
                                        n_skip=n_skip, enc_dim=ch_dim,
                                        layer_norm=True)
  m_embed = tp_layers.ln(m_embed, activation_fn=tf.nn.relu)
  m_embed = tf.expand_dims(tf.expand_dims(m_embed, axis=1), axis=1)
  m_embed = tf.tile(m_embed, multiples=[
    1, tf.shape(inputs_spa_embed)[1], tf.shape(inputs_spa_embed)[2], 1])
  m_spa_embed = tf.concat([inputs_spa_embed, m_embed], axis=-1)
  return lutils.collect_named_outputs(outputs_collections, 'pos_spa_embed_res',
                                      m_spa_embed)


@add_arg_scope
def _pos_spa_embed_res_block_v2(inputs_embed, inputs_spa_embed, ch_dim, n_blk=1,
                                n_skip=2, outputs_collections=None):
  tmp = tfc_layers.fully_connected(inputs_embed, ch_dim,
                                   normalizer_fn=None,
                                   activation_fn=None)
  m_embed = tp_layers.res_sum_blocks_v2(inputs=tmp, n_blk=n_blk,
                                        n_skip=n_skip, enc_dim=ch_dim,
                                        layer_norm=True)
  m_embed = tp_layers.ln(m_embed, activation_fn=tf.nn.relu)
  lutils.collect_named_outputs(outputs_collections,
                               'pos_spa_embed_res_v2_m_embed',
                               m_embed)
  m_embed = tf.expand_dims(tf.expand_dims(m_embed, axis=1), axis=1)
  m_embed = tf.tile(m_embed, multiples=[
    1, tf.shape(inputs_spa_embed)[1], tf.shape(inputs_spa_embed)[2], 1])
  lutils.collect_named_outputs(outputs_collections,
                               'pos_spa_embed_res_v2_m_expand_embed',
                               m_embed)

  # NOTE: presumed NHWC data format
  tmp_out_dim = inputs_spa_embed.shape.as_list()[3]
  spa_embed = tfc_layers.conv2d(inputs_spa_embed, tmp_out_dim, [1, 1],
                                normalizer_fn=tp_layers.inst_ln,
                                activation_fn=tf.nn.relu)
  lutils.collect_named_outputs(outputs_collections,
                               'pos_spa_embed_res_v2_spa_embed',
                               spa_embed)

  m_spa_embed = tf.concat([spa_embed, m_embed], axis=-1)
  return lutils.collect_named_outputs(outputs_collections,
                                      'pos_spa_embed_res_v2_out',
                                      m_spa_embed)


@add_arg_scope
def _pos_spa_embed_res_block_vold(inputs_embed, inputs_spa_embed, ch_dim, n_blk=1,
                                  n_skip=2, outputs_collections=None):
  m_embed = tp_layers.res_sum_blocks(inputs=inputs_embed, n_blk=n_blk,
                                     n_skip=n_skip, enc_dim=ch_dim,
                                     relu_input=True, layer_norm=True)
  lutils.collect_named_outputs(outputs_collections,
                               'pos_spa_embed_res_vold_m_embed',
                               m_embed)
  m_embed = tf.expand_dims(tf.expand_dims(m_embed, axis=1), axis=1)
  m_embed = tf.tile(m_embed, multiples=[
    1, tf.shape(inputs_spa_embed)[1], tf.shape(inputs_spa_embed)[2], 1])
  lutils.collect_named_outputs(outputs_collections,
                               'pos_spa_embed_res_vold_m_expand_embed',
                               m_embed)

  m_spa_embed = tf.concat([inputs_spa_embed, m_embed], axis=-1)
  return lutils.collect_named_outputs(outputs_collections,
                                      'pos_spa_embed_res_vold_out',
                                      m_spa_embed)


@add_arg_scope
def _v4_value_block(int_embed, n_v, outputs_collections=None):
  vf = tfc_layers.fully_connected(int_embed, 128)
  vf = tfc_layers.fully_connected(vf, 128)
  vf = tfc_layers.fully_connected(vf, n_v,
                                  activation_fn=None,
                                  normalizer_fn=None)
  return lutils.collect_named_outputs(outputs_collections, 'vf', vf)


@add_arg_scope
def _astar_v_oppo_vec_embed_block(inputs, enc_dim, use_score,
                                  outputs_collections=None):
  with tf.variable_scope("oppo_vec_enc") as sc:
    x_stat = tfc_layers.fully_connected(inputs['OPPO_X_VEC_PLAYER_STAT'], 64,
                                        scope='x_stat')
    x_upgrad = tfc_layers.fully_connected(inputs['OPPO_X_VEC_UPGRADE'], 128,
                                          scope='x_upgrad')
    x_game_prog = tfc_layers.fully_connected(inputs['OPPO_X_VEC_GAME_PROG'], 64,
                                             scope='x_game_prog')
    x_s_ucnt = tfc_layers.fully_connected(inputs['OPPO_X_VEC_S_UNIT_CNT'], 128,
                                          scope='x_s_ucnt')
    x_e_ucnt = tfc_layers.fully_connected(inputs['OPPO_X_VEC_E_UNIT_CNT'], 128,
                                          scope='x_e_ucnt')
    x_af_ucnt = tfc_layers.fully_connected(inputs['OPPO_X_VEC_AF_UNIT_CNT'], 64,
                                           scope='x_af_ucnt')
    x_vec_prog = tfc_layers.fully_connected(inputs['OPPO_X_VEC_PROG'], 64,
                                            scope='x_vec_prog')
    features = [x_stat, x_upgrad, x_game_prog, x_s_ucnt,
                           x_e_ucnt, x_af_ucnt, x_vec_prog]
    if use_score:
      x_vec_score = tfc_layers.fully_connected(inputs['OPPO_X_VEC_SCORE'], 64,
                                               scope='x_vec_score')
      features.append(x_vec_score)
    vec_embed = tf.concat(features, axis=-1)
    vec_embed = tfc_layers.fully_connected(vec_embed, 512, scope='x_fc1')
    vec_embed = tfc_layers.fully_connected(vec_embed, 2 * enc_dim,
                                           scope='x_fc2')
    return lutils.collect_named_outputs(outputs_collections, sc.name, vec_embed)

def baseline_embed(features):
  embeds = []
  for feat in features:
    embeds.append(tfc_layers.fully_connected(feat, 32))
  if len(embeds) >= 2:
    return [tf.concat(embeds, axis=-1)]
  elif len(embeds) == 1:
    return embeds
  else:
    return []

def astar_atan_fn(features):
  return (2.0 / 3.1415926) * tf.atan(3.1415926 / 2.0 * features)

def baseline_connects(features, addons, act_fn=None, n_blk=8, n_skip=2, enc_dim=64):
  com = baseline_embed(features)
  com = tf.concat(com+addons, axis=-1)
  com_embed = tp_layers.res_sum_blocks(inputs=com, n_blk=n_blk, n_skip=n_skip,
                                       enc_dim=enc_dim, layer_norm=True)
  return tfc_layers.fully_connected(com_embed, 1,
                                    activation_fn=act_fn,
                                    normalizer_fn=None)

@add_arg_scope
def _astar_like_value_block(int_embed,
                            z_bo, z_boc, z_bu,
                            c_bo, c_boc, c_bu,
                            upgrades,
                            oppo_int_embed,
                            oppo_c_bo, oppo_c_boc, oppo_c_bu,
                            nc, outputs_collections=None):
  """ Build AStar like value, which will need opponent's obs;
  in our implementation, we do not use all the opponent's obs but
  only some global vector features; moreover, opponent's (partial) obs
  will only be used in the winloss_v value, since other values are
  self-measured values
  """

  # winloss value
  with tf.variable_scope('winloss_v'):
    winloss_features = [tf.layers.flatten(z_bo), tf.layers.flatten(z_boc), z_bu,
                        tf.layers.flatten(c_bo), tf.layers.flatten(c_boc), c_bu, upgrades,
                        tf.layers.flatten(oppo_c_bo), tf.layers.flatten(oppo_c_boc), oppo_c_bu]
    winloss_v = baseline_connects(features=winloss_features,
                                  addons=[int_embed, oppo_int_embed],
                                  act_fn=astar_atan_fn,
                                  n_blk=16, enc_dim=nc.enc_dim)
  # bo value
  with tf.variable_scope('bo_v'):
    bo_v = baseline_connects(features=[tf.layers.flatten(z_bo), tf.layers.flatten(z_boc),
                                       tf.layers.flatten(c_bo), tf.layers.flatten(c_boc)],
                             addons=[],
                             act_fn=None, n_blk=16, enc_dim=nc.enc_dim)
  # bu value
  with tf.variable_scope('bu_v'):
    bu_v = baseline_connects(features=[z_bu, c_bu],
                             addons=[],
                             act_fn=None, n_blk=16, enc_dim=nc.enc_dim)
  # upgrades value
  with tf.variable_scope('upgrades_v'):
    upgrades_v = baseline_connects(
      features=[tf.multiply(z_bu, tf.constant([nc.upgrades_mask], dtype=tf.float32)),
                tf.multiply(c_bu, tf.constant([nc.upgrades_mask], dtype=tf.float32)),
                upgrades],
      addons=[],
      act_fn=None, n_blk=16, enc_dim=nc.enc_dim)
  # effects value
  with tf.variable_scope('effects_v'):
    effects_v = baseline_connects(
      features=[tf.multiply(z_bu, tf.constant(nc.effects_mask, dtype=tf.float32)),
                tf.multiply(c_bu, tf.constant(nc.effects_mask, dtype=tf.float32))],
      addons=[],
      act_fn=None, n_blk=16, enc_dim=nc.enc_dim)
  return lutils.collect_named_outputs(
    outputs_collections, 'vf',
    tf.concat([winloss_v, bo_v, bu_v, upgrades_v, effects_v], axis=-1))


@add_arg_scope
def _light_lstm_value_block(int_embed,
                            z_bo, z_boc, z_bu,
                            c_bo, c_boc, c_bu,
                            upgrades,
                            oppo_int_embed,
                            oppo_c_bo, oppo_c_boc, oppo_c_bu,
                            nc, outputs_collections=None):
  """AlphaStar-like centralized value, version 1.

  Build AStar like value, which needs opponent's obs.
  In our implementation, we do not use the full opponent's obs but
  only some global vector features. Moreover, the (partial) opponent's obs
  is only used by the winloss_v value, since other values are self-measured.
  """
  # shared zstat embeddings
  with tf.variable_scope('shared'):
    with tf.variable_scope('v_zstat', reuse=False):
      tar_bo_hs, tar_bo_h, tar_bu_embed = _zstat_connections(
        z_bo, z_boc, z_bu, nc, use_ln=True)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      _, cur_bo_h, cur_bu_embed = _zstat_connections(
        c_bo, c_boc, c_bu, nc, use_ln=True)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      _, oppo_cur_bo_h, oppo_cur_bu_embed = _zstat_connections(
        oppo_c_bo, oppo_c_boc, oppo_c_bu, nc, use_ln=True)
    # attention layer output
    with tf.variable_scope('v_att_tar_bo'):
      idx = tf.cast(tf.reduce_sum(z_bo, [1, 2]), tf.int32)  # [bs,]
      tar_z_bo_mask = tf.less_equal(
        tf.stack([tf.range(len(tar_bo_hs))] * nc.batch_size, axis=0),
        tf.expand_dims(tf.maximum(idx-1, 0), axis=-1))
      att_tar_bo_h = tp_layers.dot_prod_attention(
        tar_bo_hs, cur_bo_h, mask=tar_z_bo_mask)

  # winloss value
  with tf.variable_scope('winloss_v'):
    winloss_v = baseline_connects(
      features=[],
      addons=[int_embed, oppo_int_embed,
              tar_bo_h, att_tar_bo_h, cur_bo_h,
              tar_bu_embed, cur_bu_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed],
      act_fn=astar_atan_fn)
  # bo value
  with tf.variable_scope('bo_v'):
    bo_v = baseline_connects(features=[],
                             addons=[int_embed, oppo_int_embed,
                                     tar_bo_h, att_tar_bo_h, cur_bo_h,
                                     oppo_cur_bo_h, oppo_cur_bu_embed],
                             act_fn=None)
  z_bu_splits = tf.split(z_bu, nc.uc_split_indices, axis=-1)
  c_bu_splits = tf.split(c_bu, nc.uc_split_indices, axis=-1)
  # building_value
  with tf.variable_scope('building_v'):
    bu_v = baseline_connects(features=[z_bu_splits[0], c_bu_splits[0]],
                             addons=[int_embed, oppo_int_embed,
                                     oppo_cur_bo_h, oppo_cur_bu_embed],
                             act_fn=None)
  # effects value
  with tf.variable_scope('effects_v'):
    effects_v = baseline_connects(
      features=[z_bu_splits[1], c_bu_splits[1]],
      addons=[int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed],
      act_fn=None)
  # upgrades value
  with tf.variable_scope('upgrades_v'):
    upgrades_v = baseline_connects(
      features=[z_bu_splits[-1], c_bu_splits[-1], upgrades],
      addons=[int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed],
      act_fn=None)
  # Notice: the returned value order must keep consistent with that in the reward wrapper
  return lutils.collect_named_outputs(
    outputs_collections, 'vf',
    tf.concat([winloss_v, bo_v, bu_v, effects_v, upgrades_v], axis=-1))


@add_arg_scope
def _light_lstm_value_block_v2(int_embed,
                               z_bo, z_boc, z_bobt, z_bocbt, z_bu,
                               c_bo, c_boc, c_bobt, c_bocbt, c_bu,
                               upgrades,
                               oppo_int_embed,
                               oppo_c_bo, oppo_c_boc, oppo_c_bobt, oppo_c_bocbt, oppo_c_bu,
                               nc, use_creep_value=False, outputs_collections=None):
  """AlphaStar-like centralized value, version 2.

  Almost the same with v1, except that v2 includes more zstat channels.
  """
  # shared zstat embeddings
  with tf.variable_scope('shared'):
    with tf.variable_scope('v_zstat', reuse=False):
      tar_bo_hs, tar_bo_h, tar_bobt_hs, tar_bobt_h, tar_bu_embed = _zstat_connections_v2(
        z_bo, z_boc, z_bobt, z_bocbt, z_bu, nc, use_ln=True)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      _, cur_bo_h, _, cur_bobt_h, cur_bu_embed = _zstat_connections_v2(
        c_bo, c_boc, c_bobt, c_bocbt, c_bu, nc, use_ln=True)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      _, oppo_cur_bo_h, _, oppo_cur_bobt_h, oppo_cur_bu_embed = _zstat_connections_v2(
        oppo_c_bo, oppo_c_boc, oppo_c_bobt, oppo_c_bocbt, oppo_c_bu, nc, use_ln=True)
    # attention layer output
    with tf.variable_scope('v_att_tar_bo'):
      idx = tf.cast(tf.reduce_sum(z_bo, [1, 2]), tf.int32)  # [bs,]
      tar_z_bo_mask = tf.less_equal(
        tf.stack([tf.range(len(tar_bo_hs))] * nc.batch_size, axis=0),
        tf.expand_dims(tf.maximum(idx-1, 0), axis=-1))
      att_tar_bo_h = tp_layers.dot_prod_attention(
        tar_bo_hs, cur_bo_h, mask=tar_z_bo_mask)
    with tf.variable_scope('v_att_tar_bobt'):
      idx = tf.cast(tf.reduce_sum(z_bobt, [1, 2]), tf.int32)  # [bs,]
      tar_z_bobt_mask = tf.less_equal(
        tf.stack([tf.range(len(tar_bobt_hs))] * nc.batch_size, axis=0),
        tf.expand_dims(tf.maximum(idx-1, 0), axis=-1))
      att_tar_bobt_h = tp_layers.dot_prod_attention(
        tar_bobt_hs, cur_bobt_h, mask=tar_z_bobt_mask)

  # winloss value
  with tf.variable_scope('winloss_v'):
    winloss_v = baseline_connects(
      features=[],
      addons=[int_embed, oppo_int_embed,
              tar_bo_h, att_tar_bo_h, cur_bo_h,
              tar_bobt_h, att_tar_bobt_h, cur_bobt_h,
              tar_bu_embed, cur_bu_embed,
              oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
      act_fn=astar_atan_fn)
  # bo value
  with tf.variable_scope('bo_v'):
    bo_v = baseline_connects(features=[],
                             addons=[int_embed, oppo_int_embed,
                                     tar_bo_h, att_tar_bo_h, cur_bo_h,
                                     oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
                             act_fn=None)
  # bobt value
  with tf.variable_scope('bobt_v'):
    bobt_v = baseline_connects(features=[],
                               addons=[int_embed, oppo_int_embed,
                                       tar_bobt_h, att_tar_bobt_h, cur_bobt_h,
                                       oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
                               act_fn=None)

  z_bu_splits = tf.split(z_bu, nc.uc_split_indices, axis=-1)
  c_bu_splits = tf.split(c_bu, nc.uc_split_indices, axis=-1)
  # building_value
  with tf.variable_scope('building_v'):
    bu_v = baseline_connects(features=[z_bu_splits[0], c_bu_splits[0]],
                             addons=[int_embed, oppo_int_embed,
                                     oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
                             act_fn=None)
  # effects value
  with tf.variable_scope('effects_v'):
    effects_v = baseline_connects(
      features=[z_bu_splits[1], c_bu_splits[1]],
      addons=[int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
      act_fn=None)
  # upgrades value
  with tf.variable_scope('upgrades_v'):
    upgrades_v = baseline_connects(
      features=[z_bu_splits[-1], c_bu_splits[-1], upgrades],
      addons=[int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
      act_fn=None)
  values = [winloss_v, bo_v, bobt_v, bu_v, effects_v, upgrades_v]
  if use_creep_value:
    # tumor creep value
    with tf.variable_scope('tumor_creep_v'):
      tumor_creep_v = baseline_connects(
        features=[],
        addons=[int_embed, oppo_int_embed,
                oppo_cur_bo_h, oppo_cur_bobt_h, oppo_cur_bu_embed],
        act_fn=None)
      values.append(tumor_creep_v)
  # Notice: the returned value order must keep consistent with that in the reward wrapper
  return lutils.collect_named_outputs(
    outputs_collections, 'vf', tf.concat(values, axis=-1))


@add_arg_scope
def _light_lstm_value_block_v4(int_embed, score,
                               z_bo, z_boc, z_bu,
                               c_bo, c_boc, c_bu,
                               upgrades,
                               oppo_int_embed,
                               oppo_c_bo, oppo_c_boc, oppo_c_bu,
                               nc, outputs_collections=None):
  """AlphaStar-like centralized value, version 4.

  Almost the same with v2, except that bobt is removed.
  """

  # shared zstat embeddings
  with tf.variable_scope('shared'):
    with tf.variable_scope('v_zstat', reuse=False):
      tar_bo_hs, tar_bo_h, tar_bu_embed = _zstat_connections_v5(
        z_bo, z_boc, z_bu, nc, use_ln=True)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      _, cur_bo_h, cur_bu_embed = _zstat_connections_v5(
        c_bo, c_boc, c_bu, nc, use_ln=True)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      _, oppo_cur_bo_h, oppo_cur_bu_embed = _zstat_connections_v5(
        oppo_c_bo, oppo_c_boc, oppo_c_bu, nc, use_ln=True)
    # attention layer output
    with tf.variable_scope('v_att_tar_bo'):
      idx = tf.cast(tf.reduce_sum(z_bo, [1, 2]), tf.int32)  # [bs,]
      tar_z_bo_mask = tf.less_equal(
        tf.stack([tf.range(len(tar_bo_hs))] * nc.batch_size, axis=0),
        tf.expand_dims(tf.maximum(idx-1, 0), axis=-1))
      att_tar_bo_h = tp_layers.dot_prod_attention(
        tar_bo_hs, cur_bo_h, mask=tar_z_bo_mask)

  if nc.use_score_in_value:
    x_vec_score = tfc_layers.fully_connected(score, 64, scope='x_vec_score')
  # winloss value
  with tf.variable_scope('winloss_v'):
    addons = [int_embed, oppo_int_embed,
              tar_bo_h, att_tar_bo_h, cur_bo_h,
              tar_bu_embed, cur_bu_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed]
    if nc.use_score_in_value:
      addons.append(x_vec_score)
    winloss_v = baseline_connects(
      features=[], addons=addons, act_fn=astar_atan_fn)
  # bo value
  with tf.variable_scope('bo_v'):
    addons = [int_embed, oppo_int_embed,
              tar_bo_h, att_tar_bo_h, cur_bo_h,
              oppo_cur_bo_h, oppo_cur_bu_embed]
    if nc.use_score_in_value:
      addons.append(x_vec_score)
    bo_v = baseline_connects(features=[], addons=addons, act_fn=None)

  z_bu_splits = tf.split(z_bu, nc.uc_split_indices, axis=-1)
  c_bu_splits = tf.split(c_bu, nc.uc_split_indices, axis=-1)
  # building_value
  with tf.variable_scope('building_v'):
    addons = [int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed]
    if nc.use_score_in_value:
      addons.append(x_vec_score)
    bu_v = baseline_connects(features=[z_bu_splits[0], c_bu_splits[0]],
                             addons=addons, act_fn=None)
  # effects value
  with tf.variable_scope('effects_v'):
    addons = [int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed]
    if nc.use_score_in_value:
      addons.append(x_vec_score)
    effects_v = baseline_connects(
      features=[z_bu_splits[1], c_bu_splits[1]], addons=addons, act_fn=None)
  # upgrades value
  with tf.variable_scope('upgrades_v'):
    addons = [int_embed, oppo_int_embed,
              oppo_cur_bo_h, oppo_cur_bu_embed]
    if nc.use_score_in_value:
      addons.append(x_vec_score)
    upgrades_v = baseline_connects(
      features=[z_bu_splits[-1], c_bu_splits[-1], upgrades],
      addons=addons, act_fn=None)
  # Notice: the returned value order must keep consistent with that in the reward wrapper
  return lutils.collect_named_outputs(
    outputs_collections, 'vf',
    tf.concat([winloss_v, bo_v, bu_v, effects_v, upgrades_v], axis=-1))


@add_arg_scope
def _light_trans_value_block_v1(int_embed,
                                z_bo, z_boc, z_bu,
                                c_bo, c_boc, c_bu,
                                upgrades,
                                oppo_int_embed,
                                oppo_c_bo, oppo_c_boc, oppo_c_bu,
                                nc, outputs_collections=None):
  """AlphaStar-like centralized value, version trans_v1.

  Based on v4, while it use the way the bo and bu connects in _zstat_embed_block_v4d1.
  """
  # shared zstat embeddings
  with tf.variable_scope('shared'):
    with tf.variable_scope('v_zstat', reuse=False):
      tar_bo_embed = _zstat_bo_connections(z_bo, z_boc, nc, out_dim=32)
      tar_bu_embed = _zstat_bu_connections_v2(z_bu, nc, out_dim=32)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      cur_bo_embed = _zstat_bo_connections(c_bo, c_boc, nc, out_dim=32)
      cur_bu_embed = _zstat_bu_connections_v2(c_bu, nc, out_dim=32)
    with tf.variable_scope('v_zstat', reuse=tf.AUTO_REUSE):
      oppo_cur_bo_embed = _zstat_bo_connections(oppo_c_bo, oppo_c_boc, nc, out_dim=32)
      oppo_cur_bu_embed = _zstat_bu_connections_v2(oppo_c_bu, nc, out_dim=32)

  # winloss value
  with tf.variable_scope('winloss_v'):
    winloss_v = baseline_connects(
      features=[],
      addons=[int_embed, oppo_int_embed,
              tar_bo_embed, cur_bo_embed,
              tar_bu_embed, cur_bu_embed,
              oppo_cur_bo_embed, oppo_cur_bu_embed],
      act_fn=astar_atan_fn)
  # bo value
  with tf.variable_scope('bo_v'):
    bo_v = baseline_connects(features=[],
                             addons=[int_embed, oppo_int_embed,
                                     tar_bo_embed, cur_bo_embed,
                                     oppo_cur_bo_embed, oppo_cur_bu_embed],
                             act_fn=None)

  z_bu_splits = tf.split(z_bu, nc.uc_split_indices, axis=-1)
  c_bu_splits = tf.split(c_bu, nc.uc_split_indices, axis=-1)
  # building_value
  with tf.variable_scope('building_v'):
    bu_v = baseline_connects(features=[z_bu_splits[0], c_bu_splits[0]],
                             addons=[int_embed, oppo_int_embed,
                                     oppo_cur_bo_embed, oppo_cur_bu_embed],
                             act_fn=None)
  # effects value
  with tf.variable_scope('effects_v'):
    effects_v = baseline_connects(
      features=[z_bu_splits[1], c_bu_splits[1]],
      addons=[int_embed, oppo_int_embed,
              oppo_cur_bo_embed, oppo_cur_bu_embed],
      act_fn=None)
  # upgrades value
  with tf.variable_scope('upgrades_v'):
    upgrades_v = baseline_connects(
      features=[z_bu_splits[-1], c_bu_splits[-1], upgrades],
      addons=[int_embed, oppo_int_embed,
              oppo_cur_bo_embed, oppo_cur_bu_embed],
      act_fn=None)
  # Notice: the returned value order must keep consistent with that in the reward wrapper
  return lutils.collect_named_outputs(
    outputs_collections, 'vf',
    tf.concat([winloss_v, bo_v, bu_v, effects_v, upgrades_v], axis=-1))


@add_arg_scope
def _transformer_block(units_embed, units_mask, nc, outputs_collections=None):
  """
  Version 1: similar to AStar but use feed forward
  Used in v5d8; loss can normally decrease
  :param units_embed: [?, 600, feature dim]
  :param units_mask: [?, 600]
  :param n_blk: num of blocks
  :param dropout_rate:
  :param enc_dim:
  :return:
  """
  n_blk = nc.trans_n_blk  # 3 per AStar
  dropout_rate = nc.trans_dropout_rate
  enc_dim = nc.enc_dim  # per AStar
  with tf.variable_scope("transformer_encoder") as sc:
    embed = units_embed
    # Blocks
    for i in range(n_blk):
      with tf.variable_scope("num_blocks_{}".format(i)):
        # self-attention
        embed = tp_layers.multihead_attention_v2(
          queries=embed,
          keys=embed,
          values=embed,
          entry_mask=units_mask,
          num_heads=2,  # per AStar
          dropout_rate=dropout_rate,
        )
        # feed forward
        embed = tp_layers.ff(embed, num_units=[enc_dim, enc_dim])
    # TODO: in AStar, the connections right here are hard to follow. See below:
    ## "The preprocessed entities and biases are fed into a transformer with
    ## 3 layers of 2-headed self-attention. In each layer, each self-attention
    ## head uses keys, queries, and values of size 128, then passes the aggregated
    ## values through a Conv1D with kernel size 1 to double the number of channels
    ## (to 256). The head results are summed and passed through a 2-layer MLP with
    ## hidden size 1024 and output size 256. The transformer output is passed
    ## through a ReLU, 1D convolution with 256 channels and kernel size 1, and
    ## another ReLU to yield `entity_embeddings`."
    # Per AStar and roughly:
    # after 3 layers of 2-headed self-attention is a 2-layer MLP with
    # hidden size 1024 and output size 256 + relu + 256 + relu
    embed = tfc_layers.fully_connected(embed, 1024, activation_fn=None)
    embed = tfc_layers.fully_connected(embed, 1024, activation_fn=None)
    embed = tfc_layers.fully_connected(embed, 256, activation_fn=tf.nn.relu)
    embed = tfc_layers.fully_connected(embed, 256, activation_fn=tf.nn.relu)
    embed = tp_ops.mask_embed(embed, units_mask)
    # AStar uses mean-pooling to further process the units_embed.
    # We used both mean-pooling and max-pooling
    units_mask = tf.cast(tf.expand_dims(units_mask, axis=-1), tf.float32)
    embedded_unit = (tf.reduce_mean(embed * units_mask, axis=1) /
                     (tf.reduce_mean(units_mask, axis=1) + 1e-8))
    return (
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'emb',
                                   embed),
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'avg_emb',
                                   embedded_unit)
    )


@add_arg_scope
def _transformer_block_v2(units_embed, units_mask, nc,
                          outputs_collections=None):
  """
  Version 2: identical to AStar
  :param units_embed: [?, 600, feature dim]
  :param units_mask: [?, 600]
  :param n_blk: num of blocks
  :param dropout_rate:
  :param enc_dim:
  :return:
  in AStar, the connections right here are:
  "The preprocessed entities and biases are fed into a transformer with
  3 layers of 2-headed self-attention. In each layer, each self-attention head
  uses keys, queries, and values of size 128, then passes the aggregated
  values through a Conv1D with kernel size 1 (fc in our term) to double the number of channels
  (to 256). The head results are summed and passed through a 2-layer MLP with
  hidden size 1024 and output size 256. The transformer output is passed
  through a ReLU, 1D convolution with 256 channels and kernel size 1, and
  another ReLU to yield `entity_embeddings`."
  """
  n_blk = nc.trans_n_blk  # 3 per AStar
  dropout_rate = nc.trans_dropout_rate
  enc_dim = 256  # 256 per AStar, each head is 128

  with tf.variable_scope("transformer_encoder") as sc:
    embed = units_embed
    # Blocks
    for i in range(n_blk):
      with tf.variable_scope("num_blocks_{}".format(i)):
        # 2-headed self-attention
        embed = tp_layers.multihead_attention_v3(
          queries=embed,
          keys=embed,
          values=embed,
          entry_mask=units_mask,
          num_heads=2,  # per AStar
          enc_dim=enc_dim,
          dropout_rate=dropout_rate,
        )
    # The transformer output is passed through a ReLU,
    # 1D convolution with 256 channels and kernel size 1,
    # and another ReLU to yield `entity_embeddings`
    embed = tf.nn.relu(embed)
    unit_embeddings = tfc_layers.fully_connected(embed, 256,
                                                 activation_fn=tf.nn.relu)
    unit_embeddings = tp_ops.mask_embed(unit_embeddings, units_mask)
    # AStar uses mean-pooling to further process the units_embed.
    units_mask = tf.cast(tf.expand_dims(units_mask, axis=-1), tf.float32)
    embedded_unit = (tf.reduce_mean(embed * units_mask, axis=1) /
                     (tf.reduce_mean(units_mask, axis=1) + 1e-8))
    return (
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'emb',
                                   unit_embeddings),
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'avg_emb',
                                   embedded_unit)
    )


@add_arg_scope
def _transformer_block_v3(units_embed, units_mask, nc,
                          outputs_collections=None):
  """Transformer Block, v3

  Args:
    units_embed: [bs, 600, feature dim]
    units_mask: [bs, 600]
    nc:

  Returns:
    A (bs, 600, dim) Tensor representing the output units embeddings
  """

  # AStar paper goes like:
  # "The preprocessed entities and biases are fed into a transformer with
  # 3 layers of 2-headed self-attention. In each layer, each self-attention head
  # uses keys, queries, and values of size 128, then passes the aggregated
  # values through a Conv1D with kernel size 1 (fc in our term) to double the number of channels
  # (to 256). The head results are summed and passed through a 2-layer MLP with
  # hidden size 1024 and output size 256. The transformer output is passed
  # through a ReLU, 1D convolution with 256 channels and kernel size 1, and
  # another ReLU to yield `entity_embeddings`."

  n_blk = nc.trans_n_blk  # 3 per AStar
  dropout_rate = nc.trans_dropout_rate
  enc_dim = 256  # 256 per AStar, each head is 128

  with tf.variable_scope("transformer_encoder") as sc:
    embed = units_embed
    # Blocks
    for i in range(n_blk):
      with tf.variable_scope("num_blocks_{}".format(i)):
        # self-attention
        embed = tp_layers.multihead_attention_v2(
          queries=embed,
          keys=embed,
          values=embed,
          entry_mask=units_mask,
          num_heads=2,  # per AStar
          dropout_rate=dropout_rate,
        )
        # feed forward
        embed = tp_layers.ff(embed, num_units=[enc_dim, enc_dim])

    # Outputs encoder. AStar paper goes like:
    # The transformer output is passed through a ReLU, 1D convolution with 256
    # channels and kernel size 1, and another ReLU to yield `entity_embeddings`.
    # The mean of the transformer output across across the units (masked by the
    # missing entries) is fed through a linear layer of size 256 and a ReLU to
    # yield `embedded_entity`.

    # NOTE(pengsun): intentionally comment out the relu, as the embed has been
    # layer normalized
    #embed = tf.nn.relu(embed)
    # this fc is equivalent to kernel size 1 conv1d
    unit_embeddings = tfc_layers.fully_connected(embed, 256,
                                                 activation_fn=tf.nn.relu)
    unit_embeddings = tp_ops.mask_embed(unit_embeddings, units_mask)
    # masked mean pooling across the units (along axis=1)
    units_mask = tf.cast(tf.expand_dims(units_mask, axis=-1), tf.float32)
    embedded_unit = (tf.reduce_sum(embed * units_mask, axis=1) /
                     (tf.reduce_sum(units_mask, axis=1) + 1e-8))
    embedded_unit = tfc_layers.fully_connected(embedded_unit, 256,
                                               activation_fn=tf.nn.relu)
    return (
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'emb',
                                   unit_embeddings),
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'avg_emb',
                                   embedded_unit)
    )


@add_arg_scope
def _transformer_block_v4(units_embed, units_mask, nc,
                          enc_dim=256,
                          out_fc_dim=256,
                          n_blk=None,
                          scope=None,
                          outputs_collections=None):
  """Transformer Block, v4

  The same with _transformer_block v1, but removes the two linear layers.

  Args:
    units_embed: [bs, 600, feature dim]
    units_mask: [bs, 600]
    nc: net config
    enc_dim: int, encoding dim for ff layer
    out_fc_dim: int, dim for output fc layer dim
    n_blk: int, number of block; using nc.trans_n_blk when not specified
    scope:
    outputs_collections:

  Returns:
    A (bs, 600, dim) Tensor representing the output units embeddings
  """
  n_blk = n_blk or nc.trans_n_blk  # e.g., 3 per AStar
  dropout_rate = nc.trans_dropout_rate
  with tf.variable_scope(scope, default_name='transformer_encoder') as sc:
    embed = units_embed
    # Blocks
    for i in range(n_blk):
      with tf.variable_scope("num_blocks_{}".format(i)):
        # self-attention
        embed = tp_layers.multihead_attention_v2(
          queries=embed,
          keys=embed,
          values=embed,
          entry_mask=units_mask,
          num_heads=2,  # per AStar
          dropout_rate=dropout_rate,
        )
        # feed forward
        embed = tp_layers.ff(embed, num_units=[enc_dim, enc_dim])
    # TODO: in AStar, the connections right here are hard to follow. See below:
    ## "The preprocessed entities and biases are fed into a transformer with
    ## 3 layers of 2-headed self-attention. In each layer, each self-attention
    ## head uses keys, queries, and values of size 128, then passes the aggregated
    ## values through a Conv1D with kernel size 1 to double the number of channels
    ## (to 256). The head results are summed and passed through a 2-layer MLP with
    ## hidden size 1024 and output size 256. The transformer output is passed
    ## through a ReLU, 1D convolution with 256 channels and kernel size 1, and
    ## another ReLU to yield `entity_embeddings`."
    # Per AStar and roughly:
    # after 3 layers of 2-headed self-attention is a 2-layer MLP with
    # hidden size 1024 and output size 256 + relu + 256 + relu
    embed = tfc_layers.fully_connected(embed, out_fc_dim, activation_fn=tf.nn.relu)
    embed = tfc_layers.fully_connected(embed, out_fc_dim, activation_fn=tf.nn.relu)
    embed = tp_ops.mask_embed(embed, units_mask)
    # AStar uses mean-pooling to further process the units_embed.
    # We used both mean-pooling and max-pooling
    units_mask = tf.cast(tf.expand_dims(units_mask, axis=-1), tf.float32)
    embedded_unit = (tf.reduce_mean(embed * units_mask, axis=1) /
                     (tf.reduce_mean(units_mask, axis=1) + 1e-8))
    return (
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'emb',
                                   embed),
      lutils.collect_named_outputs(outputs_collections,
                                   sc.original_name_scope + 'avg_emb',
                                   embedded_unit)
    )


@add_arg_scope
def _transformer_block_v5(units_embed, units_mask, nc,
                          enc_dim=256, out_fc_dim=256,
                          scope=None,
                          outputs_collections=None):
    units_embed, embedded_unit = _transformer_block_v4(
      units_embed=units_embed,
      units_mask=units_mask,
      enc_dim=enc_dim,
      out_fc_dim=out_fc_dim,
      nc=nc,
      scope=scope,
      outputs_collections=outputs_collections)
    # the output embedded_unit from _transformer_block_v4 is not passed through a relu;
    # this trans_block may also be used for zstat embedding
    embedded_unit = tfc_layers.fully_connected(embedded_unit, out_fc_dim, activation_fn=tf.nn.relu)
    return units_embed, embedded_unit


def _action_mask_weights(inputs_ab, inputs_arg_mask, weights_include_ab=True,
                         split_to_list=True):
  """action mask as weights for a given ability batch

  :param inputs_ab: (bs,)
  :param inputs_arg_mask: (bs, N)
  :return a list of (bs,) Tensors. list len = N (weights_include_ab=True) or
   1 + N (weights_include_ab=True)
  """
  mask_weights = tp_ops.to_float32(
    tf.nn.embedding_lookup(inputs_arg_mask, inputs_ab))
  if weights_include_ab:
    bs = inputs_ab.shape[0].value
    mask_weights = tf.concat([tf.ones(shape=(bs, 1)), mask_weights], axis=-1)
  if split_to_list:
    n_action_heads = mask_weights.shape[-1].value
    mask_weights = tf.split(mask_weights, [1] * n_action_heads, axis=-1)
    mask_weights = [tf.squeeze(each, axis=-1) for each in mask_weights]
  return mask_weights


def _make_shared_embed_scopes() -> MNetV5EmbedScope:
  with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('unit_embed') as units_embed_sc:
      pass
    with tf.variable_scope('buff_embed') as buff_embed_sc:
      pass
    with tf.variable_scope('ab_embed') as ab_embed_sc:
      pass
    with tf.variable_scope('noop_num_embed') as noop_num_embed_sc:
      pass
    with tf.variable_scope('order_embed') as order_embed_sc:
      pass
    with tf.variable_scope('u_type_embed') as u_type_embed_sc:
      pass
    with tf.variable_scope('base_embed') as base_embed_sc:
      pass
  return MNetV5EmbedScope(units_embed_sc,
                          buff_embed_sc,
                          ab_embed_sc,
                          noop_num_embed_sc,
                          order_embed_sc,
                          u_type_embed_sc,
                          base_embed_sc)


def _make_inputs_palceholders(nc: MNetV6Config):
  X_ph = tp_utils.placeholders_from_gym_space(
    nc.ob_space, batch_size=nc.batch_size, name='ob_ph')

  if nc.use_self_fed_heads:
    # when testing, there are no ground-truth actions
    # A_pah = tp_utils.map_gym_space_to_structure(lambda x: None, nc.ac_space)
    A_pah = None
  else:
    A_pah = tp_utils.placeholders_from_gym_space(
      nc.ac_space, batch_size=nc.batch_size, name='ac_ph')

  neglogp = tp_utils.map_gym_space_to_structure(
    func=lambda x_sp: tf.placeholder(shape=(nc.batch_size, ),
                                     dtype=tf.float32,
                                     name='neglogp'),
    gym_sp=nc.ac_space
  )
  R = tf.placeholder(tf.float32, (nc.batch_size, nc.n_v), 'R')
  V = tf.placeholder(tf.float32, (nc.batch_size, nc.n_v), 'V')
  r = tf.placeholder(tf.float32, (nc.batch_size, nc.n_v), 'r')
  discount = tf.placeholder(tf.float32, (nc.batch_size,), 'discount')
  S = tf.placeholder(tf.float32, (nc.batch_size, nc.hs_len), 'hs')
  M = tf.placeholder(tf.float32, (nc.batch_size,), 'hsm')
  logits = tp_utils.map_gym_space_to_structure(
    func=lambda x_sp: tf.placeholder(
      shape=(nc.batch_size, make_pdtype(x_sp).param_shape()[0]),
      dtype=tf.float32,
      name='logits'
    ),
    gym_sp=nc.ac_space
  )

  return MNetV5Inputs(X_ph, A_pah, neglogp, R, V, S, M, logits, r, discount)


def _make_mnet_v5_arg_scope_a(nc: MNetV6Config, endpoints_collections: str):
  """Make arg_scope, type a"""
  ep_fns = []
  if nc.endpoints_verbosity >= 2:
    ep_fns += [
      tp_layers.linear_embed,
      _gather_units_block,
      _units_embed_block,
      _transformer_block,
      _transformer_block_v2,
      _transformer_block_v3,
      _transformer_block_v4,
      _selected_embed_block,
      _scatter_units_block,
      _spa_embed_block_v2,
      _u2d_spa_embed_block,
      _map_spa_embed_block,
      _map_spa_embed_block_v2,
      _vec_embed_block,
      _vec_embed_block_v2,
      _vec_embed_block_v3,
      _last_action_embed_block,
      _last_action_embed_block_mnet_v6,
      _last_action_embed_block_mnet_v6_v2,
      _integrate_embed,
      _integrate_embed_v2,
      lstm_embed_block,
      _zstat_embed,
    ]
  if nc.endpoints_verbosity >= 4:
    ep_fns += [
      _const_arg_mask,
      _const_mask_base_poses,
      _const_base_poses,
      _pre_discrete_action_block,
      _pre_discrete_action_res_block,
      _pre_discrete_action_fc_block,
      _pre_action_temporal_smoothing_block,
      _pre_action_res_temporal_smoothing_block,
      _pre_ptr_action_block,
      _pre_ptr_action_res_block,
      _pre_multinomial_action_block,
      _pre_multinomial_action_res_block,
      _pre_loc_action_block,
      _pre_loc_action_res_block,
      _pre_loc_action_res_bottleneck_block,
      _pre_loc_action_res_bottleneck_block_v2,
      _pre_loc_action_res_bottleneck_block_vold,
      _pre_loc_action_astar_like_block_v1,
      _pos_spa_embed_block,
      _pos_spa_embed_res_block,
      _pos_spa_embed_res_block_v2,
      _pos_spa_embed_res_block_vold,
    ]
  if nc.endpoints_verbosity >= 6:
    ep_fns += [
      tfc_layers.fully_connected,
      tfc_layers.conv2d,
      tfc_layers.max_pool2d,
    ]
  weights_regularizer = (None if nc.weight_decay is None
                         else l2_regularizer(nc.weight_decay))
  with arg_scope(ep_fns,
                 outputs_collections=endpoints_collections):
    with arg_scope([tfc_layers.conv2d, tfc_layers.fully_connected],
                   weights_regularizer=weights_regularizer) as arg_sc:
      return arg_sc


def _make_mnet_v5_vars(scope) -> MNetV5TrainableVariables:
  scope = scope if isinstance(scope, str) else scope.name + '/'
  all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
  lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                '{}.*{}'.format(scope, 'lstm_embed'))
  return MNetV5TrainableVariables(all_vars=all_vars, lstm_vars=lstm_vars)


def _make_mnet_v5_endpoints_dict(nc: MNetV6Config, endpoints_collections: str,
                                 name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask', alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese', alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses', alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base', alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embedding
  _safe_collect(ep_key='ab_emb', alias='shared/ab_embed',
                scope='{}.*{}.*'.format(name_scope, 'ab_embed'))
  _safe_collect(ep_key='noop_emb', alias='shared/noop_num_embed',
                scope='{}.*{}.*'.format(name_scope, 'noop_num_embed'))
  _safe_collect(ep_key='buff_emb', alias='shared/buff_embed',
                scope='{}.*{}.*'.format(name_scope, 'buff_embed'))
  _safe_collect(ep_key='order_emb', alias='shared/order_embed',
                scope='{}.*{}.*'.format(name_scope, 'order_embed'))
  _safe_collect(ep_key='u_type_emb', alias='shared/u_type_embed',
                scope='{}.*{}.*'.format(name_scope, 'u_type_embed'))
  # high-level embedding
  _safe_collect(ep_key='units_emb_blk', alias='units_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='selected_emb_blk', alias='selected_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='scatter_units_blk', alias='scatter_units',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding
  _safe_collect(ep_key='spa_emb_blk_img', alias='spa_embed_img',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='spa_emb_blk_vec', alias='spa_embed_vec',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding: grid net
  _safe_collect(ep_key='gridnet_input_img', alias='x_img_input',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc1', alias='x_img_fc1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc2', alias='x_img_fc2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc3', alias='x_img_fc3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  # large
  _safe_collect(ep_key='gridnet_large_c1', alias='large/conv1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c2', alias='large/conv2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_mc', alias='large/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c3', alias='large/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_rs', alias='large/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  # medium
  _safe_collect(ep_key='gridnet_medium_p1', alias='medium/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_p2', alias='medium/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_mc', alias='medium/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_c3', alias='medium/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_rs', alias='medium/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  # small
  _safe_collect(ep_key='gridnet_small_p1', alias='small/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p2', alias='small/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p3', alias='small/pool3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p4', alias='small/pool4',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_mc', alias='small/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_c5', alias='small/conv5',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_rs', alias='small/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  # high-level vector embedding
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))
  # high-level last action embedding
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))
  # high-level integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))
  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))

  # pre-layer for the heads
  # lvl 1
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 2
  _safe_collect(ep_key='head_noop_pre', alias='pre_action_temporal_smoothing',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 3
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 4, select
  _safe_collect(ep_key='head_ss_pre', alias='pre_ptr_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/s_select'))
  _safe_collect(ep_key='head_ms_pre', alias='pre_multinomial_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/m_select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))
  _safe_collect(
    ep_key='head_cmd_creep_pos_pre',
    alias='pre_loc_action',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_creep_pos')
  )
  _safe_collect(
    ep_key='head_cmd_nydus_pos_pre',
    alias='pre_loc_action',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_nydus_pos')
  )
  # lvl 4, others
  _safe_collect(ep_key='head_cmd_base_pre', alias='pre_ptr_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_unload_pre', alias='pre_ptr_action',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_unload'))

  return ep


def _make_mnet_v5d1_endpoints_dict(nc: MNetV6Config, endpoints_collections: str,
                                   name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask', alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese', alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses', alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base', alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embedding
  _safe_collect(ep_key='ab_emb', alias='shared/ab_embed',
                scope='{}.*{}.*'.format(name_scope, 'ab_embed'))
  _safe_collect(ep_key='noop_emb', alias='shared/noop_num_embed',
                scope='{}.*{}.*'.format(name_scope, 'noop_num_embed'))
  _safe_collect(ep_key='buff_emb', alias='shared/buff_embed',
                scope='{}.*{}.*'.format(name_scope, 'buff_embed'))
  _safe_collect(ep_key='order_emb', alias='shared/order_embed',
                scope='{}.*{}.*'.format(name_scope, 'order_embed'))
  _safe_collect(ep_key='u_type_emb', alias='shared/u_type_embed',
                scope='{}.*{}.*'.format(name_scope, 'u_type_embed'))
  # high-level embedding
  _safe_collect(ep_key='units_emb_blk', alias='units_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='selected_emb_blk', alias='selected_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='scatter_units_blk', alias='scatter_units',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding
  _safe_collect(ep_key='spa_emb_blk_img', alias='spa_embed_img',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='spa_emb_blk_vec', alias='spa_embed_vec',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding: grid net
  _safe_collect(ep_key='gridnet_input_img', alias='x_img_input',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc1', alias='x_img_fc1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc2', alias='x_img_fc2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc3', alias='x_img_fc3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  # large
  _safe_collect(ep_key='gridnet_large_c1', alias='large/conv1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c2', alias='large/conv2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_mc', alias='large/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c3', alias='large/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_rs', alias='large/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  # medium
  _safe_collect(ep_key='gridnet_medium_p1', alias='medium/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_p2', alias='medium/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_mc', alias='medium/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_c3', alias='medium/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_rs', alias='medium/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  # small
  _safe_collect(ep_key='gridnet_small_p1', alias='small/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p2', alias='small/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p3', alias='small/pool3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p4', alias='small/pool4',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_mc', alias='small/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_c5', alias='small/conv5',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_rs', alias='small/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  # high-level vector embedding
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))
  # high-level last action embedding
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))
  # high-level integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))
  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))

  # pre-layer for the heads
  # lvl 1
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 2
  _safe_collect(ep_key='head_noop_pre',
                alias='pre_action_res_temporal_smoothing',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 3
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 4, select
  _safe_collect(ep_key='head_ss_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/s_select'))
  _safe_collect(ep_key='head_ms_pre', alias='pre_multinomial_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/m_select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))
  _safe_collect(
    ep_key='head_cmd_creep_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_creep_pos')
  )
  _safe_collect(
    ep_key='head_cmd_nydus_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_nydus_pos')
  )
  # lvl 4, others
  _safe_collect(ep_key='head_cmd_base_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_unload_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_unload'))

  return ep


def _make_mnet_v5d7_endpoints_dict(nc: MNetV6Config, endpoints_collections: str,
                                   name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask', alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese', alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses', alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base', alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embeddings
  _safe_collect(ep_key='ab_emb', alias='shared/ab_embed',
                scope='{}.*{}.*'.format(name_scope, 'ab_embed'))
  _safe_collect(ep_key='noop_emb', alias='shared/noop_num_embed',
                scope='{}.*{}.*'.format(name_scope, 'noop_num_embed'))
  _safe_collect(ep_key='buff_emb', alias='shared/buff_embed',
                scope='{}.*{}.*'.format(name_scope, 'buff_embed'))
  _safe_collect(ep_key='order_emb', alias='shared/order_embed',
                scope='{}.*{}.*'.format(name_scope, 'order_embed'))
  _safe_collect(ep_key='u_type_emb', alias='shared/u_type_embed',
                scope='{}.*{}.*'.format(name_scope, 'u_type_embed'))

  # units embeddings related
  _safe_collect(ep_key='u2d_spa_embed_out',
                alias='u2d_spa_embed_out',
                scope='{}.*{}'.format(name_scope, 'u2d_spa_embed'))
  _safe_collect(ep_key='gather_units', alias='gather_units',
                scope='{}.*{}'.format(name_scope, 'embed/gather_units'))

  # map embeddings related
  _safe_collect(ep_key='map_spa_embed_resize',
                alias='map_spa_embed_resize',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed'))
  _safe_collect(ep_key='map_spa_embed_small',
                alias='map_spa_embed_small',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed'))
  _safe_collect(ep_key='map_spa_embed_res',
                alias='map_spa_embed_res',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed'))
  _safe_collect(ep_key='map_spa_embed_out',
                alias='map_spa_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed'))
  _safe_collect(
    ep_key='enhanced_gather_units',
    alias='enhanced_gather_units',
    scope='{}.*{}'.format(name_scope, 'embed/enhanced_gather_units')
  )

  # vector embeddings related
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))
  # last action embeddings related
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))
  # integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))
  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))
  # pre-layer for the heads
  # lvl 1
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 2
  _safe_collect(ep_key='head_noop_pre',
                alias='pre_action_res_temporal_smoothing',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 3
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 4, select
  _safe_collect(ep_key='head_ss_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/s_select'))
  _safe_collect(ep_key='head_ms_pre', alias='pre_multinomial_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/m_select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))
  _safe_collect(
    ep_key='head_cmd_creep_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_creep_pos')
  )
  _safe_collect(
    ep_key='head_cmd_nydus_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_nydus_pos')
  )
  # lvl 4, others
  _safe_collect(ep_key='head_cmd_base_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_unload_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_unload'))
  return ep


def _make_mnet_v5d9_endpoints_dict(nc: MNetV6Config, endpoints_collections: str,
                                   name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    #assert len(o) == 1
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask',
                alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese',
                alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses',
                alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base',
                alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embeddings
  _safe_collect(ep_key='ab_emb',
                alias='shared/ab_embed',
                scope='{}.*heads.*action_head/ab_embed'.format(name_scope))
  _safe_collect(ep_key='noop_emb',
                alias='shared/noop_num_embed',
                scope='{}.*heads.*/noop_num_embed'.format(name_scope))
  _safe_collect(ep_key='buff_emb',
                alias='shared/buff_embed',
                scope='{}.*embed/units_embed.*/buff_embed'.format(name_scope))
  _safe_collect(ep_key='order_emb',
                alias='shared/order_embed',
                scope='{}.*embed.*/order_embed'.format(name_scope))
  _safe_collect(ep_key='u_type_emb',
                alias='shared/u_type_embed',
                scope='{}.*heads.*/u_type_embed'.format(name_scope))

  # units embeddings related
  _safe_collect(ep_key='u2d_spa_embed_out',
                alias='u2d_spa_embed_out',
                scope='{}.*{}'.format(name_scope, 'u2d_spa_embed'))
  _safe_collect(ep_key='gather_units', alias='gather_units',
                scope='{}.*{}'.format(name_scope, 'embed/gather_units'))

  # map embeddings related
  _safe_collect(ep_key='map_spa_embed_v2_resize',
                alias='map_spa_embed_v2_resize',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed_v2'))
  _safe_collect(ep_key='map_spa_embed_v2_small',
                alias='map_spa_embed_v2_small',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed_v2'))
  _safe_collect(ep_key='map_spa_embed_v2_res',
                alias='map_spa_embed_v2_res',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed_v2'))
  _safe_collect(ep_key='map_spa_embed_v2_out',
                alias='map_spa_embed_v2_out',
                scope='{}.*{}.*'.format(name_scope, 'map_spa_embed_v2'))
  _safe_collect(
    ep_key='enhanced_gather_units',
    alias='enhanced_gather_units',
    scope='{}.*{}'.format(name_scope, 'embed/enhanced_gather_units')
  )

  # vector embeddings related
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))

  # last action embeddings related
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))

  # integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))

  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))

  # pre-layer for the heads
  # lvl 1
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 2
  _safe_collect(ep_key='head_noop_pre',
                alias='pre_action_res_temporal_smoothing',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 3
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 4, select
  _safe_collect(ep_key='head_ss_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/s_select'))
  _safe_collect(ep_key='head_ms_pre', alias='pre_multinomial_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/m_select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))
  _safe_collect(
    ep_key='head_cmd_creep_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_creep_pos')
  )
  _safe_collect(
    ep_key='head_cmd_nydus_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_nydus_pos')
  )
  # lvl 4, others
  _safe_collect(ep_key='head_cmd_base_gather_units',
                alias='gather_units',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_base_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_unload_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_unload'))

  return ep


def _make_mnet_v5d10_endpoints_dict(nc: MNetV6Config,
                                    endpoints_collections: str,
                                    name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask', alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese', alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses', alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base', alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embedding
  _safe_collect(ep_key='ab_emb', alias='shared/ab_embed',
                scope='{}.*{}.*'.format(name_scope, 'ab_embed'))
  _safe_collect(ep_key='noop_emb', alias='shared/noop_num_embed',
                scope='{}.*{}.*'.format(name_scope, 'noop_num_embed'))
  _safe_collect(ep_key='buff_emb', alias='shared/buff_embed',
                scope='{}.*{}.*'.format(name_scope, 'buff_embed'))
  _safe_collect(ep_key='order_emb', alias='shared/order_embed',
                scope='{}.*{}.*'.format(name_scope, 'order_embed'))
  _safe_collect(ep_key='u_type_emb', alias='shared/u_type_embed',
                scope='{}.*{}.*'.format(name_scope, 'u_type_embed'))
  # high-level embedding
  _safe_collect(ep_key='units_emb_blk', alias='units_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='selected_emb_blk', alias='selected_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='scatter_units_blk', alias='scatter_units',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding
  _safe_collect(ep_key='spa_emb_blk_img', alias='spa_embed_img',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='spa_emb_blk_vec', alias='spa_embed_vec',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding: grid net
  _safe_collect(ep_key='gridnet_input_img', alias='x_img_input',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc1', alias='x_img_fc1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc2', alias='x_img_fc2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc3', alias='x_img_fc3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  # large
  _safe_collect(ep_key='gridnet_large_c1', alias='large/conv1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c2', alias='large/conv2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_mc', alias='large/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c3', alias='large/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_rs', alias='large/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  # medium
  _safe_collect(ep_key='gridnet_medium_p1', alias='medium/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_p2', alias='medium/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_mc', alias='medium/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_c3', alias='medium/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_rs', alias='medium/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  # small
  _safe_collect(ep_key='gridnet_small_p1', alias='small/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p2', alias='small/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p3', alias='small/pool3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p4', alias='small/pool4',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_mc', alias='small/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_c5', alias='small/conv5',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_rs', alias='small/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  # high-level vector embedding
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))
  # high-level last action embedding
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))
  # high-level integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))
  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))

  # pre-layer for the heads
  # lvl 1
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 2
  _safe_collect(ep_key='head_noop_pre',
                alias='pre_action_res_temporal_smoothing',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 3
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 4, select
  _safe_collect(ep_key='head_ss_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/s_select'))
  _safe_collect(ep_key='head_ms_pre', alias='pre_multinomial_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/m_select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(
    ep_key='pos_spa_embed_res_v2_m_embed',
    alias='pos_spa_embed_res_v2_m_embed',
    scope='{}.*heads.*/pos/common_pos_embed'.format(name_scope)
  )
  _safe_collect(
    ep_key='pos_spa_embed_res_v2_m_expand_embed',
    alias='pos_spa_embed_res_v2_m_expand_embed',
    scope='{}.*heads.*/pos/common_pos_embed'.format(name_scope)
  )
  _safe_collect(
    ep_key='pos_spa_embed_res_v2_spa_embed',
    alias='pos_spa_embed_res_v2_spa_embed',
    scope='{}.*heads.*/pos/common_pos_embed.*'.format(name_scope)
  )
  _safe_collect(
    ep_key='pos_spa_embed_res_v2_out',
    alias='pos_spa_embed_res_v2_out',
    scope='{}.*heads.*/pos/common_pos_embed'.format(name_scope)
  )

  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))
  _safe_collect(
    ep_key='head_cmd_creep_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_creep_pos')
  )
  _safe_collect(
    ep_key='head_cmd_nydus_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_nydus_pos')
  )
  # lvl 4, others
  _safe_collect(ep_key='head_cmd_base_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_unload_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_unload'))

  return ep


def _make_mnet_v5d11_endpoints_dict(nc: MNetV6Config,
                                    endpoints_collections: str,
                                    name_scope: str):
  ep = OrderedDict()

  def _safe_collect(ep_key, scope, alias=None):
    o = tp_utils.find_tensors(endpoints_collections, scope, alias)
    if not o:
      return
    ep[ep_key] = o[0]
    if not nc.endpoints_norm:
      return
    # get its average l1-norm, i.e., the average neuron activation
    t = ep[ep_key]
    t_size = tp_utils.get_size(t)
    ep[ep_key] = tf.norm(tp_ops.to_float32(t), ord=1) / (t_size + 1e-8)

  # consts
  _safe_collect(ep_key='const_arg_mask', alias='const_arg_mask',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_mask_base_posese', alias='const_mask_base_posese',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base_poses', alias='const_base_poses',
                scope='{}.*{}.*'.format(name_scope, 'consts'))
  _safe_collect(ep_key='const_base', alias='const_base',
                scope='{}.*{}.*'.format(name_scope, 'consts'))

  # low-level embedding
  _safe_collect(ep_key='ab_emb', alias='shared/ab_embed',
                scope='{}.*{}.*'.format(name_scope, 'ab_embed'))
  _safe_collect(ep_key='noop_emb', alias='shared/noop_num_embed',
                scope='{}.*{}.*'.format(name_scope, 'noop_num_embed'))
  _safe_collect(ep_key='buff_emb', alias='shared/buff_embed',
                scope='{}.*{}.*'.format(name_scope, 'buff_embed'))
  _safe_collect(ep_key='order_emb', alias='shared/order_embed',
                scope='{}.*{}.*'.format(name_scope, 'order_embed'))
  _safe_collect(ep_key='u_type_emb', alias='shared/u_type_embed',
                scope='{}.*{}.*'.format(name_scope, 'u_type_embed'))
  # high-level embedding
  _safe_collect(ep_key='units_emb_blk', alias='units_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='selected_emb_blk', alias='selected_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='scatter_units_blk', alias='scatter_units',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding
  _safe_collect(ep_key='spa_emb_blk_img', alias='spa_embed_img',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  _safe_collect(ep_key='spa_emb_blk_vec', alias='spa_embed_vec',
                scope='{}.*{}.*'.format(name_scope, 'embed/'))
  # high-level spatial embedding: grid net
  _safe_collect(ep_key='gridnet_input_img', alias='x_img_input',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc1', alias='x_img_fc1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc2', alias='x_img_fc2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  _safe_collect(ep_key='gridnet_fc3', alias='x_img_fc3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc'))
  # large
  _safe_collect(ep_key='gridnet_large_c1', alias='large/conv1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c2', alias='large/conv2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_mc', alias='large/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_c3', alias='large/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  _safe_collect(ep_key='gridnet_large_rs', alias='large/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/large'))
  # medium
  _safe_collect(ep_key='gridnet_medium_p1', alias='medium/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_p2', alias='medium/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_mc', alias='medium/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_c3', alias='medium/conv3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  _safe_collect(ep_key='gridnet_medium_rs', alias='medium/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/medium'))
  # small
  _safe_collect(ep_key='gridnet_small_p1', alias='small/pool1',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p2', alias='small/pool2',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p3', alias='small/pool3',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_p4', alias='small/pool4',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_mc', alias='small/multi_conv',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_c5', alias='small/conv5',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  _safe_collect(ep_key='gridnet_small_rs', alias='small/resize',
                scope='{}.*{}.*'.format(name_scope, 'gridnet_enc/small'))
  # high-level vector embedding
  _safe_collect(ep_key='vec_emb_blk', alias='vec_enc',
                scope='{}.*{}.*'.format(name_scope, 'embed/vec_enc'))
  # high-level last action embedding
  _safe_collect(ep_key='last_action_blk', alias='last_action_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/last_action_embed'))
  # high-level integrate embedding
  _safe_collect(ep_key='inte_embed', alias='inte_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/inte_embed'))
  # LSTM (if any)
  _safe_collect(ep_key='lstm_emb_out', alias='lstm_embed_out',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  _safe_collect(ep_key='lstm_emb_hs', alias='lstm_embed_hs',
                scope='{}.*{}.*'.format(name_scope, 'embed/lstm_embed'))
  # zstat
  _safe_collect(ep_key='zstat_embed', alias='zstat_embed',
                scope='{}.*{}.*'.format(name_scope, 'embed/zstat_embed'))

  # pre-layer for the heads
  # lvl 1
  _safe_collect(ep_key='head_ab_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/ability'))
  # lvl 2
  _safe_collect(ep_key='head_noop_pre',
                alias='pre_action_res_temporal_smoothing',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/noop_num'))
  # lvl 3
  _safe_collect(ep_key='head_shift_pre', alias='pre_discrete_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/shift'))
  # lvl 4, select
  _safe_collect(ep_key='head_ss_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/s_select'))
  _safe_collect(ep_key='head_ms_pre', alias='pre_multinomial_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/m_select'))
  # lvl 4, cmd u
  _safe_collect(ep_key='head_cmd_u_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_u'))
  # lvl 4, positions
  _safe_collect(
    ep_key='pos_spa_embed_res_vold_m_embed',
    alias='pos_spa_embed_res_vold_m_embed',
    scope='{}.*heads.*/pos/common_pos_embed'.format(name_scope)
  )
  _safe_collect(
    ep_key='pos_spa_embed_res_vold_m_expand_embed',
    alias='pos_spa_embed_res_vold_m_expand_embed',
    scope='{}.*heads.*/pos/common_pos_embed'.format(name_scope)
  )
  _safe_collect(
    ep_key='pos_spa_embed_res_vold_out',
    alias='pos_spa_embed_res_vold_out',
    scope='{}.*heads.*/pos/common_pos_embed'.format(name_scope)
  )

  _safe_collect(ep_key='head_cmd_pos_pre', alias='pre_loc_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_pos'))
  _safe_collect(
    ep_key='head_cmd_creep_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_creep_pos')
  )
  _safe_collect(
    ep_key='head_cmd_nydus_pos_pre',
    alias='pre_loc_action_res',
    scope='{}.*{}.*'.format(name_scope, 'heads.*/pos/cmd_nydus_pos')
  )
  # lvl 4, others
  _safe_collect(ep_key='head_cmd_base_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_base'))
  _safe_collect(ep_key='head_cmd_unload_pre', alias='pre_ptr_action_res',
                scope='{}.*{}.*'.format(name_scope, 'heads.*/cmd_unload'))

  return ep


def _make_mnet_v5_loss_endpoints_dict(head_xe_loss):
  ep = OrderedDict((name, loss) for name, loss in head_xe_loss.items())
  for name, loss in head_xe_loss.items():
    ep[name] = loss


def _make_mnet_v5_compatible_params_from_v4(params_v4_order,
                                            param_names_v4_order,
                                            param_names_v5_order):
  v4_to_v5_idx = [param_names_v4_order.index(pn_v5)
                  for pn_v5 in param_names_v5_order]
  params_v5_order = [params_v4_order[idx] for idx in v4_to_v5_idx]
  return params_v5_order