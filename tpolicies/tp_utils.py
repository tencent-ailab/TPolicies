from collections import OrderedDict

import tensorflow as tf

from gym.spaces import Space
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import Box
from gym.spaces import MultiBinary
from gym.spaces import Tuple as GymTuple
from gym.spaces import Dict as GymDict


# placeholders stuff
def placeholders_from_gym_space(space: Space, batch_size=None, name='Ob'):
  """Make placeholders from gym space.

  Implemented as pre-order, depth-first recursion of the space tree.

  Args:
    space: a gym.spaces.Space
    batch_size: optional batch_size
    name: optional placeholder name

  Returns:
    A placeholders collection that is strictly organized as the space
    specifies, i.e., it recursively applies that
    * Discrete -> Tensor shape (bs,) + space.shape where caller assures each
    item be in  range(space.n)
    * MultiDiscrete -> Tensor shape: (bs,) + space.shape where caller assures
    each item in range(space.nvec[i])
    * Box -> Tensor in shape (bs,) + space.shape
    * GymTuple -> a tuple() of Tensors
    * GymDict -> a OrderedDict() of Tensors
  """
  return map_gym_space_to_structure(
    func=lambda x_sp: tf.placeholder(shape=(batch_size,) + x_sp.shape,
                                     dtype=x_sp.dtype,
                                     name=name),
    gym_sp=space
  )


# tf.nest helpers
def template_structure_from_gym_space(space: Space, filled=None):
  """Create a tf.nest compatible template structure from given gym space.

  The created template structure can be used as the `shallow_tree` arg for
   tf.nest.map_structure_up_to

  Args:
    space: gym space
    filled: what to fill in each structure entry.

  Returns:
    A tf.nest compatible structure.
  """
  return map_gym_space_to_structure(lambda x: filled, space)


def map_gym_space_to_structure(func, gym_sp):
  """Map gym space to a structure using given function.

  NOTE:
    Similar to tf.nest.map_structure, but the input is a gym space, the output
     is a tf.nest.map_structure compatible structure

  Args:
    func:
    gym_sp:

  Returns:
    A structure
  """
  if type(gym_sp) in [Discrete, MultiDiscrete, Box, MultiBinary]:
    return func(gym_sp)
  if isinstance(gym_sp, GymTuple):
    return tuple([map_gym_space_to_structure(func, sp) for sp in gym_sp.spaces])
  if isinstance(gym_sp, GymDict):
    # GymDict assures an internal OrderedDict
    return OrderedDict([
      (each_name, map_gym_space_to_structure(func, each_sp))
      for each_name, each_sp in zip(gym_sp.spaces.keys(),
                                    gym_sp.spaces.values())
    ])
  raise NotImplementedError('Unsupported gym space: {}'.format(gym_sp))


def pack_sequence_as_structure_like_gym_space(gym_sp, flat_seq):
  """Pack sequence as tf.nest compatible structure like the given gym space.

  NOTE:
    It respects the gym.spaces.Dict (which is an OrderedDict internally) order,
      which is different from the behaviour of tf.nest.pack_sequence_as, be
      cautious!

  Args:
    gym_sp:
    flat_seq:

  Returns:
    A structure filled with `flat_seq` in order.
  """
  if type(gym_sp) in [Discrete, MultiDiscrete, Box, MultiBinary]:
    return flat_seq
  if isinstance(gym_sp, GymTuple):
    return tuple([pack_sequence_as_structure_like_gym_space(sp, elem)
                  for sp, elem in zip(gym_sp.spaces, flat_seq)])
  if isinstance(gym_sp, GymDict):
    # GymDict assures an internal OrderedDict
    return OrderedDict([
      (key, pack_sequence_as_structure_like_gym_space(sp, elem))
      for key, sp, elem in zip(gym_sp.spaces.keys(), gym_sp.spaces.values(),
                               flat_seq)
    ])
  raise NotImplementedError('Unsupported gym space: {}'.format(gym_sp))


# tf.get_collection helpers
def find_tensors(collection, scope=None, alias=None):
  """Find Tensors by collection, scope, alias.

  See the two functions `collect_named_outputs` and `append_tensor_alias` from
    `tf.contrib.layers`. A `Tensor` can have the `.aliases` fields for quick
    finding/reference. This aliases mechanism is used by, for example, the
     `tf.contribe.layers` when collecting the layer output tensor.

  Args:
    collection:
    scope:
    alias: str, `Tensor`'s alias.

  Returns:
    A list of found `Tensor`s
  """
  ts = tf.get_collection(collection, scope)
  if alias is None:
    return ts

  # find by alias
  def _matched(name1, name2):
    return name1 in name2 or name2 in name1

  ats = []
  for tensor in ts:
    if not hasattr(tensor, 'aliases'):
      continue
    if any([_matched(alias, a) for a in tensor.aliases]):
      ats.append(tensor)
  return ats


# miscellaneous tf utilities
def get_size(t: tf.Tensor) -> int:
  """Get the size of the input Tensor t.

  The size is the product of each dim of t. A scalar is deemed as size 1.
  """
  shape = tf.shape(t)
  return tf.cast(tf.math.reduce_prod(shape), tf.float32)