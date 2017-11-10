# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple augmentation operations for volumetric EM data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def reflection(data, decision):
  """Conditionally reflects the data in XYZ.

  Args:
    data: input tensor, shape: [..], z, y, x, c
    decision: boolean tensor, shape 3, indicating on which spatial dimensions
       to apply the reflection (x, y, z)

  Returns:
    TF op to conditionally apply reflection.
  """
  with tf.name_scope('augment_reflection'):
    rank = data.get_shape().ndims
    spatial_dims = tf.constant([rank - 2, rank - 3, rank - 4])
    selected_dims = tf.boolean_mask(spatial_dims, decision)
    return tf.reverse(data, selected_dims)


def xy_transpose(data, decision):
  """Conditionally transposes the X and Y axes of a tensor.

  Args:
    data: input tensor, shape: [..], y, x, c.
    decision: boolean scalar indicating whether to apply the transposition

  Returns:
    TF op to conditionally apply XY transposition.
  """
  with tf.name_scope('augment_xy_transpose'):
    rank = data.get_shape().ndims
    perm = range(rank)
    perm[rank - 3], perm[rank - 2] = perm[rank - 2], perm[rank - 3]
    return tf.cond(decision,
                   lambda: tf.transpose(data, perm),
                   lambda: data)


def permute_axes(x, permutation, permutable_axes):
  """Permutes the axes of `x` using the specified permutation.

  All axes not in `permutable_axes` must be identity mapped by `permutation`:
    `permutation[i] = i   if i not in permutable_axes`

  Args:
    x: Tensor of rank N to permute.
    permutation: Rank 1 tensor of shape `[N]`.
    permutable_axes: List (not Tensor) of axes that may be permuted.

  Returns:
    Permuted Tensor obtained by calling:
      `tf.transpose(x, permutation)`
    but with additional static shape information due to the restriction imposed
    by `permutable_axes`.
  """
  x = tf.convert_to_tensor(x)
  shape = x.shape.as_list()
  result = tf.transpose(x, permutation)
  # Transpose loses shape information because it does not know that the
  # permutation is restricted to the permutable_axes, but we can restore
  # some or all of it.

  # If the static dimensions of all of the permutable axes are not the
  # same, then set them all to None.  Otherwise, they can stay as is.
  if len({shape[i] for i in permutable_axes}) > 1:
    for i in permutable_axes:
      shape[i] = None

  result.set_shape(shape)
  return result


class PermuteAndReflect(object):
  """Class for performing random permutation and reflection of axes.

  Consructing an instance of this class adds tensors to the default Tensorflow
  graph representing a randomly sampled permutation of `permutable_axes` and
  randomly sampled reflection decisions for `reflectable_axes`.

  Calling an instance of this class as a function (i.e. invoking the `__call__`
  method) applies the sampled transformations to a specified `Tensor`.

  Attributes:
    rank: The rank of Tensor that can be transformed.
    permutable_axes: 1-D int32 numpy array specifying the axes that may be
      permuted.
    reflectable_axes: 1-D int32 numpy array specifying the axes that may be
      reflected.
    reflect_decisions: bool Tensor of shape `[len(reflectable_axes)]` containing
      the sampled reflection decision for each axis in `reflectable_axes`.
    reflected_axes: Rank 1 int32 Tensor specifying the axes to be reflected
      (i.e. corresponding to a True value in `reflect_decisions`).
    permutation: int32 Tensor of shape `[len(permutable_axes)]` containing the
      sampled permutation of `permutable_axes`.
    full_permutation: int32 Tensor of shape `[rank]` that extends `permutation`
      to be a permutation of all `rank` axes, where axes not in
      `permutable_axes` are identity mapped.
  """

  def __init__(self, rank, permutable_axes, reflectable_axes,
               permutation_seed=None, reflection_seed=None):
    """Initializes the transformation nodes.

    Args:
      rank: The rank of the Tensor to be transformed.
      permutable_axes: The list (not a Tensor) of axes to be permuted.
      reflectable_axes: The list (not a Tensor) of axes to be reflected.
      permutation_seed: Optional integer.  Seed value to use for sampling axes
        permutation.
      reflection_seed: Optional integer.  Seed value to use for sampling
        reflection decisions.
    Raises:
      ValueError: if arguments are invalid.
    """
    self.rank = rank
    if len(set(permutable_axes)) != len(permutable_axes):
      raise ValueError('permutable_axes must not contain duplicates')
    if len(set(reflectable_axes)) != len(reflectable_axes):
      raise ValueError('reflectable_axes must not contain duplicates')
    if not all(0 <= x < rank for x in permutable_axes):
      raise ValueError('permutable_axes must be a subset of [0, rank-1].')
    if not all(0 <= x < rank for x in reflectable_axes):
      raise ValueError('reflectable_axes must be a subset of [0, rank-1].')
    # Cast to int32 to ensure the proper tensorflow types are used.
    self.permutable_axes = np.array(permutable_axes, dtype=np.int32)
    self.reflectable_axes = np.array(reflectable_axes, dtype=np.int32)

    if self.reflectable_axes.size > 0:
      self.reflect_decisions = tf.random_uniform([len(self.reflectable_axes)],
                                                 seed=reflection_seed) > 0.5
      self.reflected_axes = tf.boolean_mask(self.reflectable_axes,
                                            self.reflect_decisions)

    if self.permutable_axes.size > 0:
      self.permutation = tf.random_shuffle(self.permutable_axes,
                                           seed=permutation_seed)
      # full_permutation must be a list rather than an np.array of int32 because
      # some elements are set to be tensors below.
      full_permutation = [np.int32(x) for x in range(rank)]
      for i, d in enumerate(self.permutable_axes):
        full_permutation[d] = self.permutation[i]
      self.full_permutation = tf.stack(full_permutation)

  def __call__(self, x):
    """Applies the sampled permutation and reflection to `x`.

    Args:
      x: A Tensor of rank `self.rank`.

    Returns:
      The transformed Tensor, retaining as much static shape information as
      possible.
    """
    x = tf.convert_to_tensor(x)
    with tf.name_scope('permute_and_reflect'):
      if self.permutable_axes.size > 0:
        x = permute_axes(x, self.full_permutation, self.permutable_axes)
      if self.reflectable_axes.size > 0:
        x = tf.reverse(x, self.reflected_axes)
      return x
