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
"""Routines for manipulating numpy arrays of segmentation data."""

from collections import Counter

import numpy as np
import scipy.sparse
import skimage.measure


# Monkey patch fix for indexing overflow problems with 64 bit IDs.
# See also:
# http://scipy-user.10969.n7.nabble.com/SciPy-User-strange-error-when-creating-csr-matrix-td20129.html
# https://github.com/scipy/scipy/pull/4678
if scipy.__version__ in ('0.14.0', '0.14.1', '0.15.1'):
  def _get_index_dtype(*unused_args, **unused_kwargs):
    return np.int64
  scipy.sparse.compressed.get_index_dtype = _get_index_dtype
  scipy.sparse.csr.get_index_dtype = _get_index_dtype
  scipy.sparse.csc.get_index_dtype = _get_index_dtype
  scipy.sparse.bsr.get_index_dtype = _get_index_dtype


def make_labels_contiguous(labels):
  """Relabels 'labels' so that its ID space is dense.

  If N is the number of unique ids in 'labels', the new IDs will cover the range
  [0..N-1].

  Args:
    labels: ndarray of segment IDs

  Returns:
    tuple of:
      ndarray of dense segment IDs
      list of (old_id, new_id) pairs
  """
  orig_ids = np.unique(labels)
  new_ids = np.arange(len(orig_ids))
  # A sparse matrix is required so that arbitrarily large IDs can be used as
  # input. The first dimension of the matrix is dummy and has a size of 1 (the
  # first coordinate is fixed at 0).
  row_indices = np.zeros_like(orig_ids)
  col_indices = orig_ids
  relabel = scipy.sparse.csr_matrix((new_ids, (row_indices, col_indices)))
  # Index with a 2D array so that the output is a sparse matrix.
  labels2d = labels.reshape(1, labels.size)
  relabeled = relabel[0, labels2d]
  return relabeled.toarray().reshape(labels.shape), zip(orig_ids, new_ids)


def clear_dust(data, min_size=10):
  """Removes small objects from a segmentation array.

  Replaces objects smaller than `min_size` with 0 (background).

  Args:
    data: numpy array of segment IDs
    min_size: minimum size in voxels of an object to be retained

  Returns:
    the data array (modified in place)
  """
  ids, sizes = np.unique(data, return_counts=True)
  small = ids[sizes < min_size]
  small_mask = np.in1d(data.flat, small).reshape(data.shape)
  data[small_mask] = 0
  return data


def reduce_id_bits(segmentation):
  """Reduces the number of bits used for IDs.

  Assumes that one additional ID beyond the max of 'segmentation' is necessary
  (used by GALA to mark boundary areas).

  Args:
    segmentation: ndarray of int type

  Returns:
    segmentation ndarray converted to minimal uint type large enough to keep
    all the IDs.
  """
  max_id = segmentation.max()
  if max_id <= np.iinfo(np.uint8).max:
    return segmentation.astype(np.uint8)
  elif max_id <= np.iinfo(np.uint16).max:
    return segmentation.astype(np.uint16)
  elif max_id <= np.iinfo(np.uint32).max:
    return segmentation.astype(np.uint32)


def split_disconnected_components(labels):
  """Relabels the connected components of a 3-D integer array.

  Connected components are determined based on 6-connectivity, where two
  neighboring positions are considering part of the same component if they have
  identical labels.

  The label 0 is treated specially: all positions labeled 0 in the input are
  labeled 0 in the output, regardless of whether they are contiguous.

  Connected components of the input array (other than segment id 0) are given
  consecutive ids in the output, starting from 1.

  Args:
    labels: 3-D integer numpy array.

  Returns:
    The relabeled numpy array, same dtype as `labels`.
  """
  has_zero = 0 in labels
  fixed_labels = skimage.measure.label(labels, connectivity=1, background=0)
  if has_zero or (not has_zero and 0 in fixed_labels):
    if np.any((fixed_labels == 0) != (labels == 0)):
      fixed_labels[...] += 1
      fixed_labels[labels == 0] = 0
  return np.cast[labels.dtype](fixed_labels)


def clean_up(seg, split_cc=True, min_size=0, return_id_map=False):  # pylint: disable=invalid-name
  """Runs connected components and removes small objects.

  Args:
    seg: segmentation to clean as a uint64 ndarray
    split_cc: whether to recompute connected components
    min_size: connected components smaller that this value get
        removed from the segmentation; if 0, no filtering by size is done
    return_id_map: whether to compute and return a map from new IDs
        to original IDs

  Returns:
    None if not return_id_map, otherwise a dictionary mapping
    new IDs to original IDs. `seg` is modified in place.
  """
  if return_id_map:
    seg_orig = seg.copy()

  if split_cc:
    seg[...] = split_disconnected_components(seg)
  if min_size > 0:
    clear_dust(seg, min_size)

  if return_id_map:
    cc_ids, cc_idx = np.unique(seg.ravel(), return_index=True)
    orig_ids = seg_orig.ravel()[cc_idx]
    cc_to_orig = dict(zip(cc_ids, orig_ids))
    return cc_to_orig


def split_segmentation_by_intersection(a, b, min_size):
  """Computes the intersection of two segmentations.

  Intersects two spatially overlapping segmentations and assigns a new ID to
  every unique (id1, id2) pair of overlapping voxels. If 'id2' is the largest
  object overlapping 'id1', their intersection retains the 'id1' label. If the
  fragment created by intersection is smaller than 'min_size', it gets removed
  from the segmentation (assigned an id of 0 in the output).

  `a` is modified in place, `b` is not changed.

  Note that (id1, 0) is considered a valid pair and will be mapped to a non-zero
  ID as long as the size of the overlapping region is >= min_size, but (0, id2)
  will always be mapped to 0 in the output.

  Args:
    a: First segmentation.
    b: Second segmentation.
    min_size: Minimum size intersection segment to keep (not map to 0).

  Raises:
    TypeError: if a or b don't have a dtype of uint64

    ValueError: if a.shape != b.shape, or if `a` or `b` contain more than
                2**32-1 unique labels.
  """
  if a.shape != b.shape:
    raise ValueError
  a = a.ravel()
  output_array = a

  b = b.ravel()

  def remap_input(x):
    """Remaps `x` if needed to fit within a 32-bit ID space.

    Args:
      x: uint64 numpy array.

    Returns:
      `remapped, max_id, orig_values_map`, where:

        `remapped` contains the remapped version of `x` containing only
        values < 2**32.

        `max_id = x.max()`.

        `orig_values_map` is None if `remapped == x`, or otherwise an array such
        that `x = orig_values_map[remapped]`.
    Raises:
      TypeError: if `x` does not have uint64 dtype
      ValueError: if `x.max() > 2**32-1`.
    """
    if x.dtype != np.uint64:
      raise TypeError
    max_uint32 = 2**32 - 1
    max_id = x.max()
    orig_values_map = None
    if max_id > max_uint32:
      orig_values_map, x = np.unique(x, return_inverse=True)
      if len(orig_values_map) > max_uint32:
        raise ValueError('More than 2**32-1 unique labels not supported')
      x = np.cast[np.uint64](x)
      if orig_values_map[0] != 0:
        orig_values_map = np.concatenate(
            [np.array([0], dtype=np.uint64), orig_values_map])
        x[...] += 1
    return x, max_id, orig_values_map

  remapped_a, max_id, a_reverse_map = remap_input(a)
  remapped_b, _, _ = remap_input(b)

  intersection_segment_ids = np.bitwise_or(remapped_a, remapped_b << 32)

  unique_joint_labels, remapped_joint_labels, joint_counts = np.unique(
      intersection_segment_ids, return_inverse=True, return_counts=True)

  unique_joint_labels_a = np.bitwise_and(unique_joint_labels, 0xFFFFFFFF)
  unique_joint_labels_b = unique_joint_labels >> 32

  # Maps each segment id `id_a` in `remapped_a` to `(id_b, joint_count)` where
  # `id_b` is the segment id in `remapped_b` with maximum overlap, and
  # `joint_count` is the number of voxels of overlap.
  max_overlap_ids = dict()

  for label_a, label_b, count in zip(unique_joint_labels_a,
                                     unique_joint_labels_b, joint_counts):
    new_pair = (label_b, count)
    existing = max_overlap_ids.setdefault(label_a, new_pair)
    if existing[1] < count:
      max_overlap_ids[label_a] = new_pair

  # Relabel map to apply to remapped_joint_labels to obtain the output ids.
  new_labels = np.zeros(len(unique_joint_labels), np.uint64)
  for i, (label_a, label_b, count) in enumerate(zip(unique_joint_labels_a,
                                                    unique_joint_labels_b,
                                                    joint_counts)):
    if count < min_size or label_a == 0:
      new_label = 0
    elif label_b == max_overlap_ids[label_a][0]:
      if a_reverse_map is not None:
        new_label = a_reverse_map[label_a]
      else:
        new_label = label_a
    else:
      max_id += 1
      new_label = max_id
    new_labels[i] = new_label

  output_array[...] = new_labels[remapped_joint_labels]
