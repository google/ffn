# Copyright 2017-2023 Google Inc.
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

from connectomics.segmentation import labels
import numpy as np


def clear_dust(data: np.ndarray, min_size: int = 10):
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


def reduce_id_bits(segmentation: np.ndarray):
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
  return segmentation


def clean_up(seg: np.ndarray,
             split_cc=True,
             connectivity=1,
             min_size=0,
             return_id_map=False):
  """Runs connected components and removes small objects.

  Args:
    seg: segmentation to clean as a uint64 ndarray
    split_cc: whether to recompute connected components
    connectivity: used for split_cc; 1, 2, or 3; for 6-, 18-, or 26-connectivity
      respectively.
    min_size: connected components smaller that this value get removed from the
      segmentation; if 0, no filtering by size is done
    return_id_map: whether to compute and return a map from new IDs to original
      IDs

  Returns:
    None if not return_id_map, otherwise a dictionary mapping
    new IDs to original IDs. `seg` is modified in place.
  """
  cc_to_orig, _ = clean_up_and_count(
      seg,
      split_cc,
      connectivity,
      min_size,
      compute_id_map=return_id_map,
      compute_counts=False)
  if return_id_map:
    return cc_to_orig


def clean_up_and_count(seg: np.ndarray,
                       split_cc=True,
                       connectivity=1,
                       min_size=0,
                       compute_id_map=True,
                       compute_counts=True):
  """Runs connected components and removes small objects, returns metadata.

  Args:
    seg: segmentation to clean as a uint64 ndarray.  Mutated in place.
    split_cc: whether to recompute connected components
    connectivity: used for split_cc; 1, 2, or 3; for 6-, 18-, or 26-connectivity
      respectively.
    min_size: connected components smaller that this value get removed from the
      segmentation; if 0, no filtering by size is done
    compute_id_map: whether to compute a mapping of new CC ID to old ID.  If
      False, None is returned instead.
    compute_counts: whether to compute a mapping of new CC ID to voxel count. If
      False, None is returned instead.

  Returns:
    tuple of (dict of new ID to original ID, dict of new ID to voxel count).  If
    compute_id_map or compute_counts is False, the respective returned tuple
    member will be None.
  """
  if compute_id_map:
    seg_orig = seg.copy()

  if split_cc:
    seg[...] = labels.split_disconnected_components(seg, connectivity)
  if min_size > 0:
    clear_dust(seg, min_size)

  cc_to_orig, cc_to_count = None, None

  if compute_id_map or compute_counts:
    unique_result_tuple = np.unique(
        seg.ravel(), return_index=compute_id_map, return_counts=compute_counts)
    cc_ids = unique_result_tuple[0]
  if compute_id_map:
    cc_idx = unique_result_tuple[1]
    orig_ids = seg_orig.ravel()[cc_idx]
    cc_to_orig = dict(zip(cc_ids, orig_ids))
  if compute_counts:
    cc_counts = unique_result_tuple[-1]
    cc_to_count = dict(zip(cc_ids, cc_counts))

  return cc_to_orig, cc_to_count


def split_segmentation_by_intersection(a: np.ndarray, b: np.ndarray,
                                       min_size: int):
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
  for i, (label_a, label_b, count) in enumerate(
      zip(unique_joint_labels_a, unique_joint_labels_b, joint_counts)):
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
