# Copyright 2020 Google Inc.
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
"""Utilities for processing segmentation data."""

from typing import Sequence

import numpy as np
from scipy import ndimage
import scipy.sparse
import skimage.morphology


def relabel(labels: np.ndarray, orig_ids, new_ids):
  """Relabels `labels` by mapping `orig_ids` to `new_ids`.

  Args:
    labels: ndarray of segment IDs
    orig_ids: iterable of existing segment IDs
    new_ids: iterable of new segment IDs (len(new_ids) == len(orig_ids))

  Returns:
    int64 ndarray with updated segment IDs
  """
  orig_ids = np.asarray(orig_ids)
  new_ids = np.asarray(new_ids)
  assert orig_ids.size == new_ids.size

  # A sparse matrix is required so that arbitrarily large IDs can be used as
  # input. The first dimension of the matrix is dummy and has a size of 1 (the
  # first coordinate is fixed at 0). This is necessary because only 2d sparse
  # arrays are supported.
  row_indices = np.zeros_like(orig_ids)
  col_indices = orig_ids

  relabel_mtx = scipy.sparse.csr_matrix((new_ids, (row_indices, col_indices)),
                                        shape=(1, int(col_indices.max()) + 1))
  # Index with a 2D array so that the output is a sparse matrix.
  labels2d = labels.reshape(1, labels.size)
  relabeled = relabel_mtx[0, labels2d]
  return relabeled.toarray().reshape(labels.shape)


def make_labels_contiguous(labels: np.ndarray) -> np.ndarray:
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
  orig_ids = np.unique(np.append(labels, np.uint64(0)))
  new_ids = np.arange(len(orig_ids))
  return relabel(labels, orig_ids, new_ids), list(zip(orig_ids, new_ids))


def watershed_expand(seg: np.ndarray,
                     voxel_size: Sequence[float],
                     max_distance: float = None,
                     mask: np.ndarray = None):
  """Grows existings segments using watershed.

  All segments are grown at an uniform rate, using the Euclidean distance
  transform of the empty space of the input segmentation. This results in
  all empty voxels getting assigned the ID of the nearest segment, up to
  `max_distance`.

  Args:
    seg: 3d int ZYX array of segmentation data
    voxel_size: x, y, z voxel size in nm
    max_distance: max distance in nm to expand the seeds
    mask: 3d bool array of the same shape as `seg`; positive values define the
      region where watershed will be applied. If not specified, wastershed is
      applied everywhere in the subvolume.

  Returns:
    expanded segmentation, distance transform over the empty space
    of the original segmentation prior to expansion
  """
  # Map to low IDs for watershed to work.
  seg_low, orig_to_low = make_labels_contiguous(seg)
  edt = ndimage.distance_transform_edt(seg_low == 0, sampling=voxel_size[::-1])

  if mask is None:
    mask = np.ones(seg_low.shape, dtype=np.bool)

  if max_distance is not None:
    mask[edt > max_distance] = False

  ws = skimage.morphology.watershed(edt, seg_low, mask=mask).astype(np.uint64)

  # Restore any segment parts that might have been removed by the mask.
  nmask = np.logical_not(mask)
  if np.any(nmask):
    ws[nmask] = seg_low[nmask]

  orig_ids, low_ids = zip(*orig_to_low)
  return relabel(ws, np.array(low_ids), np.array(orig_ids)), edt
