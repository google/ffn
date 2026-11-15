# Copyright 2020-2023 Google Inc.
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
"""Utilities for identifying and processing decision points."""

import itertools
from typing import Optional, Sequence

from connectomics.common import bounding_box
from connectomics.segmentation import labels
from ffn.inference import segmentation as segmentation_lib
import numpy as np
import pandas as pd


def find_decision_points(
    seg: np.ndarray,
    voxel_size: Sequence[float],
    max_distance: Optional[float] = None,
    subvol_box: Optional[bounding_box.BoundingBox] = None,
    optimize_sparse: bool = False,
    sparse_noise_threshold: int = 0,
) -> dict[tuple[int, int], tuple[float, np.ndarray]]:
  """Identifies decision points in a segmentation subvolume.

  Args:
    seg: 3d uint64 ndarray of segmentation data
    voxel_size: 3-tuple (xyz) defining the physical voxel size
    max_distance: maximum distance between the segment and the decision point
      (same units as voxel_size); if None, distances will not be limited
    subvol_box: selector for a subvolume within `seg` within which to search for
      decision points; the whole subvolume is always used to compute the
      distance transform
    optimize_sparse: if True, first counts the number of segments in `seg` and
      returns early if there are fewer than 2.
    sparse_noise_threshold: if > 0 and `optimize_sparse` is True, ignores
      components with voxel counts < this threshold when counting segments.

  Returns:
    dict from segment ID pairs to tuples of:
      approximate physical distance from the segment to the decision point
      (x, y, z) decision point
  """
  if optimize_sparse:
    _, counts = segmentation_lib.clean_up_and_count(
        seg,
        split_cc=False,
        min_size=sparse_noise_threshold,
        compute_id_map=False,
    )

    if counts is not None and len([k for k in counts.keys() if k > 0]) <= 1:
      # If there are 0 or 1 unique segments (excluding background),
      # they cannot possibly touch another segment.
      return {}

  # EDT is the Euclidean Distance Transform, specifying how far voxels added
  # in 'expanded_seg' are from the seeds in 'seg'.
  expanded_seg, edt = labels.watershed_expand(seg, voxel_size, max_distance)
  if subvol_box is not None:
    expanded_seg = expanded_seg[subvol_box.to_slice3d()]
    edt = edt[subvol_box.to_slice3d()]

  a_list = []
  b_list = []
  dist_list = []
  x_list = []
  y_list = []
  z_list = []

  # Need to examine 7 offsets to identify all possible connections within a
  # 3x3x3 neighborhood.
  for off in itertools.product((0, -1), (0, -1), (0, -1)):
    if off == (0, 0, 0):
      continue

    # Slicing optimization
    slice_a = []
    slice_b = []
    for o in off:
      if o == 0:
        slice_a.append(slice(None))
        slice_b.append(slice(None))
      elif o == -1:
        slice_a.append(slice(0, -1))
        slice_b.append(slice(1, None))
    slice_a = tuple(slice_a)
    slice_b = tuple(slice_b)

    a_part = expanded_seg[slice_a]
    b_part = expanded_seg[slice_b]
    touching = (a_part > 0) & (b_part > 0) & (a_part != b_part)
    if not np.any(touching):
      continue

    mean_edt = (edt[slice_a][touching] + edt[slice_b][touching]) / 2

    # Enforce standard ID order within the pair (low, hi).
    ab = np.array([a_part[touching], b_part[touching]], dtype=np.uint64)
    ab.sort(axis=0)

    z, y, x = np.where(touching)
    a_list.append(ab[0, :])
    b_list.append(ab[1, :])
    dist_list.append(mean_edt)
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)

  if not a_list:
    return {}

  # Find points with the minimum distance.
  df = pd.DataFrame({
      'a': np.concatenate(a_list),
      'b': np.concatenate(b_list),
      'dist': np.concatenate(dist_list),
      'x': np.concatenate(x_list),
      'y': np.concatenate(y_list),
      'z': np.concatenate(z_list),
  })
  min_points = df[df.groupby(['a', 'b'])['dist'].transform('min') == df['dist']]

  ret = {}
  # For every pair of objects, select a single point with the minimum distance.
  for (a, b), data in min_points.groupby(['a', 'b']):
    points = np.array(data[['x', 'y', 'z']])
    dist = np.array(data['dist'])[0]
    # Find point located closest to the centroid of all points with min.
    # distance.
    idx = np.argmin(np.sum(np.square(points - np.mean(points, axis=0)), axis=1))
    ret[(a, b)] = (dist, points[idx])

  return ret
