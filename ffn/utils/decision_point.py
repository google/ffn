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
"""Utilities for identifying and processing decision points."""

import itertools
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from . import segmentation
from . import bounding_box


def find_decision_points(seg: np.ndarray,
                         voxel_size: Sequence[float],
                         max_distance: float = None,
                         subvol_box: bounding_box.BoundingBox = None
                        ) -> Dict[Tuple[int, int], Tuple[float, np.ndarray]]:
  """Identifies decision points in a segmentation subvolume.

  Args:
    seg: 3d uint64 ndarray of segmentation data
    voxel_size: 3-tuple (xyz) defining the physical voxel size
    max_distance: maximum distance between the segment and the decision point
      (same units as voxel_size); if None, distances will not be limited
    subvol_box: selector for a subvolume within `seg` within which
      to search for decision points; the whole subvolume is always used
      to compute the distance transform

  Returns:
    dict from segment ID pairs to tuples of:
      approximate physical distance from the segment to the decision point
      (x, y, z) decision point
  """
  # EDT is the Euclidean Distance Transform, specifying how far voxels added
  # in 'expanded_seg' are from the seeds in 'seg'.
  expanded_seg, edt = segmentation.watershed_expand(seg, voxel_size,
                                                    max_distance)
  if subvol_box is not None:
    expanded_seg = expanded_seg[subvol_box.to_slice()]
    edt = edt[subvol_box.to_slice()]

  a = expanded_seg
  dataframes = []

  # Need to examine 7 offsets to identify all possible connections within a
  # 3x3x3 neighborhood.
  for off in itertools.product((0, -1), (0, -1), (0, -1)):
    if off == (0, 0, 0):
      continue

    b = ndimage.shift(expanded_seg, off, order=0)
    touching = (a > 0) & (b > 0) & (a != b)
    if not np.any(touching):
      continue

    edt2 = np.roll(edt, off, (0, 1, 2))
    mean_edt = (edt[touching] + edt2[touching]) / 2

    # Enforce standard ID order within the pair (low, hi).
    ab = np.array([a[touching], b[touching]], dtype=np.uint64)
    ab.sort(axis=0)

    z, y, x = np.where(touching)
    dataframes.append(
        pd.DataFrame({
            'a': ab[0, :],
            'b': ab[1, :],
            'dist': mean_edt,
            'x': x,
            'y': y,
            'z': z
        }))

  if not dataframes:
    return {}

  # Find points with the minimum distance.
  df = pd.concat(dataframes)
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
