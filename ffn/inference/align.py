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
"""Classes to support ad-hoc alignment for inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from ..utils import bounding_box


class Alignment(object):
  """Base class to represent local ad-hoc alignment of Subvolumes.

  This base class implements an identity / no-op alignment.
  """

  def __init__(self, corner, size):
    """Initializes the alignment.

    Args:
      corner: the lower bound of the region from which this alignment is derived
      size: the size of the region from which this alignment is derived
    """
    self._corner = corner
    self._size = size

  @property
  def corner(self):
    return self._corner

  @property
  def size(self):
    return self._size

  def expand_bounds(self, corner, size, forward=True):
    """Computes the bounds that will contain the given bounds after alignment.

    Args:
      corner: (z, y, x) bounding box corner
      size: (z, y, x) bounding box size
      forward: True for forward transformation, False for inverse

    Returns:
      A tuple of (corner, size), the minimum bounds that will contain the voxels
      in the input bounding box after they've been transformed by this
      alignment.
    """
    del forward  # forward and inverse are identical for identity transforms
    return corner, size

  def transform_shift_mask(self, corner, scale, mask):
    """Transforms a shift mask by this alignment.

    Args:
      corner: (z, y, x) mask corner in un-scaled coordinates
      scale: the scale factor between full-scale raw coordinates and mask
          coordinates
      mask: a 4-d numpy array representing the shift field. Axis 0 should have
          at least two indices corresponding to y then x.

    Returns:
      The mask, approximately as it would have been had it been computed post-
      alignment.
    """
    # For identity transforms, this is just a pass-through.
    del corner, scale
    return mask

  def align_and_crop(self,
                     src_corner,
                     source,
                     dst_corner,
                     dst_size,
                     fill=0,
                     forward=True):
    """Aligns the subvolume and crops to a bounding box.

    Args:
      src_corner: (z, y, x) corner of the source subvolume
      source: a three dimensional numpy array to align
      dst_corner: (z, y, x) corner of the output subvolume
      dst_size: (z, y, x) size of the output subvolume
      fill: the value to assign missing data.
      forward: True for forward transformation, False for inverse

    Returns:
      An aligned subvolume with the requested geometry. Regions in the output
      that do not correspond post-transformation to voxels in the subvolume are
      assigned the fill value.
    """
    del forward  # forward and inverse are identical for identity transforms

    # If the source and destination geometries are the same, just return source
    if np.all(np.array(src_corner) == np.array(dst_corner)) and np.all(
        np.array(source.shape) == np.array(dst_size)):
      return source

    # Otherwise, use fill value for OOB regions.
    destination = np.full(dst_size, fill, dtype=source.dtype)

    zyx_offset = np.array(src_corner) - np.array(dst_corner)
    src_size = np.array(source.shape)
    dst_beg = np.clip(zyx_offset, 0, dst_size).astype(np.int)
    dst_end = np.clip(dst_size, 0, src_size + zyx_offset).astype(np.int)
    src_beg = np.clip(-zyx_offset, 0, src_size).astype(np.int)
    src_end = np.clip(src_size, 0, dst_size - zyx_offset).astype(np.int)

    if np.any(dst_end - dst_beg == 0) or np.any(src_end - src_beg == 0):
      return destination

    destination[dst_beg[0]:dst_end[0],
                dst_beg[1]:dst_end[1],
                dst_beg[2]:dst_end[2]] = source[src_beg[0]:src_end[0],
                                                src_beg[1]:src_end[1],
                                                src_beg[2]:src_end[2]]
    return destination

  def transform(self, zyx, forward=True):
    """Transforms a set of 3d points.

    Args:
      zyx: a numpy array with shape [3 n]. The first axis is ordered (z, y, x)
      forward: True for forward transformation, False for inverse

    Returns:
      transformed coordinates.
    """
    del forward  # forward and inverse are identical for identity transforms
    return zyx

  def rescaled(self, zyx_scale):
    """Return a rescaled copy of the alignment.

    Args:
      zyx_scale: the relative amount to scale the alignment.

    Returns:
      a new alignment with the given rescaling.
    """
    zyx_scale = np.array(zyx_scale)
    return Alignment(zyx_scale * self.corner, zyx_scale * self.size)


class Aligner(object):
  """Base class to represent local ad-hoc alignment generators.

  An Aligner is responsible for generating an Alignment that is valid for some
  local region.

  This base class just returns identity / no-op alignments.
  """

  def generate_alignment(self, corner, size):
    """Generates an alignment local to the given bounding box.

    Args:
      corner: (zyx) the lower bound of the bounding box
      size: (zyx) the size of the bounding box

    Returns:
      the generated alignment.
    """
    return Alignment(corner, size)
