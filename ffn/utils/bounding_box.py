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
"""BoundingBox built on Numpy, interoperable with bounding_box_pb2.

Composed of Numpy arrays (3-vectors actually) to support natural arithmetic
operations.  Easily instantiable from and convertible to a BoundingBox proto.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bisect import bisect_right
import copy

import numpy as np

from . import bounding_box_pb2
from . import geom_utils


class BoundingBox(object):
  """BoundingBox built on Numpy, interoperable with bounding_box_pb2."""

  def __init__(self, start=None, size=None, end=None):
    """Initialize a BoundingBox from an existing BoundingBox or explicit bounds.

    If start is not a BoundingBox object or proto, then exactly two of start,
    size, and end must be specified.

    Args:
      start: a Vector3j, 3-element sequence specifying the (inclusive) start
          bound, or BoundingBox proto/object, in which case no other arguments
          may be specified.
      size: a Vector3j or 3-element sequence specifying the size.
      end: a Vector3j or 3-element sequence specifying the (exclusive) end
          bound.

    Raises:
      ValueError: on bad inputs.
    """
    if start is not None:
      if (isinstance(start, bounding_box_pb2.BoundingBox) or
          isinstance(start, BoundingBox)):
        if size is not None or end is not None:
          raise ValueError('a BoundingBox object/proto must be specified alone')
        size = start.size
        start = start.start

    if (end is not None) + (start is not None) + (size is not None) != 2:
      raise ValueError('exactly two of start, end, and size must be specified')

    if start is not None:
      self.start = geom_utils.ToNumpy3Vector(start)
    if size is not None:
      self.size = geom_utils.ToNumpy3Vector(size)
    if end is not None:
      end = geom_utils.ToNumpy3Vector(end)

    if end is not None:
      if size is not None:
        self.start = end - size
      else:
        self.size = end - start

  def adjusted_by(self, start=None, end=None):
    """Adds an offset to the start and/or end bounds of the bounding box.

    Both arguments can be any argument type supported by
    geom_utils.ToNumpy3Vector, i.e. a Vector3j proto, a 3-tuple, or a
    3-element numpy array.

    Args:
      start: vector offset added to the start bound
      end: vector offset added to the end bound

    Returns:
      A new BoundingBox with adjusted bounds.

    Raises:
      ValueError: on bad inputs.
    """
    start_pos = self.start
    end_pos = self.end
    if start is not None:
      # We don't use += because that would modify our self.start in place.
      start_pos = start_pos + geom_utils.ToNumpy3Vector(start)  # pylint: disable=g-no-augmented-assignment
    if end is not None:
      end_pos = end_pos + geom_utils.ToNumpy3Vector(end)  # pylint: disable=g-no-augmented-assignment
    return BoundingBox(start=start_pos, end=end_pos)

  @property
  def end(self):
    """Returns the (exclusive) end bound as a 3-element int64 numpy array.

    Returns:
      self.start + self.size
    """
    return self.start + self.size

  def Sub(self, start=None, end=None, size=None):
    """Returns a new BoundingBox with the specified bounds relative to self.

    Args:
      start: Specifies the new start bound, relative to self.start.  If not
          specified, the current start bound is kept, unless end and size are
          both specified, in which case it is inferred.
      end: Specifies the new end bound, relative to self.start.  If not
          specified, the current end bound is kept, unless start and size are
          both specified, in which case it is inferred.
      size: In conjunction with start or end (but not both), specifies the new
          size.

    Returns:
      A new BoundingBox with adjusted bounds, or self if no arguments are
    specified.

    Raises:
      ValueError: if invalid arguments are specified.
    """
    if start is None:
      if end is None:
        if size is not None:
          raise ValueError('size must be specified with either end or start')
        return self
      else:
        end = geom_utils.ToNumpy3Vector(end)
        if size is None:
          return BoundingBox(self.start, end)
        else:
          size = geom_utils.ToNumpy3Vector(size)
          start = self.start + end - size
          return BoundingBox(start, size)
    else:
      # start specified.
      start = geom_utils.ToNumpy3Vector(start)
      if end is None:
        if size is None:
          size = self.size - start
        return BoundingBox(self.start + start, size)
      else:
        # end specified.
        if size is not None:
          raise ValueError(
              'size must not be specified if both start and end are given')
        return BoundingBox(self.start + start, end - start)

  def to_proto(self):
    """Returns a corresponding BoundingBox proto."""
    proto = bounding_box_pb2.BoundingBox()
    proto.start.CopyFrom(geom_utils.ToVector3j(self.start))
    proto.size.CopyFrom(geom_utils.ToVector3j(self.size))
    return proto

  def to_slice(self):
    """Returns slice in C-order (ZYX)."""
    return np.index_exp[self.start[2]:self.end[2],  #
                        self.start[1]:self.end[1],  #
                        self.start[0]:self.end[0]]

  def __repr__(self):
    return 'BoundingBox(start=%s, size=%s)' % (tuple(self.start),
                                               tuple(self.size))

  def __eq__(self, other):
    if isinstance(other, bounding_box_pb2.BoundingBox):
      other = BoundingBox(other)
    elif not isinstance(other, BoundingBox):
      return False
    return (np.all(self.start == other.start) and
            np.all(self.size == other.size))

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((tuple(self.start), tuple(self.size)))


def intersection(box0, box1):
  """Get intersection between two bounding boxes, or None."""
  if isinstance(box0, bounding_box_pb2.BoundingBox):
    box0 = BoundingBox(box0.start, box0.size)
  if isinstance(box1, bounding_box_pb2.BoundingBox):
    box1 = BoundingBox(box1.start, box1.size)
  if not isinstance(box0, BoundingBox):
    raise ValueError('box0 must be a BoundingBox')
  if not isinstance(box0, BoundingBox):
    raise ValueError('box1 must be a BoundingBox')
  start = np.maximum(box0.start, box1.start)
  end = np.minimum(box0.end, box1.end)
  if np.any(end <= start): return None
  return BoundingBox(start=start, end=end)


def intersections(boxes0, boxes1):
  """Get intersections between two sequences of boxes.

  Args:
    boxes0: a sequence of BoundingBoxes
    boxes1: a sequence of BoundingBoxes

  Returns:
    list of intersections between the two sequences.  Each element of boxes0 is
    intersected with each element of boxes1, and any non-None are added to the
    list.
  """
  intersections = []
  for box0 in boxes0:
    current_intersections = [intersection(box0, box1) for box1 in boxes1]
    intersections.extend([i for i in current_intersections if i is not None])
  return intersections


def containing(*boxes):
  """Get the minimum bounding box containing all specified boxes.

  Args:
    *boxes: one or more bounding boxes

  Returns:
    The minimum bounding box that contains all boxes.

  Raises:
    ValueError: if invalid arguments are 217specified.
  """
  if not boxes:
    raise ValueError('At least one bounding box must be specified')
  boxes_objs = map(BoundingBox, boxes)
  start = boxes_objs[0].start
  end = boxes_objs[0].end
  for box in boxes_objs[1:]:
    start = np.minimum(start, box.start)
    end = np.maximum(end, box.end)
  return BoundingBox(start=start, end=end)


class OrderlyOverlappingCalculator(object):
  """Helper for calculating orderly overlapping sub-boxes.

  Provides a serial generator as well as num_sub_boxes and index_to_sub_box to
  support parallel dynamic generation.
  """

  def __init__(self,
               outer_box,
               sub_box_size,
               overlap,
               include_small_sub_boxes=False,
               back_shift_small_sub_boxes=False):
    """Helper for calculating orderly overlapping sub-boxes.

    Args:
      outer_box: BoundingBox to be broken into sub-boxes.
      sub_box_size: 3-sequence giving desired 3d size of each sub-box.  Smaller
        sub-boxes may be included at the back edge of the volume, but not if
        they are smaller than overlap (in that case they are completely included
        in the preceding box) unless include_small_sub_boxes is True.  If an
        element is None, the entire range available within outer_box is used for
        that dimension.
      overlap: 3-sequence giving the overlap between neighboring sub-boxes. Must
        be < sub_box_size.
      include_small_sub_boxes: Whether to include small subvolumes at the back
        end which are smaller than overlap
      back_shift_small_sub_boxes: If True, do not produce undersized boxes at
        the back edge of the outer_box.  Instead, shift the start of these boxes
        back so that they can maintain sub_box_size.  This means that the boxes
        at the back edge will have more overlap than the rest of the boxes.

    Raises:
      ValueError: if overlap >= sub_box_size.
    """
    # Allow sub_box_size elements to be None, which means use the whole range.
    sub_box_size = list(sub_box_size)
    for i, x in enumerate(sub_box_size):
      if x is None:
        sub_box_size[i] = outer_box.size[i]

    overlap = np.array(overlap)
    stride = np.array(sub_box_size) - overlap
    if np.any(stride <= 0):
      raise ValueError('sub_box_size must be greater than overlap: %r versus %r'
                       % (sub_box_size, overlap))

    if not include_small_sub_boxes:
      # Don't include subvolumes at the back end that are smaller than overlap;
      # these are included completely in the preceding subvolume.
      end = outer_box.end - overlap
    else:
      end = outer_box.end

    self.start = outer_box.start
    self.stride = stride
    self.end = end
    self.sub_box_size = sub_box_size
    self.outer_box = outer_box

    size = end - self.start
    self.total_sub_boxes_xyz = (size + stride - 1) // stride  # ceil_div
    self.back_shift_small_sub_boxes = back_shift_small_sub_boxes

  def start_to_box(self, start):
    full_box = BoundingBox(start=start, size=self.sub_box_size)
    if self.back_shift_small_sub_boxes:
      shift = np.maximum(full_box.end - self.outer_box.end, 0)
      if shift.any():
        return BoundingBox(start=full_box.start - shift, size=self.sub_box_size)
      return full_box
    else:
      return intersection(full_box, self.outer_box)

  def index_to_sub_box(self, index):
    """Translates a linear index to appropriate sub box.

    Args:
      index: The linear index in [0, num_sub_boxes)

    Returns:
      The corresponding BoundingBox.

    The boxes are guaranteed to be generated in Fortran order, i.e. x fastest
    changing.  (This means that VolumeStore Subvolumes generated from contiguous
    indices will be near each other in x.)
    """
    coords = np.unravel_index(index, self.total_sub_boxes_xyz, order='F')
    return self.start_to_box(coords * self.stride + self.start)

  def offset_to_index(self, index, offset):
    """Calculate the index of another box at offset w.r.t.

    current index.

    Args:
      index: the current flat index from which to calculate the offset index.
      offset: the xyz offset from current index at which to calculate the new
        index.

    Returns:
      The flat index at offset from current index, or None if the given offset
      goes beyond the range of sub-boxes.

    This is usually used to calculate the boxes that neighbor the current box.
    """
    coords = np.unravel_index(index, self.total_sub_boxes_xyz, order='F')
    offset_coords = np.array(coords) + offset
    if np.any(offset_coords < 0) or np.any(
        offset_coords >= self.total_sub_boxes_xyz):
      return None
    return np.ravel_multi_index(
        offset_coords, self.total_sub_boxes_xyz, order='F')

  def num_sub_boxes(self):
    """Total number of sub-boxes."""
    prod = self.total_sub_boxes_xyz.astype(object).prod()
    return prod

  def generate_sub_boxes(self):
    """Generates all sub-boxes in raster order."""
    for z in range(self.start[2], self.end[2], self.stride[2]):
      for y in range(self.start[1], self.end[1], self.stride[1]):
        for x in range(self.start[0], self.end[0], self.stride[0]):
          yield _required(self.start_to_box((x, y, z)))

  def batched_sub_boxes(self,
                        batch_size,
                        begin_index=0,
                        end_index=None):
    """Generates iterators for batches of sub-boxes.

    Args:
      batch_size: how many sub-boxes per iterable.
      begin_index: the inclusive beginning numerical index.
      end_index: the exclusive ending numerical index.

    Yields:
      An iterable of sub-boxes for each batch.
    """
    if end_index is None:
      end_index = self.num_sub_boxes()
    for i_begin in range(begin_index, end_index, batch_size):
      i_end = min(i_begin + batch_size, end_index)
      yield (
          _required(self.index_to_sub_box(i)) for i in range(i_begin, i_end))

  def tag_border_locations(self, index):
    """Checks whether a box touches the border of the BoundingBox.

    Args:
      index: flat index identifying the box to check

    Returns:
      2-tuple of bool 3d ndarrays (dim order: x, y, z).
      True if the box touches the border at the start/end (respectively for the
      1st and 2nd element of the tuple) of the bbox along the given dimension.
    """
    coords_xyz = np.array(
        np.unravel_index(index, self.total_sub_boxes_xyz, order='F'))
    is_start = coords_xyz == 0
    is_end = coords_xyz == self.total_sub_boxes_xyz - 1
    return is_start, is_end
