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
