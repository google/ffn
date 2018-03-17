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
"""Functions related to the movement of the FFN FoV."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import json
import weakref
import numpy as np
from scipy.special import logit
import tensorflow as tf

from ..training.import_util import import_symbol

# Unless stated otherwise, all shape/coordinate triples in this file are in zyx
# order.


# TODO(mjanusz): This has the potential problem that when an 'Y'-like or more
# complex fork is present, the model could fail to follow one of the branches.
# Doesn't seem to matter in practice though, at least with the current
# FoVs/datasets.
#
# For larger FoVs, we would need to threshold the probability map for every
# face, and look at the max probability point in  every connected component
# within a face. Probably best to implement this in C++ and just use a Python
# wrapper.
def get_scored_move_offsets(deltas, prob_map, threshold=0.9):
  """Looks for potential moves for a FFN.

  The possible moves are determined by extracting probability map values
  corresponding to cuboid faces at +/- deltas, and considering the highest
  probability value for every face.

  Args:
    deltas: (z,y,x) tuple of base move offsets for the 3 axes
    prob_map: current probability map as a (z,y,x) numpy array
    threshold: minimum score required at the new FoV center for a move to be
        considered valid

  Yields:
    tuples of:
      score (probability at the new FoV center),
      position offset tuple (z,y,x) relative to center of prob_map

    The order of the returned tuples is arbitrary and should not be depended
    upon. In particular, the tuples are not necessarily sorted by score.
  """
  center = np.array(prob_map.shape) // 2
  assert center.size == 3
  # Selects a working subvolume no more than +/- delta away from the current
  # center point.
  subvol_sel = [slice(c - dx, c + dx + 1) for c, dx
                in zip(center, deltas)]

  done = set()
  for axis, axis_delta in enumerate(deltas):
    if axis_delta == 0:
      continue
    for axis_offset in (-axis_delta, axis_delta):
      # Move exactly by the delta along the current axis, and select the face
      # of the subvolume orthogonal to the current axis.
      face_sel = subvol_sel[:]
      face_sel[axis] = axis_offset + center[axis]
      face_prob = prob_map[face_sel]
      shape = face_prob.shape

      # Find voxel with maximum activation.
      face_pos = np.unravel_index(face_prob.argmax(), shape)
      score = face_prob[face_pos]

      # Only move if activation crosses threshold.
      if score < threshold:
        continue

      # Convert within-face position to be relative vs the center of the face.
      relative_pos = [face_pos[0] - shape[0] // 2, face_pos[1] - shape[1] // 2]
      relative_pos.insert(axis, axis_offset)
      ret = (score, tuple(relative_pos))

      if ret not in done:
        done.add(ret)
        yield ret


class BaseMovementPolicy(object):
  """Base class for movement policy queues.

  The principal usage is to initialize once with the policy's parameters and
  set up a queue for candidate positions. From this queue candidates can be
  iteratively consumed and the scores should be updated in the FFN
  segmentation loop.
  """

  def __init__(self, canvas, scored_coords, deltas):
    """Initializes the policy.

    Args:
      canvas: Canvas object for FFN inference
      scored_coords: mutable container of tuples (score, zyx coord)
      deltas: step sizes as (z,y,x)
    """
    # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
    self.canvas = weakref.proxy(canvas)
    self.scored_coords = scored_coords
    self.deltas = np.array(deltas)

  def __len__(self):
    return len(self.scored_coords)

  def __iter__(self):
    return self

  def next(self):
    raise StopIteration()

  def append(self, item):
    self.scored_coords.append(item)

  def update(self, prob_map, position):
    """Updates the state after an FFN inference call.

    Args:
      prob_map: object probability map returned by the FFN (in logit space)
      position: postiion of the center of the FoV where inference was performed
          (z, y, x)
    """
    raise NotImplementedError()

  def get_state(self):
    """Returns the state of this policy as a pickable Python object."""
    raise NotImplementedError()

  def restore_state(self, state):
    raise NotImplementedError()

  def reset_state(self, start_pos):
    """Resets the policy.

    Args:
      start_pos: starting position of the current object as z, y, x
    """
    raise NotImplementedError()


class FaceMaxMovementPolicy(BaseMovementPolicy):
  """Selects candidates from maxima on prediction cuboid faces."""

  def __init__(self, canvas, deltas=(4, 8, 8), score_threshold=0.9):
    self.done_rounded_coords = set()
    self.score_threshold = score_threshold
    self._start_pos = None
    super(FaceMaxMovementPolicy, self).__init__(canvas, deque([]), deltas)

  def reset_state(self, start_pos):
    self.scored_coords = deque([])
    self.done_rounded_coords = set()
    self._start_pos = start_pos

  def get_state(self):
    return [(self.scored_coords, self.done_rounded_coords)]

  def restore_state(self, state):
    self.scored_coords, self.done_rounded_coords = state[0]

  def __next__(self):
    """Pops positions from queue until a valid one is found and returns it."""
    while self.scored_coords:
      _, coord = self.scored_coords.popleft()
      coord = tuple(coord)
      if self.quantize_pos(coord) in self.done_rounded_coords:
        continue
      if self.canvas.is_valid_pos(coord):
        break
    else:  # Else goes with while, not with if!
      raise StopIteration()

    return tuple(coord)

  def next(self):
    return self.__next__()

  def quantize_pos(self, pos):
    """Quantizes the positions symmetrically to a grid downsampled by deltas."""
    # Compute offset relative to the origin of the current segment and
    # shift by half delta size. This ensures that all directions are treated
    # approximately symmetrically -- i.e. the origin point lies in the middle of
    # a cell of the quantized lattice, as opposed to a corner of that cell.
    rel_pos = (np.array(pos) - self._start_pos)
    coord = (rel_pos + self.deltas // 2) // np.maximum(self.deltas, 1)
    return tuple(coord)

  def update(self, prob_map, position):
    """Adds movements to queue for the cuboid face maxima of ``prob_map``."""
    qpos = self.quantize_pos(position)
    self.done_rounded_coords.add(qpos)

    scored_coords = get_scored_move_offsets(self.deltas, prob_map,
                                            threshold=self.score_threshold)
    scored_coords = sorted(scored_coords, reverse=True)
    for score, rel_coord in scored_coords:
      # convert to whole cube coordinates
      coord = [rel_coord[i] + position[i] for i in range(3)]
      self.scored_coords.append((score, coord))


def get_policy_fn(request, ffn_model):
  """Returns a policy class based on the InferenceRequest proto."""

  if request.movement_policy_name:
    movement_policy_class = globals().get(request.movement_policy_name, None)
    if movement_policy_class is None:
      movement_policy_class = import_symbol(request.movement_policy_name)
  else:  # Default / fallback.
    movement_policy_class = FaceMaxMovementPolicy

  if request.movement_policy_args:
    kwargs = json.loads(request.movement_policy_args)
  else:
    kwargs = {}
  if 'deltas' not in kwargs:
    kwargs['deltas'] = ffn_model.deltas[::-1]
  if 'score_threshold' not in kwargs:
    kwargs['score_threshold'] = logit(request.inference_options.move_threshold)

  return lambda canvas: movement_policy_class(canvas, **kwargs)


class MovementRestrictor(object):
  """Restricts the movement of the FFN FoV."""

  def __init__(self, mask=None, shift_mask=None, shift_mask_fov=None,
               shift_mask_threshold=4, shift_mask_scale=1, seed_mask=None):
    """Initializes the restrictor.

    Args:
      mask: 3d ndarray-like of shape (z, y, x); positive values indicate voxels
          that are not going to be segmented
      shift_mask: 4d ndarray-like of shape (2, z, y, x) representing a 2d shift
          vector field
      shift_mask_fov: bounding_box.BoundingBox around large shifts in which to
          restrict movement.  BoundingBox specified as XYZ, start can be
          negative.
      shift_mask_threshold: if any component of the shift vector exceeds this
          value within the FoV, the location will not be segmented
      shift_mask_scale: an integer factor specifying how much larger the pixels
          of the shift mask are compared to the data set processed by the FFN
    """
    self.mask = mask
    self.seed_mask = seed_mask

    self._shift_mask_scale = shift_mask_scale
    self.shift_mask = None
    if shift_mask is not None:
      self.shift_mask = (np.max(np.abs(shift_mask), axis=0) >=
                         shift_mask_threshold)

      assert shift_mask_fov is not None
      self._shift_mask_fov_pre_offset = shift_mask_fov.start[::-1]
      self._shift_mask_fov_post_offset = shift_mask_fov.end[::-1] - 1

  def is_valid_seed(self, pos):
    """Checks whether a given position is a valid seed point.

    Args:
      pos: position within the dataset as (z, y, x)

    Returns:
      True iff location is a valid seed
    """
    if self.seed_mask is not None and self.seed_mask[pos]:
      return False

    return True

  def is_valid_pos(self, pos):
    """Checks whether a given position should be segmented.

    Args:
      pos: position within the dataset as (z, y, x)

    Returns:
      True iff location should be segmented
    """

    # Location masked?
    if self.mask is not None and self.mask[pos]:
      return False

    if self.shift_mask is not None:
      np_pos = np.array(pos)
      fov_low = np.maximum(np_pos + self._shift_mask_fov_pre_offset, 0)
      fov_high = np_pos + self._shift_mask_fov_post_offset
      start = fov_low // self._shift_mask_scale
      end = fov_high // self._shift_mask_scale

      # Do not allow movement through highly distorted areas, which often
      # result in merge errors. In the simplest case, the distortion magnitude
      # is quantified with a patch-based cross-correlation map.
      if np.any(self.shift_mask[fov_low[0]:(fov_high[0] + 1),
                                start[1]:(end[1] + 1),
                                start[2]:(end[2] + 1)]):
        return False

    return True
