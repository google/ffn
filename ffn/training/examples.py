# Copyright 2024 Google Inc.
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
"""Utilities for building training examples for FFN training."""

import collections
from concurrent import futures
import itertools
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy import special

from . import mask
from . import model as ffn_model
from . import tracker

GetOffsets = Callable[
    [ffn_model.ModelInfo, np.ndarray, np.ndarray, tracker.EvalTracker],
    Iterable[tuple[int, int, int]]]


def get_example(load_example, eval_tracker: tracker.EvalTracker,
                info: ffn_model.ModelInfo, get_offsets: GetOffsets,
                seed_pad: float, seed_shape: tuple[int, int, int]):
  """Generates individual training examples.

  Args:
    load_example: callable returning a tuple of image and label ndarrays as well
      as the seed coordinate and volume name of the example
    eval_tracker: tracker.EvalTracker object
    info: ModelInfo metadata about the model
    get_offsets: callable returning an iterable of (x, y, z) offsets to
      investigate within the training patch
    seed_pad: value to fill the empty areas of the seed with
    seed_shape: z, y, x shape of the seed

  Yields:
    tuple of [1, z, y, x, 1]-shaped arrays for:
      seed, image, label, weights
  """
  while True:
    ex = load_example()
    full_patches, full_labels, loss_weights, coord, volname = ex

    # Start with a clean seed.
    seed = special.logit(mask.make_seed(seed_shape, 1, pad=seed_pad))

    for off in get_offsets(info, seed, full_labels, eval_tracker):
      predicted = mask.crop_and_pad(seed, off, info.input_seed_size[::-1])
      patches = mask.crop_and_pad(full_patches, off,
                                  info.input_image_size[::-1])
      labels = mask.crop_and_pad(full_labels, off, info.pred_mask_size[::-1])
      weights = mask.crop_and_pad(loss_weights, off, info.pred_mask_size[::-1])

      # Necessary, since the caller is going to update the array and these
      # changes need to be visible in the following iterations.
      assert predicted.base is seed
      yield predicted, patches, labels, weights

    eval_tracker.add_patch(full_labels, seed, loss_weights, coord,
                           volume_name=volname)


ExampleGenerator = Iterable[tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray]]
_BatchGenerator = Iterable[tuple[Sequence[np.ndarray], Sequence[np.ndarray],
                                 Sequence[np.ndarray], Sequence[np.ndarray]]]


def _batch_gen(make_example_generator_fn: Callable[[], ExampleGenerator],
               batch_size: int) -> _BatchGenerator:
  """Generates batches of training examples."""
  # Create a separate generator for every element in the batch. This generator
  # will automatically advance to a different training example once the
  # allowed moves for the current location are exhausted.
  example_gens = [make_example_generator_fn() for _ in range(batch_size)]

  with futures.ThreadPoolExecutor(max_workers=batch_size) as tpe:
    while True:
      fs = []
      for gen in example_gens:
        fs.append(tpe.submit(next, gen))

      # `batch` is sequence of `batch_size` tuples returned by the
      # `get_example` generator, to which we apply the following transformation:
      #   [(a0, b0), (a1, b1), .. (an, bn)] -> [(a0, a1, .., an),
      #                                         (b0, b1, .., bn)]
      # (where n is the batch size) to get a sequence, each element of which
      # represents a batch of values of a given type (e.g., seed, image, etc.)
      batch = [f.result() for f in fs]
      yield tuple(zip(*batch))


class BatchExampleIter:
  """Generates batches of training examples."""

  def __init__(self, example_generator_fn: Callable[[], ExampleGenerator],
               eval_tracker: tracker.EvalTracker, batch_size: int,
               info: ffn_model.ModelInfo):
    self._eval_tracker = eval_tracker
    self._batch_generator = _batch_gen(example_generator_fn, batch_size)
    self._seeds = None
    self._info = info

  def __iter__(self):
    return self

  def __next__(self):
    seeds, patches, labels, weights = next(self._batch_generator)
    self._seeds = seeds
    batched_seeds = np.concatenate(seeds)
    batched_weights = np.concatenate(weights)
    self._eval_tracker.track_weights(batched_weights)
    return (batched_seeds, np.concatenate(patches), np.concatenate(labels),
            batched_weights)

  def update_seeds(self, batched_seeds: np.typing.ArrayLike):
    """Distributes updated predictions back to the generator buffers.

    Args:
      batched_seeds: [b, z, y, x, c] array of the part of the seed updated by
        the model
    """
    assert self._seeds is not None

    # Convert to numpy array in case this function was called with an array-like
    # object backed by accelerator memory.
    batched_seeds = np.asarray(batched_seeds)

    dx = self._info.input_seed_size[0] - self._info.pred_mask_size[0]
    dy = self._info.input_seed_size[1] - self._info.pred_mask_size[1]
    dz = self._info.input_seed_size[2] - self._info.pred_mask_size[2]

    if dz == 0 and dy == 0 and dx == 0:
      for i in range(len(self._seeds)):
        self._seeds[i][:] = batched_seeds[i, ...]
    else:
      for i in range(len(self._seeds)):
        self._seeds[i][:,  #
                       dz // 2:-(dz - dz // 2),  #
                       dy // 2:-(dy - dy // 2),  #
                       dx // 2:-(dx - dx // 2),  #
                       :] = batched_seeds[i, ...]


def _eval_move(seed: np.ndarray, labels: np.ndarray,
               off_xyz: tuple[int, int, int], seed_threshold: float,
               label_threshold: float) -> tuple[bool, bool]:
  """Evaluates a FOV move."""
  valid_move = seed[:,  #
                    seed.shape[1] // 2 + off_xyz[2],  #
                    seed.shape[2] // 2 + off_xyz[1],  #
                    seed.shape[3] // 2 + off_xyz[0],  #
                    0] >= seed_threshold
  wanted_move = (
      labels[:,  #
             labels.shape[1] // 2 + off_xyz[2],  #
             labels.shape[2] // 2 + off_xyz[1],  #
             labels.shape[3] // 2 + off_xyz[0],  #
             0] >= label_threshold)

  return valid_move, wanted_move


FovShifts = Optional[Iterable[tuple[int, int, int]]]


def fixed_offsets(info: ffn_model.ModelInfo,
                  seed: np.ndarray,
                  labels: np.ndarray,
                  eval_tracker: tracker.EvalTracker,
                  threshold: float,
                  fov_shifts: FovShifts = None):
  """Generates offsets based on a fixed list."""
  del info

  label_threshold = special.expit(threshold)
  for off in itertools.chain([(0, 0, 0)], fov_shifts):  # xyz
    valid_move, wanted_move = _eval_move(seed, labels, off, threshold,
                                         label_threshold)
    eval_tracker.record_move(wanted_move, valid_move, off)
    if not valid_move:
      continue

    yield off


def fixed_offsets_window(info: ffn_model.ModelInfo,
                         seed: np.ndarray,
                         labels: np.ndarray,
                         eval_tracker: tracker.EvalTracker,
                         threshold: float,
                         fov_shifts: FovShifts = None,
                         radius: int = 4):
  """Like fixed_offsets, but allows more flexible moves.

  Instead of looking at the single voxel pointed to by the offset vector,
  considers a small window in the plane orthogonal to the movement direction.

  This helps with training on thin processes that might not be followed by the
  'fixed_offsets' movement policy.

  Args:
    info: ModelInfo object
    seed: seed array (logits)
    labels: label array (probabilities)
    eval_tracker: EvalTracker object
    threshold: value that the seed needs to match or exceed in order to be
      considered a valid move target
    fov_shifts: list of XYZ moves to evaluate
    radius: max distance away from the offset vector to look for voxels crossing
      threshold (within a plan ortohogonal to that vector)

  Yields:
    XYZ offset tuples indicating moves to take relative to the center of 'seed'
  """
  off = 0, 0, 0
  label_threshold = special.expit(threshold)
  valid_move, wanted_move = _eval_move(seed, labels, off, threshold,
                                       label_threshold)
  eval_tracker.record_move(wanted_move, valid_move, off)
  if valid_move:
    yield off

  seed_center = np.array(seed.shape[1:4]) // 2
  label_center = np.array(labels.shape[1:4]) // 2

  # Define a thin shell at distance of 'delta' around the center.
  hz, hy, hx = np.mgrid[:seed.shape[1], :seed.shape[2], :seed.shape[3]]
  hz -= seed_center[0]
  hy -= seed_center[1]
  hx -= seed_center[2]
  halo = ((np.abs(hx) <= info.deltas[0]) &  #
          (np.abs(hy) <= info.deltas[1]) &  #
          (np.abs(hz) <= info.deltas[2]) & (  #
              (np.abs(hx) == info.deltas[0]) |  #
              (np.abs(hy) == info.deltas[1]) |  #
              (np.abs(hz) == info.deltas[2])))

  for off in fov_shifts:  # xyz
    # Look for voxels within a window of radius 'radius' around the standard
    # move point. We can extend this window in any direction below since
    # the 'halo' array is set up to restrict us to relevant voxels only.
    off_center = seed_center + off[::-1]
    pre = off_center - radius
    post = off_center + radius + 1
    zz, yy, xx = np.where(halo[pre[0]:post[0], pre[1]:post[1], pre[2]:post[2]])

    zz_s = zz + pre[0]
    yy_s = yy + pre[1]
    xx_s = xx + pre[2]
    xx_l = xx_s + label_center[2] - seed_center[2]
    yy_l = yy_s + label_center[1] - seed_center[1]
    zz_l = zz_s + label_center[0] - seed_center[0]

    # Under 'fixed_offsets' the exact voxel at the offset vector would
    # have to cross the threshold. Here it is instead sufficient that any voxel
    # with a specified radius does.
    valid_move = np.any(seed[:, zz_s, yy_s, xx_s, :] >= threshold)
    wanted_move = np.any(labels[:, zz_l, yy_l, xx_l, :] >= label_threshold)
    eval_tracker.record_move(wanted_move, valid_move, off)
    if valid_move:
      yield off


def no_offsets(info: ffn_model.ModelInfo, seed: np.ndarray, labels: np.ndarray,
               eval_tracker: tracker.EvalTracker):
  del info, labels, seed
  eval_tracker.record_move(True, True, (0, 0, 0))
  yield (0, 0, 0)


# TODO(mjanusz): This has the potential problem that when an 'Y'-like or more
# complex fork is present, the model could fail to follow one of the branches.
# Doesn't seem to matter in practice though, at least with the current
# FoVs/datasets.
#
# For larger FoVs, we would need to threshold the probability map for every
# face, and look at the max probability point in  every connected component
# within a face. Probably best to implement this in C++ and just use a Python
# wrapper.
def get_scored_move_offsets(
    deltas: tuple[int, int, int] | np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.9,
):
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
  subvol_sel = [slice(c - dx, c + dx + 1) for c, dx in zip(center, deltas)]

  done = set()
  for axis, axis_delta in enumerate(deltas):
    if axis_delta == 0:
      continue
    for axis_offset in (-axis_delta, axis_delta):
      # Move exactly by the delta along the current axis, and select the face
      # of the subvolume orthogonal to the current axis.
      face_sel = subvol_sel[:]
      face_sel[axis] = axis_offset + center[axis]
      face_prob = prob_map[tuple(face_sel)]
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


def max_pred_offsets(info: ffn_model.ModelInfo, seed: np.ndarray,
                     labels: np.ndarray, eval_tracker: tracker.EvalTracker,
                     threshold: float, max_radius: np.ndarray):
  """Generates offsets with the policy used for inference."""
  # Always start at the center.
  queue = collections.deque([(0, 0, 0)])  # xyz
  done = set()

  label_threshold = special.expit(threshold)
  deltas = np.array(info.deltas)
  while queue:
    offset = np.array(queue.popleft())

    # Drop any offsets that would take us beyond the image fragment we
    # loaded for training.
    if np.any(np.abs(np.array(offset)) > max_radius):
      continue

    # Ignore locations that were visited previously.
    quantized_offset = tuple((offset + deltas / 2) // np.maximum(deltas, 1))

    if quantized_offset in done:
      continue

    valid, wanted = _eval_move(seed, labels, tuple(offset), threshold,
                               label_threshold)
    eval_tracker.record_move(wanted, valid, (0, 0, 0))

    if not valid or (not wanted and quantized_offset != (0, 0, 0)):
      continue

    done.add(quantized_offset)

    yield tuple(offset)

    # Look for new offsets within the updated seed.
    curr_seed = mask.crop_and_pad(seed, offset, info.pred_mask_size[::-1])
    todos = sorted(
        get_scored_move_offsets(
            info.deltas[::-1], curr_seed[0, ..., 0], threshold=threshold),
        reverse=True)
    queue.extend((x[2] + offset[0], x[1] + offset[1], x[0] + offset[2])
                 for _, x in todos)
