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
"""Utilities for running FFN inference."""

from io import BytesIO
import logging
import os
import threading
import time
from . import inference_pb2
from . import movement
from . import seed
from . import storage
from connectomics.segmentation import labels as label_utils
from .inference_utils import Counters
from .inference_utils import TimedIter
from .inference_utils import timer_counter
import numpy as np

from numpy.lib.stride_tricks import as_strided
from ..utils import ortho_plane_visualization
from scipy.special import expit
from scipy.special import logit
from tensorflow.io import gfile
from ..training import model as ffn_model

MSEC_IN_SEC = 1000
MAX_SELF_CONSISTENT_ITERS = 32


# Visualization.
# ---------------------------------------------------------------------------


class DynamicImage:
  def UpdateFromPIL(self, new_img):
    # pylint: disable=g-import-not-at-top
    from IPython import display   # pytype:disable=import-error
    display.clear_output(wait=True)
    image = BytesIO()
    new_img.save(image, format='png')
    display.display(display.Image(image.getvalue()))


def _cmap_rgb1(drw):
  """Default color palette from gnuplot."""
  r = np.sqrt(drw)
  g = np.power(drw, 3)
  b = np.sin(drw * np.pi)

  return (np.dstack([r, g, b]) * 250.0).astype(np.uint8)


def visualize_state(seed_logits, pos, movement_policy, dynimage):
  """Visualizes the inference state.

  Args:
    seed_logits: ndarray (z, y, x) with the current predicted mask
    pos: current FoV position within 'seed' as z, y, x
    movement_policy: movement policy object
    dynimage: DynamicImage object which is to be updated with the
        state visualization
  """
  from PIL import Image

  planes = ortho_plane_visualization.cut_ortho_planes(
      seed_logits, center=pos, cross_hair=True)
  to_vis = ortho_plane_visualization.concat_ortho_planes(planes)

  if isinstance(movement_policy.scored_coords, np.ndarray):
    scores = movement_policy.scored_coords
    # Upsample the grid.
    zf, yf, xf = movement_policy.deltas
    zz, yy, xx = scores.shape
    zs, ys, xs = scores.strides
    new_sh = (zz, zf, yy, yf, xx, xf)
    new_st = (zs, 0, ys, 0, xs, 0)
    scores_up = as_strided(scores, new_sh, new_st)
    scores_up = scores_up.reshape((zz * zf, yy * yf, xx * xf))
    # TODO(mkillinger) might need padding in some cases, if crashes: fix.
    # The grid might be too large, cut it to be symmetrical
    cut = (np.array(scores_up.shape) - np.array(seed_logits.shape)) // 2
    sh = seed_logits.shape
    scores_up = scores_up[cut[0]:cut[0] + sh[0],
                          cut[1]:cut[1] + sh[1],
                          cut[2]:cut[2] + sh[2]]
    grid_planes = ortho_plane_visualization.cut_ortho_planes(
        scores_up, center=pos, cross_hair=True)
    grid_view = ortho_plane_visualization.concat_ortho_planes(grid_planes)
    grid_view *= 4  # Looks better this way
    to_vis = np.concatenate((to_vis, grid_view), axis=1)

  val = _cmap_rgb1(expit(to_vis))
  y, x = pos[1:]

  # Mark seed in the xy plane.
  val[(y - 1):(y + 2), (x - 1):(x + 2), 0] = 255
  val[(y - 1):(y + 2), (x - 1):(x + 2), 1:] = 0

  vis = Image.fromarray(val)
  dynimage.UpdateFromPIL(vis)


# TODO(mjanusz): Add support for sparse inference.
class Canvas:
  """Tracks state of the inference progress and results within a subvolume."""

  def __init__(self,
               model: ffn_model.FFNModel,
               tf_executor,
               image,
               options,
               counters=None,
               restrictor=None,
               movement_policy_fn=None,
               keep_history=False,
               checkpoint_path=None,
               checkpoint_interval_sec=0,
               corner_zyx=None):
    """Initializes the canvas.

    Args:
      model: FFNModel object
      tf_executor: Executor object to use for inference
      image: 3d ndarray-like of shape (z, y, x)
      options: InferenceOptions proto
      counters: (optional) counter container, where __getitem__ returns a
          counter compatible with the MR Counter API
      restrictor: (optional) a MovementRestrictor object which can exclude
          some areas of the data from the segmentation process
      movement_policy_fn: callable taking the Canvas object as its
          only argument and returning a movement policy object
          (see movement.BaseMovementPolicy)
      keep_history: whether to maintain a record of locations visited by the
          FFN, together with any associated metadata; note that this data is
          kept only for the object currently being segmented
      checkpoint_path: (optional) path at which to save a checkpoint file
      checkpoint_interval_sec: how often to save a checkpoint file (in
          seconds); if <= 0, no checkpoint are going to be saved
      corner_zyx: 3 element array-like indicating the spatial corner of the
          image in (z, y, x)
    """
    self.model = model
    self.image = image
    self.executor = tf_executor
    self._exec_client_id = None

    self.options = inference_pb2.InferenceOptions()
    self.options.CopyFrom(options)
    # Convert thresholds, etc. to logit values for efficient inference.
    for attr in ('init_activation', 'pad_value', 'move_threshold',
                 'segment_threshold'):
      setattr(self.options, attr, logit(getattr(self.options, attr)))

    self.counters = counters if counters is not None else Counters()
    self.checkpoint_interval_sec = checkpoint_interval_sec
    self.checkpoint_path = checkpoint_path
    self.checkpoint_last = time.time()

    self._keep_history = keep_history
    self.corner_zyx = corner_zyx
    self.shape = image.shape

    if restrictor is None:
      self.restrictor = movement.MovementRestrictor()
    else:
      self.restrictor = restrictor

    # Cast to array to ensure we can do elementwise expressions later.
    # All of these are in zyx order.
    self._pred_size = np.array(model.info.pred_mask_size[::-1])
    self._input_seed_size = np.array(model.info.input_seed_size[::-1])
    self._input_image_size = np.array(model.info.input_image_size[::-1])
    self.margin = self._input_image_size // 2

    self._pred_delta = (self._input_seed_size - self._pred_size) // 2
    assert np.all(self._pred_delta >= 0)

    # Current working area. This represents an object probability map
    # in logit form, and is fed directly as the mask input to the FFN
    # model.
    self.seed = np.zeros(self.shape, dtype=np.float32)
    self.segmentation = np.zeros(self.shape, dtype=np.int32)
    self.seg_prob = np.zeros(self.shape, dtype=np.uint8)

    # When an initial segmentation is provided, maps the global ID space
    # to locally used IDs.
    self.global_to_local_ids = {}

    self.seed_policy = None
    self._seed_policy_state = None

    # Maximum segment ID already assigned.
    self._max_id = 0

    # Maps of segment id -> ..
    self.origins = {}  # seed location
    self.overlaps = {}  # (ids, number overlapping voxels)

    # Whether to always create a new seed in segment_at.
    self.reset_seed_per_segment = True

    if movement_policy_fn is None:
      # The model.deltas are (for now) in xyz order and must be swapped to zyx.
      self.movement_policy = movement.FaceMaxMovementPolicy(
          self, deltas=model.info.deltas[::-1],
          score_threshold=self.options.move_threshold)
    else:
      self.movement_policy = movement_policy_fn(self)

    self.reset_state((0, 0, 0))
    self.t_last_predict = None

  def _register_client(self):
    if self._exec_client_id is None:
      self._exec_client_id = self.executor.start_client()
      logging.info('Registered as client %d.', self._exec_client_id)

  def _deregister_client(self):
    if self._exec_client_id is not None:
      logging.info('Deregistering client %d', self._exec_client_id)
      self.executor.finish_client(self._exec_client_id)
      self._exec_client_id = None

  def __del__(self):
    # Note that the presence of this method will cause a memory leak in
    # case the Canvas object is part of a reference cycle. Use weakref.proxy
    # where such cycles are really needed.
    self._deregister_client()

  def local_id(self, segment_id):
    return self.global_to_local_ids.get(segment_id, segment_id)

  def reset_state(self, start_pos):
    # Resetting the movement_policy is currently necessary to update the
    # policy's bitmask for whether a position is already segmented (the
    # canvas updates the segmented mask only between calls to segment_at
    # and therefore the policy does not update this mask for every call.).
    self.movement_policy.reset_state(start_pos)
    self.history = []
    self.history_deleted = []

    self._min_pos = np.array(start_pos)
    self._max_pos = np.array(start_pos)
    self._register_client()

  def is_valid_pos(self, pos, ignore_move_threshold=False):
    """Returns True if segmentation should be attempted at the given position.

    Args:
      pos: position to check as (z, y, x)
      ignore_move_threshold: (boolean) when starting a new segment at pos the
          move threshold can and must be ignored.

    Returns:
      Boolean indicating whether to run FFN inference at the given position.
    """

    if not ignore_move_threshold:
      if self.seed[pos] < self.options.move_threshold:
        self.counters['skip_threshold'].Increment()
        logging.debug('.. seed value below threshold.')
        return False

    # Not enough image context?
    np_pos = np.array(pos)
    low = np_pos - self.margin
    high = np_pos + self.margin

    if np.any(low < 0) or np.any(high >= self.shape):
      self.counters['skip_invalid_pos'].Increment()
      logging.debug('.. too close to border: %r', pos)
      return False

    # Location already segmented?
    if self.segmentation[pos] > 0:
      self.counters['skip_invalid_pos'].Increment()
      logging.debug('.. segmentation already active: %r', pos)
      return False

    return True

  def _get_image(self, pos):
    """Returns the image for the FOV centered at 'pos'."""
    # Top-left corner of the FoV.
    start = np.array(pos) - self.margin
    end = start + self._input_image_size
    img = self.image[tuple(slice(s, e) for s, e in zip(start, end))]
    return img

  def predict(self, pos, logit_seed):
    """Runs a single step of FFN prediction.

    Args:
      pos: (z, y, x) position of the center of the FoV
      logit_seed: current seed to feed to the model as input, z, y, x ndarray

    Returns:
      updated logits
    """
    with timer_counter(self.counters, 'predict'):
      with timer_counter(self.counters, 'get-image'):
        img = self._get_image(pos)

      # Record the amount of time spent on non-prediction tasks.
      if self.t_last_predict is not None:
        delta_t = time.time() - self.t_last_predict
        self.counters['inference-not-predict-ms'].IncrementBy(
            delta_t * MSEC_IN_SEC)

      fetches = {'logits': self.model.logits}
      with timer_counter(self.counters, 'inference'):
        fetches = self.executor.predict(self._exec_client_id,
                                        logit_seed, img, fetches)

      self.t_last_predict = time.time()

    logits = fetches.pop('logits')
    return logits[..., 0]

  def update_at(self, pos):
    """Updates object mask prediction at a specific position.

    Note that depending on the settings of the canvas, the update might involve
    more than 1 inference run of the FFN.

    Args:
      pos: (z, y, x) position of the center of the FoV

    Returns:
      ndarray of the predicted mask in logit space
    """
    with timer_counter(self.counters, 'update_at'):
      off = self._input_seed_size // 2  # zyx

      start = np.array(pos) - off
      end = start + self._input_seed_size
      logit_seed = np.array(
          self.seed[tuple(slice(s, e) for s, e in zip(start, end))]
      )
      init_prediction = np.isnan(logit_seed)
      logit_seed[init_prediction] = np.float32(self.options.pad_value)

      logits = self.predict(pos, logit_seed)
      start += self._pred_delta
      end = start + self._pred_size
      sel = tuple(slice(s, e) for s, e in zip(start, end))

      # Bias towards oversegmentation by making it impossible to reverse
      # disconnectedness predictions in the course of inference.
      if self.options.disco_seed_threshold >= 0:
        th_max = logit(0.5)
        old_seed = self.seed[sel]

        if self._keep_history:
          self.history_deleted.append(
              np.sum((old_seed >= logit(0.8)) & (logits < th_max)))

        if (np.mean(logits >= self.options.move_threshold) >
            self.options.disco_seed_threshold):
          # Because (x > NaN) is always False, this mask excludes positions that
          # were previously uninitialized (i.e. set to NaN in old_seed).
          old_err = np.seterr(invalid='ignore')
          try:
            mask = ((old_seed < th_max) & (logits > old_seed))
          finally:
            np.seterr(**old_err)
          logits[mask] = old_seed[mask]

      # Update working space.
      self.seed[sel] = logits

    return logits

  def init_seed(self, pos):
    """Reinitiailizes the object mask with a seed.

    Args:
      pos: position at which to place the seed (z, y, x)
    """
    self.seed[...] = np.nan
    self.seed[pos] = self.options.init_activation

  def segment_at(self, start_pos, dynamic_image=None,
                 vis_update_every=10,
                 vis_fixed_z=False):
    """Runs FFN segmentation starting from a specific point.

    Args:
      start_pos: location at which to run segmentation as (z, y, x)
      dynamic_image: optional DynamicImage object which to update with
          a visualization of the segmentation state
      vis_update_every: number of FFN iterations between subsequent
          updates of the dynamic image
      vis_fixed_z: if True, the z position used for visualization is
          fixed at the starting value specified in `pos`. Otherwise,
          the current FoV of the FFN is used to determine what to
          visualize.

    Returns:
      number of iterations performed
    """
    if self.reset_seed_per_segment:
      self.init_seed(start_pos)
    # This includes a reset of the movement policy, see comment in method body.
    self.reset_state(start_pos)

    num_iters = 0

    if not self.movement_policy:
      # Add first element with arbitrary priority 1 (it will be consumed
      # right away anyway).
      item = (self.movement_policy.score_threshold * 2, start_pos)
      self.movement_policy.append(item)

    with timer_counter(self.counters, 'segment_at-loop'):
      for pos in self.movement_policy:
        # Terminate early if the seed got too weak.
        if self.seed[start_pos] < self.options.move_threshold:
          self.counters['seed_got_too_weak'].Increment()
          break

        if not self.restrictor.is_valid_pos(pos):
          self.counters['skip_restriced_pos'].Increment()
          continue

        pred = self.update_at(pos)
        self._min_pos = np.minimum(self._min_pos, pos)
        self._max_pos = np.maximum(self._max_pos, pos)
        num_iters += 1

        with timer_counter(self.counters, 'movement_policy'):
          self.movement_policy.update(pred, pos)

        with timer_counter(self.counters, 'segment_at-overhead'):
          if self._keep_history:
            self.history.append(pos)

          if dynamic_image is not None and num_iters % vis_update_every == 0:
            vis_pos = pos if not vis_fixed_z else (start_pos[0], pos[1],
                                                   pos[2])
            visualize_state(self.seed, vis_pos, self.movement_policy,
                            dynamic_image)

          assert np.all(pred.shape == self._pred_size)

          self._maybe_save_checkpoint()

    return num_iters

  def log_info(self, string, *args, **kwargs):
    logging.info('[cl %d] ' + string, self._exec_client_id,
                 *args, **kwargs)

  def segment_all(self, seed_policy=seed.PolicyPeaks):
    """Segments the input image.

    Segmentation is attempted from all valid starting points provided by
    the seed policy.

    Args:
      seed_policy: callable taking the image and the canvas object as arguments
          and returning an iterator over proposed seed point.
    """
    self.seed_policy = seed_policy(self)
    if self._seed_policy_state is not None:
      self.seed_policy.set_state(self._seed_policy_state)
      self._seed_policy_state = None

    with timer_counter(self.counters, 'segment_all'):
      mbd = self.options.min_boundary_dist
      mbd = np.array([mbd.z, mbd.y, mbd.x])

      for pos in TimedIter(self.seed_policy, self.counters, 'seed-policy'):
        # When starting a new segment the move_threshold on the probability
        # should be ignored when determining if the position is valid.
        if not (self.is_valid_pos(pos, ignore_move_threshold=True)
                and self.restrictor.is_valid_pos(pos)
                and self.restrictor.is_valid_seed(pos)):
          continue

        self._maybe_save_checkpoint()

        # Too close to an existing segment?
        low = np.array(pos) - mbd
        high = np.array(pos) + mbd + 1
        sel = tuple(slice(s, e) for s, e in zip(low, high))
        if np.any(self.segmentation[sel] > 0):
          logging.debug('Too close to existing segment.')
          self.segmentation[pos] = -1
          continue

        self.log_info('Starting segmentation at %r (zyx)', pos)

        # Try segmentation.
        seg_start = time.time()
        num_iters = self.segment_at(pos)
        t_seg = time.time() - seg_start

        # Check if segmentation was successful.
        if num_iters <= 0:
          self.counters['invalid-other-time-ms'].IncrementBy(t_seg *
                                                             MSEC_IN_SEC)
          self.log_info('Failed: num iters was %d', num_iters)
          continue

        # Original seed too weak?
        if self.seed[pos] < self.options.move_threshold:
          # Mark this location as excluded.
          if self.segmentation[pos] == 0:
            self.segmentation[pos] = -1
          self.log_info('Failed: weak seed')
          self.counters['invalid-weak-time-ms'].IncrementBy(t_seg * MSEC_IN_SEC)
          continue

        # Restrict probability map -> segment processing to a bounding box
        # covering the area that was actually changed by the FFN. In case the
        # segment is going to be rejected due to small size, this can
        # significantly reduce processing time.
        sel = tuple(
            slice(max(s, 0), e + 1)
            for s, e in zip(
                self._min_pos - self._pred_size // 2,  #
                self._max_pos + self._pred_size // 2,
            )
        )

        # We only allow creation of new segments in areas that are currently
        # empty.
        mask = self.seed[sel] >= self.options.segment_threshold
        raw_segmented_voxels = np.sum(mask)

        # Record existing segment IDs overlapped by the newly added object.
        overlapped_ids, counts = np.unique(self.segmentation[sel][mask],
                                           return_counts=True)
        valid = overlapped_ids > 0
        overlapped_ids = overlapped_ids[valid]
        counts = counts[valid]

        mask &= self.segmentation[sel] <= 0
        actual_segmented_voxels = np.sum(mask)

        # Segment too small?
        if actual_segmented_voxels < self.options.min_segment_size:
          if self.segmentation[pos] == 0:
            self.segmentation[pos] = -1
          self.log_info('Failed: too small: %d', actual_segmented_voxels)
          self.counters['invalid-small-time-ms'].IncrementBy(t_seg *
                                                             MSEC_IN_SEC)
          continue

        self.counters['voxels-segmented'].IncrementBy(actual_segmented_voxels)
        self.counters['voxels-overlapping'].IncrementBy(
            raw_segmented_voxels - actual_segmented_voxels)

        # Find the next free ID to assign.
        self._max_id += 1
        while self._max_id in self.origins:
          self._max_id += 1

        self.segmentation[sel][mask] = self._max_id
        self.seg_prob[sel][mask] = storage.quantize_probability(
            expit(self.seed[sel][mask]))

        self.log_info('Created supervoxel:%d  seed(zyx):%s  size:%d  iters:%d',
                      self._max_id, pos,
                      actual_segmented_voxels, num_iters)

        self.overlaps[self._max_id] = np.array([overlapped_ids, counts])

        # Record information about how a given supervoxel was created.
        self.origins[self._max_id] = storage.OriginInfo(pos, num_iters, t_seg)
        self.counters['valid-time-ms'].IncrementBy(t_seg * MSEC_IN_SEC)

    self.log_info('Segmentation done.')

    # It is important to deregister ourselves when the segmentation is complete.
    # This matters particularly if less than a full batch of subvolumes remains
    # to be segmented. Without the deregistration, the executor will wait to
    # fill the full batch (no longer possible) instead of proceeding with
    # inference.
    self._deregister_client()

  def init_segmentation_from_volume(self, volume, corner, end,
                                    align_and_crop=None):
    """Initializes segmentation from an existing VolumeStore.

    This is useful to start inference with an existing segmentation.
    The segmentation does not need to be generated with an FFN.

    Args:
      volume: volume object, as returned by storage.decorated_volume.
      corner: location at which to read data as (z, y, x)
      end: location at which to stop reading data as (z, y, x)
      align_and_crop: callable to align & crop a 3d segmentation subvolume
    """
    self.log_info('Loading initial segmentation from (zyx) %r:%r',
                  corner, end)

    init_seg = volume[:,  #
                      corner[0]:end[0],  #
                      corner[1]:end[1],  #
                      corner[2]:end[2]]

    init_seg, global_to_local = label_utils.make_contiguous(init_seg)
    init_seg = init_seg[0, ...]

    self.global_to_local_ids = dict(global_to_local)

    self.log_info('Segmentation loaded, shape: %r. Canvas segmentation is %r',
                  init_seg.shape, self.segmentation.shape)
    if align_and_crop is not None:
      init_seg = align_and_crop(init_seg)
      self.log_info('Segmentation cropped to: %r', init_seg.shape)

    self.segmentation[:] = init_seg
    self.seg_prob[self.segmentation > 0] = storage.quantize_probability(
        np.array([1.0]))
    self._max_id = np.max(self.segmentation)
    self.log_info('Max restored ID is: %d.', self._max_id)

  def restore_checkpoint(self, path):
    """Restores state from the checkpoint at `path`."""
    self.log_info('Restoring inference checkpoint: %s', path)
    with gfile.GFile(path, 'rb') as f:
      data = np.load(f)

      self.segmentation[:] = data['segmentation']
      self.seed[:] = data['seed']
      self.seg_prob[:] = data['seg_qprob']
      self.history_deleted = list(data['history_deleted'])
      self.history = list(data['history'])
      self.origins = data['origins'].item()
      if 'overlaps' in data:
        self.overlaps = data['overlaps'].item()

      segmented_voxels = np.sum(self.segmentation != 0)
      self.counters['voxels-segmented'].Set(segmented_voxels)
      self._max_id = np.max(self.segmentation)

      self.movement_policy.restore_state(data['movement_policy'])

      seed_policy_state = data['seed_policy_state']
      # When restoring the state of a previously unused Canvas, the seed
      # policy will not be defined. We just save the seed policy state here
      # for future use in .segment_all().
      self._seed_policy_state = seed_policy_state

      self.counters.loads(data['counters'].item())

    self.log_info('Inference checkpoint restored.')

  def save_checkpoint(self, path):
    """Saves a inference checkpoint to `path`."""
    self.log_info('Saving inference checkpoint to %s.', path)
    with timer_counter(self.counters, 'save_checkpoint'):
      gfile.makedirs(os.path.dirname(path))
      with storage.atomic_file(path) as fd:
        seed_policy_state = None
        if self.seed_policy is not None:
          seed_policy_state = self.seed_policy.get_state()

        np.savez_compressed(fd,
                            movement_policy=self.movement_policy.get_state(),
                            segmentation=self.segmentation,
                            seg_qprob=self.seg_prob,
                            seed=self.seed,
                            origins=self.origins,
                            overlaps=self.overlaps,
                            history=np.array(self.history),
                            history_deleted=np.array(self.history_deleted),
                            seed_policy_state=seed_policy_state,
                            counters=self.counters.dumps())
    self.log_info('Inference checkpoint saved.')

  def _maybe_save_checkpoint(self):
    """Attempts to save a checkpoint.

    A checkpoint is only saved if the canvas is configured to keep checkpoints
    and if sufficient time has passed since the last one was saved.
    """
    if self.checkpoint_path is None or self.checkpoint_interval_sec <= 0:
      return

    if time.time() - self.checkpoint_last < self.checkpoint_interval_sec:
      return

    self.save_checkpoint(self.checkpoint_path)
    self.checkpoint_last = time.time()
