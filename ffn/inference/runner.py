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
"""Helper code for managing FFN inference runs."""

import functools
import json
from typing import Any

from absl import logging
from connectomics.common import bounding_box
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.io import gfile
from ..training import model as ffn_model
from ..training.import_util import import_symbol
from . import align
from . import executor
from . import inference
from . import inference_pb2
from . import inference_utils
from . import movement
from . import seed
from . import storage
from .inference_utils import timer_counter


class Runner:
  """Helper for managing FFN inference runs.

  Takes care of initializing the FFN model and any related functionality
  (e.g. movement policies), as well as input/output of the FFN inference
  data (loading inputs, saving segmentations).
  """

  ALL_MASKED = 1

  request: inference_pb2.InferenceRequest
  executor: executor.BatchExecutor
  init_seg_volume: storage.Volume | None
  _image_volume: storage.Volume | None
  _mask_volumes: dict[str, storage.Volume]
  model: ffn_model.FFNModel
  _shift_mask_volume: storage.Volume | None
  _aligner: align.Aligner
  session: tf.Session
  movement_policy_fn: Any

  def __init__(self):
    self.counters = inference_utils.Counters()
    self.executor = None

  def __del__(self):
    self.stop_executor()

  def stop_executor(self):
    """Shuts down the executor.

    No-op when no executor is active.
    """
    if self.executor is not None:
      self.executor.stop_server()
      self.executor = None

  def _load_model_checkpoint(self, checkpoint_path):
    """Restores the inference model from a training checkpoint.

    Args:
      checkpoint_path: the string path to the checkpoint file to load
    """
    with timer_counter(self.counters, 'restore-tf-checkpoint'):
      logging.info('Loading checkpoint.')
      saver = tf.train.Saver()
      saver.restore(self.session, checkpoint_path)
      logging.info('Checkpoint loaded.')

  def start(self, request, batch_size=1, exec_cls=None, session=None):
    """Opens input volumes and initializes the FFN."""
    self.request = request
    assert self.request.segmentation_output_dir

    logging.debug('Received request:\n%s', request)

    if not gfile.exists(request.segmentation_output_dir):
      gfile.makedirs(request.segmentation_output_dir)

    with timer_counter(self.counters, 'volstore-open'):
      # Disabling cache compression can improve access times by 20-30%
      # as of Aug 2016.
      self._image_volume = storage.decorated_volume(
          request.image, cache_max_bytes=int(1e8),
          cache_compression=False)
      assert self._image_volume is not None

      if request.HasField('init_segmentation'):
        self.init_seg_volume = storage.decorated_volume(
            request.init_segmentation, cache_max_bytes=int(1e8))
      else:
        self.init_seg_volume = None

      def _open_or_none(settings):
        if settings.WhichOneof('volume_path') is None:
          return None
        return storage.decorated_volume(
            settings, cache_max_bytes=int(1e7), cache_compression=False)
      self._mask_volumes = {}
      self._shift_mask_volume = _open_or_none(request.shift_mask)

      alignment_options = request.alignment_options
      null_alignment = inference_pb2.AlignmentOptions.NO_ALIGNMENT
      if not alignment_options or alignment_options.type == null_alignment:
        self._aligner = align.Aligner()
      else:
        type_name = inference_pb2.AlignmentOptions.AlignType.Name(
            alignment_options.type)
        error_string = 'Alignment for type %s is not implemented' % type_name
        logging.error(error_string)
        raise NotImplementedError(error_string)

    self.stop_executor()

    if session is None:
      config = tf.ConfigProto()
      tf.reset_default_graph()
      session = tf.Session(config=config)
    self.session = session
    logging.info('Available TF devices: %r', self.session.list_devices())

    # Initialize the FFN model.
    model_class = import_symbol(request.model_name)
    if request.model_args:
      args = json.loads(request.model_args)
    else:
      args = {}

    args['batch_size'] = batch_size
    self.model = model_class(**args)

    if exec_cls is None:
      exec_cls = executor.ThreadingBatchExecutor

    self.executor = exec_cls(
        self.model, self.session, self.counters, batch_size)
    self.movement_policy_fn = movement.get_policy_fn(request, self.model.info)

    self._load_model_checkpoint(request.model_checkpoint_path)

    self.executor.start_server()

  def make_restrictor(self, corner, subvol_size, image, alignment):
    """Builds a MovementRestrictor object."""
    kwargs = {}

    if self.request.masks:
      with timer_counter(self.counters, 'load-mask'):
        final_mask = storage.build_mask(self.request.masks,
                                        corner, subvol_size,
                                        self._mask_volumes,
                                        image, alignment)

        if np.all(final_mask):
          logging.info('Everything masked.')
          return self.ALL_MASKED

        kwargs['mask'] = final_mask

    if self.request.seed_masks:
      with timer_counter(self.counters, 'load-seed-mask'):
        seed_mask = storage.build_mask(self.request.seed_masks,
                                       corner, subvol_size,
                                       self._mask_volumes,
                                       image, alignment)

        if np.all(seed_mask):
          logging.info('All seeds masked.')
          return self.ALL_MASKED

        kwargs['seed_mask'] = seed_mask

    if self._shift_mask_volume:
      with timer_counter(self.counters, 'load-shift-mask'):
        s = self.request.shift_mask_scale
        shift_corner = np.array(corner) // (1, s, s)
        shift_size = -(-np.array(subvol_size) // (1, s, s))

        shift_alignment = alignment.rescaled(
            np.array((1.0, 1.0, 1.0)) / (1, s, s))
        src_corner, src_size = shift_alignment.expand_bounds(
            shift_corner, shift_size, forward=False)
        src_corner, src_size = storage.clip_subvolume_to_bounds(
            src_corner, src_size, self._shift_mask_volume)
        src_end = src_corner + src_size

        expanded_shift_mask = self._shift_mask_volume[
            0:2,  #
            src_corner[0]:src_end[0],  #
            src_corner[1]:src_end[1],  #
            src_corner[2]:src_end[2]]
        shift_mask = np.array([
            shift_alignment.align_and_crop(
                src_corner, expanded_shift_mask[i], shift_corner, shift_size)
            for i in range(2)])
        shift_mask = alignment.transform_shift_mask(corner, s, shift_mask)

        if self.request.HasField('shift_mask_fov'):
          shift_mask_fov = bounding_box.BoundingBox(
              start=self.request.shift_mask_fov.start,
              size=self.request.shift_mask_fov.size)
        else:
          shift_mask_diameter = np.array(self.model.info.input_image_size)
          shift_mask_fov = bounding_box.BoundingBox(
              start=-(shift_mask_diameter // 2), size=shift_mask_diameter)

        kwargs.update({
            'shift_mask': shift_mask,
            'shift_mask_fov': shift_mask_fov,
            'shift_mask_scale': self.request.shift_mask_scale,
            'shift_mask_threshold': self.request.shift_mask_threshold})

    if kwargs:
      return movement.MovementRestrictor(**kwargs)
    else:
      return None

  def make_canvas(self, corner, subvol_size, **canvas_kwargs):
    """Builds the Canvas object for inference on a subvolume.

    Args:
      corner: start of the subvolume (z, y, x)
      subvol_size: size of the subvolume (z, y, x)
      **canvas_kwargs: passed to Canvas

    Returns:
      A tuple of:
        Canvas object
        Alignment object
    """
    subvol_counters = self.counters.get_sub_counters()
    with timer_counter(subvol_counters, 'load-image'):
      logging.info('Process subvolume: %r', corner)

      # A Subvolume with bounds defined by (src_size, src_corner) is guaranteed
      # to result in no missing data when aligned to (dst_size, dst_corner).
      # Likewise, one defined by (dst_size, dst_corner) is guaranteed to result
      # in no missing data when reverse-aligned to (corner, subvol_size).
      alignment = self._aligner.generate_alignment(corner, subvol_size)

      # Bounding box for the aligned destination subvolume.
      dst_corner, dst_size = alignment.expand_bounds(
          corner, subvol_size, forward=True)
      # Bounding box for the pre-aligned imageset to be fetched from the volume.
      src_corner, src_size = alignment.expand_bounds(
          dst_corner, dst_size, forward=False)
      # Ensure that the request bounds don't extend beyond volume bounds.
      src_corner, src_size = storage.clip_subvolume_to_bounds(
          src_corner, src_size, self._image_volume)

      logging.info('Requested bounds are %r + %r', corner, subvol_size)
      logging.info('Destination bounds are %r + %r', dst_corner, dst_size)
      logging.info('Fetch bounds are %r + %r', src_corner, src_size)

      # Fetch the image from the volume using the src bounding box.
      def get_data_3d(volume: storage.Volume, bbox: bounding_box.BoundingBox):
        slc = bbox.to_slice()
        assert volume is not None
        if volume.ndim == 4:
          slc = np.index_exp[0:1] + slc
        data = volume[slc]
        if data.ndim == 4:
          data = data.squeeze(axis=0)
        return data
      src_bbox = bounding_box.BoundingBox(
          start=src_corner[::-1], size=src_size[::-1])
      src_image = get_data_3d(self._image_volume, src_bbox)
      logging.info('Fetched image of size %r prior to transform',
                   src_image.shape)

      def align_and_crop(image):
        return alignment.align_and_crop(src_corner, image, dst_corner, dst_size,
                                        forward=True)

      # Align and crop to the dst bounding box.
      image = align_and_crop(src_image)
      # image now has corner dst_corner and size dst_size.

      logging.info('Image data loaded, shape: %r.', image.shape)

    restrictor = self.make_restrictor(dst_corner, dst_size, image, alignment)

    image = (image.astype(np.float32) -
             self.request.image_mean) / self.request.image_stddev
    if restrictor == self.ALL_MASKED:
      return None, None

    canvas = inference.Canvas(
        self.model,
        self.executor,
        image,
        self.request.inference_options,
        counters=subvol_counters,
        restrictor=restrictor,
        movement_policy_fn=self.movement_policy_fn,
        checkpoint_path=storage.checkpoint_path(
            self.request.segmentation_output_dir, corner),
        checkpoint_interval_sec=self.request.checkpoint_interval,
        corner_zyx=dst_corner,
        **canvas_kwargs)

    if self.request.HasField('init_segmentation'):
      canvas.init_segmentation_from_volume(self.init_seg_volume, src_corner,
                                           src_bbox.end[::-1], align_and_crop)
    return canvas, alignment

  def get_seed_policy(self, corner, subvol_size):
    """Get seed policy generating callable.

    Args:
      corner: the original corner of the requested subvolume, before any
          modification e.g. dynamic alignment.
      subvol_size: the original requested size.

    Returns:
      A callable for generating seed policies.
    """
    policy_cls = getattr(seed, self.request.seed_policy)
    kwargs = {'corner': corner, 'subvol_size': subvol_size}
    if self.request.seed_policy_args:
      kwargs.update(json.loads(self.request.seed_policy_args))
    return functools.partial(policy_cls, **kwargs)

  def save_segmentation(self, canvas, alignment, target_path, prob_path):
    """Saves segmentation to a file.

    Args:
      canvas: Canvas object containing the segmentation
      alignment: the local Alignment used with the canvas, or None
      target_path: path to the file where the segmentation should
          be saved
      prob_path: path to the file where the segmentation probability
          map should be saved
    """
    def unalign_image(im3d):
      if alignment is None:
        return im3d
      return alignment.align_and_crop(
          canvas.corner_zyx,
          im3d,
          alignment.corner,
          alignment.size,
          forward=False)

    def unalign_origins(origins, canvas_corner):
      out_origins = dict()
      for key, value in origins.items():
        zyx = np.array(value.start_zyx) + canvas_corner
        zyx = alignment.transform(zyx[:, np.newaxis], forward=False).squeeze()
        zyx -= canvas_corner
        out_origins[key] = value._replace(start_zyx=tuple(zyx))
      return out_origins

    # Remove markers.
    canvas.segmentation[canvas.segmentation < 0] = 0

    # Save segmentation results. Reduce # of bits per item if possible.
    storage.save_subvolume(
        unalign_image(canvas.segmentation),
        unalign_origins(canvas.origins, np.array(canvas.corner_zyx)),
        target_path,
        request=self.request.SerializeToString(),
        counters=canvas.counters.dumps(),
        overlaps=canvas.overlaps)

    # Save probability map separately. This has to happen after the
    # segmentation is saved, as `save_subvolume` will create any necessary
    # directories.
    prob = unalign_image(canvas.seg_prob)
    with storage.atomic_file(prob_path) as fd:
      np.savez_compressed(fd, qprob=prob)

  def run(self, corner, subvol_size, reset_counters=True):
    """Runs FFN inference over a subvolume.

    Args:
      corner: start of the subvolume (z, y, x)
      subvol_size: size of the subvolume (z, y, x)
      reset_counters: whether to reset the counters

    Returns:
      Canvas object with the segmentation or None if the canvas could not
      be created or the segmentation subvolume already exists.
    """
    if reset_counters:
      self.counters.reset()

    seg_path = storage.segmentation_path(
        self.request.segmentation_output_dir, corner)
    prob_path = storage.object_prob_path(
        self.request.segmentation_output_dir, corner)
    cpoint_path = storage.checkpoint_path(
        self.request.segmentation_output_dir, corner)

    if gfile.exists(seg_path):
      return None

    canvas, alignment = self.make_canvas(corner, subvol_size)
    if canvas is None:
      return None

    if gfile.exists(cpoint_path):
      canvas.restore_checkpoint(cpoint_path)

    if self.request.alignment_options.save_raw:
      image_path = storage.subvolume_path(self.request.segmentation_output_dir,
                                          corner, 'align')
      with storage.atomic_file(image_path) as fd:
        np.savez_compressed(fd, im=canvas.image)

    canvas.segment_all(seed_policy=self.get_seed_policy(corner, subvol_size))
    self.save_segmentation(canvas, alignment, seg_path, prob_path)

    # Attempt to remove the checkpoint file now that we no longer need it.
    try:
      gfile.remove(cpoint_path)
    except:  # pylint: disable=bare-except
      pass

    return canvas
