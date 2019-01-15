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
"""Storage-related FFN utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from contextlib import contextmanager
import logging
import json
import os
import re
import tempfile

import h5py
import numpy as np

from tensorflow import gfile
from . import align
from . import segmentation
from ..utils import bounding_box

OriginInfo = namedtuple('OriginInfo', ['start_zyx', 'iters', 'walltime_sec'])


def decorated_volume(settings, **kwargs):
  """Converts DecoratedVolume proto object into volume objects.

  Args:
    settings: DecoratedVolume proto object.
    **kwargs: forwarded to VolumeStore constructor if volinfo volume_path.

  Returns:
    A volume object corresponding to the settings proto.  The returned type
  should support at least __getitem__, shape, and ndim with reasonable numpy
  compatibility.  The returned volume can have ndim in (3, 4).

  Raises:
    ValueError: On bad specification.
  """
  if settings.HasField('volinfo'):
    raise NotImplementedError('VolumeStore operations not available.')
  elif settings.HasField('hdf5'):
    path = settings.hdf5.split(':')
    if len(path) != 2:
      raise ValueError('hdf5 volume_path should be specified as file_path:'
                       'hdf5_internal_dataset_path.  Got: ' + settings.hdf5)
    volume = h5py.File(path[0])[path[1]]
  else:
    raise ValueError('A volume_path must be set.')

  if settings.HasField('decorator_specs'):
    if not settings.HasField('volinfo'):
      raise ValueError('decorator_specs is only valid for volinfo volumes.')
    raise NotImplementedError('VolumeStore operations not available.')

  if volume.ndim not in (3, 4):
    raise ValueError('Volume must be 3d or 4d.')

  return volume


# TODO(mjanusz): Consider switching to pyglib.atomic_file.
@contextmanager
def atomic_file(path, mode='w+b'):
  """Atomically saves data to a target path.

  Any existing data at the target path will be overwritten.

  Args:
    path: target path at which to save file
    mode: optional mode string

  Yields:
    file-like object
  """
  with tempfile.NamedTemporaryFile(mode=mode) as tmp:
    yield tmp
    tmp.flush()
    # Necessary when the destination is on CNS.
    gfile.Copy(tmp.name, '%s.tmp' % path, overwrite=True)
  gfile.Rename('%s.tmp' % path, path, overwrite=True)


def quantize_probability(prob):
  """Quantizes a probability map into a byte array."""
  ret = np.digitize(prob, np.linspace(0.0, 1.0, 255))

  # Digitize never uses the 0-th bucket.
  ret[np.isnan(prob)] = 0
  return ret.astype(np.uint8)


def dequantize_probability(prob):
  """Dequantizes a byte array representing a probability map."""
  dq = 1.0 / 255
  ret = ((prob - 0.5) * dq).astype(np.float32)
  ret[prob == 0] = np.nan
  return ret


def save_subvolume(labels, origins, output_path, **misc_items):
  """Saves an FFN subvolume.

  Args:
    labels: 3d zyx number array with the segment labels
    origins: dictionary mapping segment ID to origin information
    output_path: path at which to save the segmentation in the form
        of a .npz file
    **misc_items: (optional) additional values to save
        in the output file
  """
  seg = segmentation.reduce_id_bits(labels)
  gfile.MakeDirs(os.path.dirname(output_path))
  with atomic_file(output_path) as fd:
    np.savez_compressed(fd,
                        segmentation=seg,
                        origins=origins,
                        **misc_items)


def legacy_subvolume_path(output_dir, corner, suffix):
  """Returns an old-style path to a file with FFN subvolume data.

  Args:
    output_dir: directory containing subvolume data
    corner: (z, y, x) subvolume corner
    suffix: file suffix

  Returns:
    subvolume file path (string)
  """
  return os.path.join(output_dir, 'seg-%s.%s' % (
      '_'.join([str(x) for x in corner[::-1]]), suffix))


def subvolume_path(output_dir, corner, suffix):
  """Returns path to a file with FFN subvolume data.

  Args:
    output_dir: directory containing subvolume data
    corner: (z, y, x) subvolume corner
    suffix: file suffix

  Returns:
    subvolume file path (string)
  """
  return os.path.join(
      output_dir, str(corner[2]), str(corner[1]),
      'seg-%s.%s' % ('_'.join([str(x) for x in corner[::-1]]), suffix))


def get_corner_from_path(path):
  """Returns subvolume corner as (z, y, x)."""
  match = re.search(r'(\d+)_(\d+)_(\d+).npz', os.path.basename(path))
  if match is None:
    raise ValueError('Unrecognized path: %s' % path)
  coord = tuple([int(x) for x in match.groups()])
  return coord[::-1]


def get_existing_corners(segmentation_dir):
  corners = []
  # Legacy path format.
  for path in gfile.Glob(os.path.join(segmentation_dir, 'seg-*_*_*.npz')):
    corners.append(get_corner_from_path(path))
  for path in gfile.Glob(os.path.join(segmentation_dir, '*/*/seg-*_*_*.npz')):
    corners.append(get_corner_from_path(path))
  return corners


def checkpoint_path(output_dir, corner):
  return subvolume_path(output_dir, corner, 'cpoint')


def segmentation_path(output_dir, corner):
  return subvolume_path(output_dir, corner, 'npz')


def object_prob_path(output_dir, corner):
  return subvolume_path(output_dir, corner, 'prob')


def legacy_segmentation_path(output_dir, corner):
  return legacy_subvolume_path(output_dir, corner, 'npz')


def legacy_object_prob_path(output_dir, corner):
  return legacy_subvolume_path(output_dir, corner, 'prob')


def get_existing_subvolume_path(segmentation_dir, corner, allow_cpoint=False):
  """Returns the path to an existing FFN subvolume.

  This like `get_subvolume_path`, but returns paths to existing data only.

  Args:
    segmentation_dir: directory containing FFN subvolumes
    corner: lower corner of the FFN subvolume as a (z, y, x) tuple
    allow_cpoint: whether to return a checkpoint path in case the final
        segmentation is not ready

  Returns:
    path to an existing FFN subvolume (string) or None if no such subvolume
    is found
  """
  target_path = segmentation_path(segmentation_dir, corner)
  if gfile.Exists(target_path):
    return target_path

  target_path = legacy_segmentation_path(segmentation_dir, corner)
  if gfile.Exists(target_path):
    return target_path

  if allow_cpoint:
    target_path = checkpoint_path(segmentation_dir, corner)
    if gfile.Exists(target_path):
      return target_path

  return None


def threshold_segmentation(segmentation_dir, corner, labels, threshold):
  prob_path = object_prob_path(segmentation_dir, corner)
  if not gfile.Exists(prob_path):
    prob_path = legacy_object_prob_path(segmentation_dir, corner)
    if not gfile.Exists(prob_path):
      raise ValueError('Cannot find probability map %s' % prob_path)

  with gfile.Open(prob_path, 'rb') as f:
    data = np.load(f)
    if 'qprob' not in data:
      raise ValueError('Invalid FFN probability map.')

    prob = dequantize_probability(data['qprob'])
    labels[prob < threshold] = 0


def load_origins(segmentation_dir, corner):
  target_path = get_existing_subvolume_path(segmentation_dir, corner, False)
  if target_path is None:
    raise ValueError('Segmentation not found: %s, %s' % (segmentation_dir,
                                                         corner))

  with gfile.Open(target_path, 'rb') as f:
    data = np.load(f)
    return data['origins'].item()


def clip_subvolume_to_bounds(corner, size, volume):
  """Clips a subvolume bounding box to the image volume store bounds.

  Args:
    corner: start of a subvolume (z, y, x)
    size: size of a subvolume (z, y, x)
    volume: a Volume to which the subvolume bounds are to be clipped

  Returns:
    corner: the corner argument, clipped to the volume bounds
    size: the size argument, clipped to the volume bounds
  """
  volume_size = volume.shape
  if volume.ndim == 4:
    volume_size = volume_size[1:]
  volume_bounds = bounding_box.BoundingBox(start=(0, 0, 0), size=volume_size)
  subvolume_bounds = bounding_box.BoundingBox(start=corner, size=size)
  clipped_bounds = bounding_box.intersection(volume_bounds, subvolume_bounds)
  return clipped_bounds.start, clipped_bounds.size


def build_mask(masks, corner, subvol_size, mask_volume_map=None,
               image=None, alignment=None):
  """Builds a boolean mask.

  Args:
    masks: iterable of MaskConfig protos
    corner: lower corner of the subvolume for which to build the
        mask, as a (z, y, x) tuple
    subvol_size: size of the subvolume for which to build the mask,
        as a (z, y, x) tuple
    mask_volume_map: optional dict mapping volume proto hashes to open
        volumes; use this as a cache to avoid opening volumes
        multiple times.
    image: 3d image ndarray; only needed if the mask config uses
        the image as input
    alignment: optional Alignemnt object

  Returns:
    boolean mask built according to the specified config
  """
  final_mask = None
  if mask_volume_map is None:
    mask_volume_map = {}

  if alignment is None:
    alignment = align.Alignment(corner, subvol_size)  # identity

  src_corner, src_size = alignment.expand_bounds(
      corner, subvol_size, forward=False)
  for config in masks:
    curr_mask = np.zeros(subvol_size, dtype=np.bool)

    source_type = config.WhichOneof('source')
    if source_type == 'coordinate_expression':
      # pylint:disable=eval-used,unused-variable
      z, y, x = np.mgrid[src_corner[0]:src_corner[0] + src_size[0],
                         src_corner[1]:src_corner[1] + src_size[1],
                         src_corner[2]:src_corner[2] + src_size[2]]
      bool_mask = eval(config.coordinate_expression.expression)
      # pylint:enable=eval-used,unused-variable
      curr_mask |= alignment.align_and_crop(
          src_corner, bool_mask, corner, subvol_size)
    else:
      if source_type == 'image':
        channels = config.image.channels
        mask = image[np.newaxis, ...]
      elif source_type == 'volume':
        channels = config.volume.channels

        volume_key = config.volume.mask.SerializeToString()
        if volume_key not in mask_volume_map:
          mask_volume_map[volume_key] = decorated_volume(config.volume.mask)
        volume = mask_volume_map[volume_key]

        clipped_corner, clipped_size = clip_subvolume_to_bounds(
            src_corner, src_size, volume)
        clipped_end = clipped_corner + clipped_size
        mask = volume[:,  #
                      clipped_corner[0]:clipped_end[0],  #
                      clipped_corner[1]:clipped_end[1],  #
                      clipped_corner[2]:clipped_end[2]]
      else:
        logging.fatal('Unsupported mask source: %s', source_type)

      for chan_config in channels:
        channel_mask = mask[chan_config.channel, ...]
        channel_mask = alignment.align_and_crop(
            src_corner, channel_mask, corner, subvol_size)

        if chan_config.values:
          bool_mask = np.in1d(channel_mask,
                              chan_config.values).reshape(channel_mask.shape)
        else:
          bool_mask = ((channel_mask >= chan_config.min_value) &
                       (channel_mask <= chan_config.max_value))
        if chan_config.invert:
          bool_mask = np.logical_not(bool_mask)

        curr_mask |= bool_mask

    if config.invert:
      curr_mask = np.logical_not(curr_mask)

    if final_mask is None:
      final_mask = curr_mask
    else:
      final_mask |= curr_mask

  return final_mask


def load_segmentation(segmentation_dir, corner, allow_cpoint=False,
                      threshold=None, split_cc=True, min_size=0,
                      mask_config=None):
  """Loads segmentation from an FFN subvolume.

  Args:
    segmentation_dir: directory containing FFN subvolumes
    corner: lower corner of the FFN subvolume as a (z, y, x) tuple
    allow_cpoint: whether to return incomplete segmentation from a checkpoint
        when a final segmentation is not available
    threshold: optional probability threshold at which to generate the
        segmentation; in order for this to work, the probability file must
        be present, and the segmentation in the main FFN subvolume file must
        have been generated at a threshold lower than the one requested now
    split_cc: whether to recompute connected components within the subvolume
    min_size: minimum (post-CC, if enabled) segment size in voxels; if 0,
        no size filtering is done
    mask_config: optional MaskConfigs proto specifying the mask to apply
        to the loaded subvolume

  Returns:
    tuple of:
      3d uint64 numpy array with segmentation labels,
      dictionary mapping segment IDs to information about their origins.
      This is currently a tuple of (seed location in x, y, z;
      number of FFN iterations used to produce the segment;
      wall clock time in seconds used for inference).

  Raises:
    ValueError: when requested segmentation cannot be found
  """
  target_path = get_existing_subvolume_path(segmentation_dir, corner,
                                            allow_cpoint)
  if target_path is None:
    raise ValueError('Segmentation not found, %s, %r.' %
                     (segmentation_dir, corner))

  with gfile.Open(target_path, 'rb') as f:
    data = np.load(f)
    if 'segmentation' in data:
      seg = data['segmentation']
    else:
      raise ValueError('FFN NPZ file %s does not contain valid segmentation.' %
                       target_path)

    origins = data['origins'].item()
    output = seg.astype(np.uint64)

    logging.info('loading segmentation from: %s', target_path)

    if threshold is not None:
      logging.info('thresholding at %f', threshold)
      threshold_segmentation(segmentation_dir, corner, output, threshold)

    if mask_config is not None:
      mask = build_mask(mask_config.masks, corner, seg.shape)
      output[mask] = 0

    if split_cc or min_size:
      logging.info('clean up with split_cc=%r, min_size=%d', split_cc,
                   min_size)
      new_to_old = segmentation.clean_up(output, split_cc,
                                         min_size,
                                         return_id_map=True)
      new_origins = {}
      for new_id, old_id in new_to_old.items():
        if old_id in origins:
          new_origins[new_id] = origins[old_id]

      origins = new_origins

  return output, origins


def load_segmentation_from_source(source, corner):
  """Loads an FFN segmentation subvolume.

  Args:
    source: SegmentationSource proto
    corner: (z, y, x) subvolume corner

  Returns:
    see the return value of `load_segmentation`
  """
  kwargs = {}
  if source.HasField('threshold'):
    kwargs['threshold'] = source.threshold
  if source.HasField('split_cc'):
    kwargs['split_cc'] = source.split_cc
  if source.HasField('min_size'):
    kwargs['min_size'] = source.min_size
  if source.HasField('mask'):
    kwargs['mask_config'] = source.mask

  return load_segmentation(source.directory, corner, **kwargs)
