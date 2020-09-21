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
"""Tensorflow Python ops and utilities for generating network inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import tensorflow as tf

from tensorflow import gfile
from ..utils import bounding_box


def create_filename_queue(coordinates_file_pattern, shuffle=True):
  """Creates a queue for reading coordinates from coordinate file.

  Args:
    coordinates_file_pattern: File pattern for TFRecords of
                              input examples of the form of a glob
                              pattern or path@shards.
    shuffle: Whether to shuffle the coordinate file list. Note that the expanded
             coordinates_file_pattern is not guaranteed to be sorted
             alphabetically.

  Returns:
    Tensorflow queue with coordinate filenames
  """
  m = re.search(r'@(\d{1,})', coordinates_file_pattern)
  if m:
    num_shards = int(m.group(1))
    coord_file_list = [
      re.sub(r'@(\d{1,})', '-%.5d-of-%.5d' % (i, num_shards),
             coordinates_file_pattern)
      for i in range(num_shards)]
  else:
    coord_file_list = gfile.Glob(coordinates_file_pattern)
  return tf.train.string_input_producer(coord_file_list, shuffle=shuffle)


def load_patch_coordinates_from_filename_queue(filename_queue):
  """Loads coordinates and volume names from filename queue.

  Args:
    filename_queue: Tensorflow queue created from create_filename_queue()

  Returns:
    Tuple of coordinates (shape `[1, 3]`) and volume name (shape `[1]`) tensors.
  """
  record_options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  keys, protos = tf.TFRecordReader(options=record_options).read(filename_queue)
  examples = tf.parse_single_example(protos, features=dict(
      center=tf.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
      label_volume_name=tf.FixedLenFeature(shape=[1], dtype=tf.string),
  ))
  coord = examples['center']
  volname = examples['label_volume_name']
  return coord, volname


def load_patch_coordinates(coordinates_file_pattern,
                           shuffle=True,
                           scope='load_patch_coordinates'):
  """Loads coordinates and volume names from tables of VolumeStoreInputExamples.

  Args:
    coordinates_file_pattern: File pattern for TFRecords of
                              input examples of the form of a glob
                              pattern or path@shards.
    shuffle: Whether to shuffle the coordinate file list. Note that the expanded
             coordinates_file_pattern is not guaranteed to be sorted
             alphabetically.
    scope: Passed to name_scope.

  Returns:
    Tuple of coordinates (shape `[1, 3]`) and volume name (shape `[1]`) tensors.
  """
  with tf.name_scope(scope):
    filename_queue = create_filename_queue(
        coordinates_file_pattern, shuffle=shuffle)
    return load_patch_coordinates_from_filename_queue(filename_queue)


def load_from_numpylike(coordinates, volume_names, shape, volume_map,
                        name=None):
  """TensorFlow Python op that loads data from Numpy-like volumes.

  The volume object must support Numpy-like indexing, as well as shape, ndim,
  and dtype properties.  The volume can be 3d or 4d.

  Args:
    coordinates: tensor of shape [1, 3] containing XYZ coordinates of the
        center of the subvolume to load.
    volume_names: tensor of shape [1] containing names of volumes to load data
        from.
    shape: a 3-sequence giving the XYZ shape of the data to load.
    volume_map: a dictionary mapping volume names to volume objects.  See above
        for API requirements of the Numpy-like volume objects.
    name: the op name.

  Returns:
    Tensor result of reading data of shape [1] + shape[::-1] + [num_channels]
  from given center coordinate and volume name.  Dtype matches input volumes.

  Raises:
    ValueError: if volumes in volume_map have inconsistent dtypes or number of
  channels.
  """
  def _num_channels(volume):
    if volume.ndim == 3:
      return 1
    return volume.shape[0]

  # Validate that all volumes are compatible.
  volumes = iter(volume_map.values())
  first_vol = next(volumes)
  dtype = first_vol.dtype
  num_channels = _num_channels(first_vol)
  for volume in volumes:
    if volume.dtype != dtype:
      raise ValueError('All volumes should have same dtype.')
    if _num_channels(volume) != num_channels:
      raise ValueError('All volumes should have same number of channels.')

  start_offset = (np.array(shape) - 1) // 2
  def _load_from_numpylike(coord, volname):
    """Load from coord and volname, handling 3d or 4d volumes."""
    volume = volume_map[volname.decode('ascii')]
    # Get data, including all channels if volume is 4d.
    starts = np.array(coord) - start_offset
    slc = bounding_box.BoundingBox(start=starts, size=shape).to_slice()
    if volume.ndim == 4:
      slc = np.index_exp[:] + slc
    data = volume[slc]

    # If 4d, move channels to back.  Otherwise, just add flat channels dim.
    if data.ndim == 4:
      data = np.rollaxis(data, 0, start=4)
    else:
      data = np.expand_dims(data, data.ndim)

    # Add flat batch dim and return.
    data = np.expand_dims(data, 0)
    return data

  with tf.name_scope(name, 'LoadFromNumpyLike',
                     [coordinates, volume_names]) as scope:
    # For historical reasons these have extra flat dims.
    coordinates = tf.squeeze(coordinates, axis=0)
    volume_names = tf.squeeze(volume_names, axis=0)

    loaded = tf.py_func(
        _load_from_numpylike, [coordinates, volume_names], [dtype],
        name=scope)[0]
    loaded.set_shape([1] + list(shape[::-1]) + [num_channels])
    return loaded


def get_offset_scale(volname,
                     offset_scale_map=(),
                     default_offset=0.0,
                     default_scale=1.0,
                     name='get_offset_scale'):
  """Gets offset and scale from map matching volname, or defaults.

  Args:
    volname: scalar string tensor (note LoadPatchCoordinates returns a
             1-vector instead).
    offset_scale_map: map of string volnames to (offset, scale) pairs.
    default_offset: used if volname is not in offset_scale_map.
    default_scale: used if volname is not in offset_scale_map.
    name: scope name.

  Returns:
    Tuple of offset, scale scalar float32 tensors.
  """

  def _get_offset_scale(volname):
    if volname in offset_scale_map:
      offset, scale = offset_scale_map[volname]
    else:
      offset = default_offset
      scale = default_scale
    return np.float32(offset), np.float32(scale)

  offset, scale = tf.py_func(
      _get_offset_scale, [volname], [tf.float32, tf.float32],
      stateful=False,
      name=name)
  offset.set_shape([])
  scale.set_shape([])
  return offset, scale


def offset_and_scale_patches(patches,
                             volname,
                             offset_scale_map=(),
                             default_offset=0.0,
                             default_scale=1.0,
                             scope='offset_and_scale_patches'):
  """Apply offset and scale from map matching volname, or defaults.

  Args:
    patches: tensor to apply offset and scale to.
    volname: scalar string tensor (note LoadPatchCoordinates returns a 1-vector
             instead.)
    offset_scale_map: map of string volnames to (offset, scale) pairs.
    default_offset: used if volname is not in offset_scale_map.
    default_scale: used if volname is not in offset_scale_map.
    scope: TensorFlow scope for subops.

  Returns:
    patches cast to float32, less offset, divided by scale for given volname, or
    else defaults.
  """
  with tf.name_scope(scope):
    offset, scale = get_offset_scale(
        volname,
        offset_scale_map=offset_scale_map,
        default_offset=default_offset,
        default_scale=default_scale)
    return (tf.cast(patches, tf.float32) - offset) / scale


def redundant_lom(label, radius, scope='redundant_lom'):
  """Convert label tensor into redundant LOM representation.

  Args:
    label: Tensor with dimensions batch, z, y, x, channels.  Channels should be
           flat.
    radius: 3-sequence of z, y, x LOM radii.
    scope: TF scope for ops.

  Returns:
    Tensor with dimensions batch, z, y, x, lomz, lomy, lomx.  Unfortunately,
    rank 7 tensors are not supported by many TF ops.  Use the helpers below to
    flatten / unflatten either the ZYX or LOM dims.

  Raises:
    ValueError: if input tensor is wrong shape.

  The LOM generated is smaller in z, y, x by 2 * radius.  Each z, y, x location
  has a full complement of lomz, lomy, lomx entries, which means that all the
  affinities except the edges are doubly represented, once at each terminal node
  voxel.

  TODO(phli): Benchmark alternative implementations.
  """
  if len(label.shape_as_list()) != 5:
    raise ValueError(
        'Input tensor must have dimensions batch, z, y, x, channels.')
  if label.shape_as_list()[4] != 1:
    raise ValueError('Input tensor must have single channel.')

  with tf.name_scope(scope):

    # Central crop to be compared to offset crops.
    core_start = [0] + list(radius) + [0]
    core_shape = list(label.shape_as_list())
    core_shape[1] -= 2 * radius[0]
    core_shape[2] -= 2 * radius[1]
    core_shape[3] -= 2 * radius[2]

    core_end = tf.add(core_start, core_shape)
    core = tf.strided_slice(label, core_start, core_end)
    core = tf.reshape(core, core_shape, name='lom_core')

    # Offset crops.  Currently this clobbers the flat channel dimension with the
    # LOMs.
    # TODO(phli): Would be nice to replace this with extract_patches, but that
    # hasn't been exposed in the TF api.
    shifts = []
    dims = lom_dims(radius)
    for z in range(dims[0]):
      for y in range(dims[1]):
        for x in range(dims[2]):
          shifts.append(
              tf.reshape(
                  tf.strided_slice(label, (0, z, y, x, 0),
                                   tf.add((0, z, y, x, 0), core_shape)),
                  core_shape,
                  name='slice_lom_shift'))
    shift_tensor = tf.concat(shifts, 4, name='concat_lom_shifts')

    lom = tf.logical_and(
        tf.equal(core, shift_tensor), core > 0, name='compute_redunant_lom')
    return unravel_lom_dims(lom, radius)


def lom_radius(tensor):
  lomzyx = np.array(tensor.shape_as_list()[-3:])
  if not np.all(lomzyx % 2 == 1):
    raise ValueError('Input tensor does not have compatible LOM dims.')
  return lomzyx // 2


def lom_dims(radius):
  return np.array(radius) * 2 + 1


def unravel_lom_dims(tensor, radius_zyx, name='unravel_lom_dims'):
  """Assumes LOM is flattened in the last dim."""
  return tf.reshape(
      tensor,
      tensor.shape_as_list()[:-1] + list(lom_dims(radius_zyx)),
      name=name)


def ravel_lom_dims(tensor, name='ravel_lom_dims'):
  """Assumes LOM is in the last 3 dims."""
  return tf.reshape(tensor, tensor.shape_as_list()[:-3] + [-1], name=name)


def ravel_zyx_dims(tensor, name='ravel_zyx_dims'):
  """Assumes ZYX are dims 1, 2, 3."""
  return tf.reshape(
      tensor,
      tensor.shape_as_list()[0:1] + [-1] + tensor.shape_as_list()[4:],
      name=name)


def unravel_zyx_dims(tensor, zyxdims, name='unravel_zyx_dims'):
  """Assumes ZYX are flattened in dim 1."""
  return tf.reshape(
      tensor,
      tensor.shape_as_list()[0:1] + list(zyxdims) + tensor.shape_as_list()[2:],
      name=name)


def soften_labels(bool_labels, softness=0.05, scope='soften_labels'):
  """Converts boolean labels into float32.

  Args:
    bool_labels: Tensor with dtype `boolean`
    softness: The float value to use for False.  1 - softness is implicitly used
              for True
    scope: passed to op_scope

  Returns:
    Tensor with same shape as bool_labels with dtype `float32` and values 0.05
    for False and 0.95 for True.
  """
  with tf.op_scope([bool_labels, softness], scope):
    label_shape = tf.shape(bool_labels, name='label_shape')
    return tf.where(bool_labels,
                    tf.fill(label_shape, 1.0 - softness, name='soft_true'),
                    tf.fill(label_shape, softness, name='soft_false'))
