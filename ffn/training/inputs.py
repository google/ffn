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

import random
import re
from typing import Any, Callable, Optional, Sequence

from absl import logging
from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.segmentation import labels as label_utils
from connectomics.volume import metadata
from ffn.training import augmentation
from ffn.training import variables
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.io import gfile


def create_filename_queue(coordinates_file_pattern, shuffle=True):
  """Creates a queue for reading coordinates from coordinate file.

  Args:
    coordinates_file_pattern: File pattern for TFRecords of input examples of
      the form of a glob pattern or path@shards or comma-separated file
      patterns.
    shuffle: Whether to shuffle the coordinate file list. Note that the expanded
      coordinates_file_pattern is not guaranteed to be sorted alphabetically.

  Returns:
    Tensorflow queue with coordinate filenames
  """
  coord_file_list = []
  for pattern in coordinates_file_pattern.split(','):
    m = re.search(r'@(\d{1,})', pattern)
    if m:
      num_shards = int(m.group(1))
      coord_file_list.extend([
          re.sub(
              r'@(\d{1,})',
              '-%.5d-of-%.5d' % (i, num_shards),
              pattern,
          )
          for i in range(num_shards)
      ])
    else:
      coord_file_list.extend(gfile.glob(pattern))
  return tf.train.string_input_producer(coord_file_list, shuffle=shuffle)


def load_patch_coordinates_from_filename_queue(filename_queue,
                                               file_format='tfrecord'):
  """Loads coordinates and volume names from filename queue.

  Args:
    filename_queue: Tensorflow queue created from create_filename_queue()
    file_format: String indicating the format of the files in the queue.
                 Can be 'sstable' or 'tfrecord'. Defaults to 'tfrecord'.

  Returns:
    Tuple of coordinates (shape `[1, 3]`) and volume name (shape `[1]`) tensors.
  """
  if file_format == 'tfrecord':
    record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    _, protos = tf.TFRecordReader(options=record_options).read(filename_queue)
    examples = tf.parse_single_example(protos, features=dict(
        center=tf.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
        label_volume_name=tf.FixedLenFeature(shape=[1], dtype=tf.string),
    ))
    coord = examples['center']
    volname = examples['label_volume_name']
  else:
    raise ValueError(f'Unsupported file format: {file_format}.')

  return coord, volname


def sample_patch_coordinates(
    bboxes: Sequence[Sequence[bounding_box.BoundingBox]],
    volnames: list[str],
    name='sample_patch_coordinates',
    rng_seed: Optional[int] = None,
) -> tf.data.Dataset:
  """Samples a coordinate uniformly at random from specified bboxes.

  Args:
    bboxes: sequence of sequences for bounding boxes (one seq. per volume)
    volnames: a list of volume names
    name: passed to `name_scope`
    rng_seed: Random number generator seed allowing to make the dataset
      deterministic.

  Returns:
    tuple of:
      [1, 3] int64 xyz coord tensor
      [1] string tensor with the volume label

  Raises:
    ValueError: if len(bboxes) != len(volinfo_map) or if an invalid bbox is
        passed
  """
  if len(bboxes) != len(volnames):
    raise ValueError(
        'Numbers of bounding boxes and volume names do not match.'
    )

  volumes, flat_boxes = [], []
  total_voxels = 0
  for vol_id, volume_boxes in enumerate(bboxes):
    for b in volume_boxes:
      w = np.prod(b.size)
      if w < 0:
        raise ValueError('Volume %d, bbox %r is too small.' % (vol_id, b))
      total_voxels += w
      flat_boxes.append(b)
      volumes.append(vol_id)

  calc = box_generator.MultiBoxGenerator(
      flat_boxes, box_size=(1, 1, 1), box_overlap=(0, 0, 0)
  )
  def _sample_volinfo_and_bbox(idx):
    idx = idx[0]
    vol_idx = volumes[calc.index_to_generator_index(idx)[0]]
    _, coord_bbox = calc.generate(idx)
    assert coord_bbox is not None
    logging.log_first_n(
        logging.INFO,
        'Sampled location %r from volume %s',
        2,
        coord_bbox.start,
        volnames[vol_idx])
    coord = np.array([coord_bbox.start]).astype(np.int64)
    return coord, volnames[vol_idx]

  def _sample(rng_seed):
    with tf.name_scope(name=name):
      coord, label = tf.py_func(
          _sample_volinfo_and_bbox,
          [
              tf.random.stateless_uniform(
                  [1],
                  rng_seed,
                  maxval=total_voxels,
                  dtype=tf.int64,
                  name='rand',
              )
          ],
          [tf.int64, tf.string],
          name='sample_volinfo_and_bbox',
          stateful=False,
      )
      label.set_shape([])
      coord.set_shape([1, 3])
      return {'coord': coord, 'volname': tf.reshape(label, [1])}

  # This is faster than calling _sample_volinfo_and_bbox via .from_generator.
  return tf.data.Dataset.random(seed=rng_seed).batch(2).map(_sample)


def get_vol_map(volinfo_paths: Sequence[str]):
  return ','.join(
      'vol%d:%s' % (i, volinfo) for i, volinfo in enumerate(volinfo_paths)
  )


def parse_tf_coords(x):
  return tf.io.parse_single_example(
      x,
      features=dict(
          coord=tf.FixedLenFeature(shape=[1, 3], dtype=tf.int64),
          volname=tf.FixedLenFeature(shape=[1], dtype=tf.string),
          label=tf.FixedLenFeature(shape=[1], dtype=tf.int64),
          segment_id=tf.FixedLenFeature(
              shape=[1],
              dtype=tf.int64,
              default_value=tf.constant([0], dtype=tf.int64),
          ),
          radius=tf.FixedLenFeature(
              shape=[1],
              dtype=tf.float32,
              default_value=tf.constant([0], dtype=tf.float32),
          ),
      ),
  )


def load_patch_coordinates(coordinates_file_pattern,
                           shuffle=True,
                           scope='load_patch_coordinates',
                           file_format='tfrecord'):
  """Loads coordinates and volume names from tables of VolumeStoreInputExamples.

  Args:
    coordinates_file_pattern: File pattern for TFRecords of
                              input examples of the form of a glob
                              pattern or path@shards.
    shuffle: Whether to shuffle the coordinate file list. Note that the expanded
             coordinates_file_pattern is not guaranteed to be sorted
             alphabetically.
    scope: Passed to name_scope.
    file_format: String indicating the format of the files in the queue.
                 Can be 'sstable' or 'tfrecord'. Defaults to 'tfrecord'.

  Returns:
    Tuple of coordinates (shape `[1, 3]`) and volume name (shape `[1]`) tensors.
  """
  with tf.name_scope(scope):
    filename_queue = create_filename_queue(
        coordinates_file_pattern, shuffle=shuffle)
    return load_patch_coordinates_from_filename_queue(
        filename_queue, file_format=file_format)


def weighted_load_patch_coordinates(
    coord_paths: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    scope: str = 'weighted_load_patch_coordinates',
    file_format: str = 'tfrecord',
):
  """Like the unweighted version, but pulls data from multiple sources.

  Args:
    coord_paths: glob patterns for files containing volume input examples
    weights: weights determining the relative frequency the corresponding paths
      will be sampled; needs to be same length as coord_paths; weights do not
      need to be normalized
    scope: passed to name_scape
    file_format: String indicating the format of the files in the queue.
                 Can be 'sstable' or 'tfrecord'. Defaults to 'tfrecord'.

  Returns:
    TF op to pull a tuple of coordinates and volume name from a queue.
  """
  if weights is None:
    weights = [1.0] * len(coord_paths)
  if len(coord_paths) != len(weights):
    raise ValueError(
        '# of coord paths: %d does not match # of weights %d'
        % (len(coord_paths), len(weights))
    )

  weights = np.array(weights)
  weights /= np.sum(weights)
  cum_weights = np.cumsum(weights)

  with tf.name_scope(scope):
    # Filename queues have to be created in the main graph, outside of
    # tf.switch_case branches.
    load_queues = []
    for path in coord_paths:
      load_queues.append(create_filename_queue(path, shuffle=True))

    with tf.variable_scope(None, 'coord_source'):
      dist = variables.DistributionTracker(len(coord_paths))
      rates = dist.get_rates()
      for i in range(len(coord_paths)):
        tf.summary.scalar('source_%d' % i, rates[i])

      # Choose source at random and pull coordinates from the associated queue.
      source_num = tf.cast(
          tf.reduce_min(tf.where(tf.random.uniform(shape=[1]) < cum_weights)),
          tf.int32,
      )
      with tf.control_dependencies([dist.record_class(source_num)]):
        return tf.switch_case(
            source_num,
            [
                lambda qq=q: load_patch_coordinates_from_filename_queue(
                    qq, file_format=file_format)
                for q in load_queues
            ],
            # Use a default invalid value so that the process crashes if
            # no valid case is found instead of silently selecting the
            # last branch.
            default=lambda: (  # pylint:disable=g-long-lambda
                tf.constant([[0, 0, 0]], dtype=tf.int64),
                tf.constant(['missing'], dtype=tf.string),
            ),
        )


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
    slc = bounding_box.BoundingBox(start=starts, size=shape).to_slice3d()
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
    volname = volname.decode('utf-8')
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


def make_labels_contiguous(labels: tf.Tensor) -> tf.Operation:
  """Maps the labels to [0..N].

  Args:
    labels: [1, z, y, x, 1] int64 tensor of labels

  Returns:
    labels mapped to the range [0..N] if N distinct non-zero values are
    present in the input tensor
  """
  ret = tf.py_func(
      label_utils.make_contiguous,
      inp=[labels],
      Tout=tf.int64,
      name='make_labels_contiguous',
  )
  ret.set_shape(labels.shape)
  return ret


def apply_augmentation(
    data: dict[str, Any],
    section_augment: bool,
    section_augmentation_args: Optional[dict[str, Any]],
    permute_and_reflect_augment: bool,
    permutable_axes: list[int],
    reflectable_axes: list[int],
    rotation_augmentation: Optional[str],
    voxel_size: Optional[tuple[float, float, float]],
) -> dict[str, Any]:
  """Applies augmentations to a subvolume of data and corresponding labels.

  Args:
    data: dict containing at least 'labels' and 'patches' tensors
    section_augment: whether to apply section augmentations
    section_augmentation_args: kwargs for
      augmentation.apply_section_augmentations
    permute_and_reflect_augment: whether to apply permutation/reflection
    permutable_axes: list of axes to permute
    reflectable_axes: list of axes to reflect
    rotation_augmentation: type of rotation augmenation to perform ('2d', '3d')
    voxel_size: xyz voxel size of the input data (only needed when applying
      rotation augmentation

  Returns:
    'data' dict with 'labels' and 'patches' entries updated according to the
    chosen augmentations
  """
  labels = data['labels']
  patches = data['patches']

  # Apply section-wise augmentations.
  if section_augment:
    final_data_zyx = patches.shape_as_list()[1:4]
    final_label_zyx = labels.shape_as_list()[1:4]
    patches, labels, _ = augmentation.apply_section_augmentations(
        patches,
        labels,
        labels,
        final_data_zyx,
        final_label_zyx,
        final_label_zyx,
        **section_augmentation_args,
    )

  # Apply basic augmentations.
  if permute_and_reflect_augment:
    transform_axes = augmentation.PermuteAndReflect(
        rank=5,
        permutable_axes=permutable_axes,
        reflectable_axes=reflectable_axes,
    )
    labels = transform_axes(labels)
    patches = transform_axes(patches)

  rot_mtx = None
  if rotation_augmentation == '2d':
    rot_mtx = augmentation.random_2d_rotation_matrix()
  elif rotation_augmentation == '3d':
    rot_mtx = augmentation.random_3d_rotation_matrix()

  if rot_mtx is not None:
    if labels.dtype == tf.int64:
      labels = tf.cond(
          tf.reduce_any(labels > np.iinfo(np.int32).max),  #
          lambda: make_labels_contiguous(labels),  #
          lambda: labels,
      )
      labels = tf.cast(labels, tf.int32)

    assert voxel_size is not None
    patches = augmentation.apply_rotation(patches, rot_mtx, voxel_size)
    if labels.shape.as_list() != [1, 1, 1, 1, 1]:
      labels = augmentation.apply_rotation(labels, rot_mtx, voxel_size)

  data['labels'] = labels
  data['patches'] = patches
  return data


def interleave(datasets: Sequence[tf.data.Dataset], repeat=True):
  """Interleave two or more datasets together, one at a time.

  Interleaves two independently generated datasets together, contrary to
  Dataset.interleave which interleaves new Datasets generated from each input
  item.

  Args:
    datasets: Sequence of datasets to interleave.
    repeat: repeat the interleaved sequence.

  Returns:
    tf.data.Dataset with interleaved results.
  """
  choice_dataset = tf.data.Dataset.range(len(datasets))
  if repeat:
    choice_dataset = choice_dataset.repeat()
  return tf.data.experimental.choose_from_datasets(datasets, choice_dataset)


def sample(
    datasets: Sequence[tf.data.Dataset],
    repeat=True,
    weights: Optional[Sequence[float]] = None,
):
  """Weighted sample of two or more datasets.

  Args:
    datasets: Sequence of datasets to sample.
    repeat: repeat the sampled sequence.
    weights: relative weight of each respective dataset.

  Returns:
    tf.data.Dataset with sampled results.
  """
  if weights is None:
    weights = [1.0] * len(datasets)
  sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights)
  if repeat:
    sampled_dataset = sampled_dataset.repeat()
  return sampled_dataset


def coordinates_in_bounds(
    coordinates: tf.Tensor,
    volname: tf.Tensor,
    radius: Sequence[int],
    volinfo_map_string: str,
    use_bboxes: bool = True,
    name: str = 'coordinates_in_bounds',
) -> tf.Tensor:
  """Tensorflow Python Op returning boolean whether coordinates are in bounds.

  Args:
    coordinates: int64 Tensor of shape `[1, 3]` representing center coordinates
      from which to retrieve patches.
    volname: string Tensor of shape `[1]` giving volume from which patch should
      be retrieved.
    radius: length 3 sequence indicating the radius of the patches to be
      retrieved around each coordinates (xyz).
    volinfo_map_string: comma delimited string mapping volname:volinfo_path,
      where volinfo_path points to the metadata of the volume from which 
      patches should be extracted.
    use_bboxes: whether to use the bounding boxes declared in the volume; if
      False, the physical size of the volume is used as the bounding box.
    name: passed to name_scope.

  Returns:
    Boolean scalar Tensor indicating whether the patch specified by coordinates
    and radius fits within the volume specified by volname and
    volinfo_map_string.  This can be passed to tf.cond to select either
    coordinates or an empty constant of shape `[0, 3]`, which can then be
    passed to batching (e.g. see tests).
  """
  boxes_by_volname = {}
  for mapping in volinfo_map_string.split(','):
    k, volinfo_path = mapping.split(':')
    k = k.encode('utf-8')
    assert k not in boxes_by_volname

    if volinfo_path.endswith('metadata.json'):
      f = open(volinfo_path, 'r')
      meta = metadata.VolumeMetadata.from_json(f.read())
      if use_bboxes:
        bboxes = meta.bounding_boxes
      else:
        bboxes = [
            bounding_box.BoundingBox(
                (0, 0, 0),
                (meta.volume_size.x, meta.volume_size.y, meta.volume_size.z),
            )
        ]
      boxes_by_volname[k] = bboxes

  if not boxes_by_volname:
    raise ValueError('boxes_by_volname is empty.')

  def _in_bounds_fn(coordinates, volname):
    boxes = boxes_by_volname[volname[0]]
    patch_start = np.array(coordinates) - radius
    patch_back = np.array(coordinates) + radius
    for box in boxes:
      if (box.start <= patch_start).all() and (box.end > patch_back).all():
        return True
    return False

  with tf.name_scope(name, values=[coordinates, volname]) as scope:
    assert coordinates.shape_as_list() == [1, 3]
    assert volname.shape_as_list() == [1]
    in_bounds = tf.py_func(
        _in_bounds_fn,
        [coordinates, volname],
        [tf.bool],
        name=scope,
        stateful=False,
    )[0]
    in_bounds.set_shape([])
    return in_bounds


def filter_oob(
    item: dict[str, tf.Tensor], volinfo_map_string: str, patch_size: list[int],
    use_bboxes: bool = True
) -> tf.Tensor:
  radius = np.floor_divide(patch_size, 2)
  coord = tf.reshape(item['coord'], [1, 3])
  volname = tf.reshape(item['volname'], [1])
  return coordinates_in_bounds(
      coord, volname, radius, volinfo_map_string,
      use_bboxes
  )


def make_oob_mask(
    coordinates: tf.Tensor,
    volname: tf.Tensor,
    volinfo_map_string: str,
    radius: Optional[Sequence[int]] = None,
    shape: Optional[Sequence[int]] = None,
    name='make_oob_mask',
):
  """Builds a tensor masking voxels that are outside of bounding boxes.

  Args:
    coordinates: int64 Tensor of shape `[1, 3]` representing center coordinates
      from which to retrieve patches.
    volname: string Tensor of shape `[1]` giving volume from which patch should
      be retrieved.
    volinfo_map_string: comma delimited string mapping volname:volinfo_path,
      where volinfo_path is a gfile with text_format VolumeInfo proto for the
      volume from which patches should be extracted.
    radius: XYZ radius of patches to extract; exclusive with 'shape'
    shape: XYZ shape of patches to extract; exclusive with 'radius'
    name: passed to name_scope.

  Returns:
    float32 tensor of shape [1, dz, dy, dx, 1], where every voxel has one of
    two values:
      1: if the voxel is inside one or more bounding boxes associated with
         the volume specified by `volname`
      0: otherwise
    and where (dx, dy, dz) = 2 * radius + 1.
  """
  boxes_by_volname = {}
  for mapping in volinfo_map_string.split(','):
    k, volinfo_path = mapping.split(':')
    k = k.encode('utf-8')
    assert k not in boxes_by_volname

    if volinfo_path.endswith('metadata.json'):
      f = open(volinfo_path, 'r')
      meta = metadata.VolumeMetadata.from_json(f.read())
      boxes_by_volname[k] = meta.bounding_boxes
  if shape is None:
    assert radius is not None
    diameter_xyz = np.array(radius) * 2 + 1
  else:
    assert radius is None
    diameter_xyz = np.array(shape)
    radius = diameter_xyz // 2

  mask_shape = [1] + diameter_xyz.tolist()[::-1] + [1]

  def _oob_mask_fn(coordinates, volname):  # pylint:disable=missing-docstring
    boxes = boxes_by_volname[volname[0]]
    patch_box = bounding_box.BoundingBox(
        start=np.array(coordinates[0, :]) - radius, size=diameter_xyz
    )
    oob_mask = np.zeros(mask_shape, dtype=np.float32)
    for box in boxes:
      ibox = patch_box.intersection(box)
      if ibox is None:
        continue
      rel_ibox = bounding_box.BoundingBox(
          start=ibox.start - patch_box.start, size=ibox.size
      )
      oob_mask[np.index_exp[:] + rel_ibox.to_slice3d() + np.index_exp[:]] = 1
    return oob_mask

  with tf.name_scope(name, values=[coordinates, volname]) as scope:
    assert coordinates.shape_as_list() == [1, 3]
    assert volname.shape_as_list() == [1]
    oob_mask = tf.py_func(
        _oob_mask_fn, [coordinates, volname], [tf.float32], name=scope
    )[0]
    oob_mask.set_shape(mask_shape)
    return oob_mask
