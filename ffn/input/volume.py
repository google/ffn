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

"""Input pipeline components for loading volumetric data."""

import copy
import dataclasses
import functools as ft
from typing import Any, Callable, Sequence, TypeVar

from absl import logging
import array_record
from connectomics.common import array
from connectomics.common import bounding_box
from connectomics.common import io_utils
from ffn.input import segmentation
from ffn.training import augmentation
from ffn.training import inputs
from ffn.training import mask
import numpy as np
import tensorflow as tf


tf.config.threading.set_inter_op_parallelism_threads(128)

T = TypeVar('T')
Example = dict[str, tf.Tensor]
ExampleTransform = Callable[[Example], Example]


@dataclasses.dataclass
class AugmentationConfig:
  """Specifies how to modify the loaded data."""

  # Axis numbers refer to a 5D tensor with the following axes:
  # [batch, z, y, x, channels]
  permutable_axes: Sequence[int] = (2, 3)
  reflectable_axes: Sequence[int] = (1, 2, 3)

  # If rotation augmentation is requested, int64 (segmnentation)
  # arrays will be cast to int32, with values mapped to the lower
  # end of the 0..2^31-1 range if necessary.
  rotation: str | None = None

  # Contrast adjustment, a random contrast factor will be generated between
  # the tuple of min and max values:
  # (min_contrast_factor, max_contrast_factor).
  contrast_factor_range: tuple[float, float] | None = None
  # Brightness adjustment, a random delta factor will be generated between
  # the tuple of min and max values:
  # (min_brightness_factor, max_brightness_factor).
  brightness_factor_range: tuple[float, float] | None = None

  # Specify if the brightness and contrast adjustment will be only applied
  # to either foreground or background voxels.
  # The available options so far are ['foreground', 'background', None].
  # where None means the adjustment will be applied to all voxels.
  apply_adjustment_to: str | None = None


TupleXYZ = array.Tuple3i


@dataclasses.dataclass
class SamplingConfig:
  """Specifies where to load the data from."""

  # Axis aligned bounding box to sample from. Each volume may be configured with
  # multiple bounding boxes. The configuration consists of a map from strings
  # identifying volumes to lists of bounding boxes in each volume to sample
  # from.
  bounding_boxes: dict[str, Sequence[bounding_box.BoundingBox]] | None = None

  # As above, but the file format is ArrayRecord or Bag.
  arrayrecord_coords: dict[str, float] | str | None = None
  bag_coords: dict[str, float] | str | None = None


@dataclasses.dataclass
class VolumeConfig:
  """Describes a source of volumetric data."""

  # It is assumed that all volumes in this dictionary have the same
  # voxel size, data type, and number of channels. If the sampling
  # scheme uses volume names, these have to match the keys in this dict.
  paths: dict[str, str]  # volname -> volinfo

  # Shape of the data to load. Given a sampled location x, data is loaded
  # from a bounding box starting at x - load_shape // 2.
  load_shape: TupleXYZ

  filter_shape: TupleXYZ | None = None
  default_value: Any | None = None

  # If true, generate OOB mask instead of actually loading data. The
  # bounding boxes specified in the VolumeInfo are used to decide which
  # voxels are considered 'in bounds'.
  oob_mask: bool = False

  # If true, this is a volume to which photometric augmentations should
  # be applied.
  photometric: bool = False

  # Paths to text files of relabel maps in the format of "old_id,new_id"
  # (per-line). Only applies to segmentation (uint64) volumes. The keys
  # of the dictionary should match those in 'patch'.
  relabel_maps: dict[str, str] | None = None

  # Number of bytes to use for the in-memory cache.
  cache_bytes: int = 1_000_000_000


@dataclasses.dataclass
class InputConfig:
  """Describes an input pipeline with volumetric data sources."""

  sampling: SamplingConfig
  # The keys in the dictionary define the name under which the loaded
  # data will be available.
  volumes: dict[str, VolumeConfig]
  augmentation: AugmentationConfig
  read_parallelism: int = 64
  augment_parallelism: int = 64


def _update_config_for_augmentation(
    config: InputConfig, voxel_size: tuple[float, float, float]
) -> InputConfig:
  """Updates spatial sizes to account for augmentations."""
  config = copy.deepcopy(config)

  if config.augmentation.rotation in ('2d', '3d'):
    for vol_cfg in config.volumes.values():
      if vol_cfg.load_shape != (1, 1, 1):
        vol_cfg.load_shape = tuple(
            augmentation.input_size_for_rotated_output(
                vol_cfg.load_shape, voxel_size
            )
        )

      if vol_cfg.filter_shape != (1, 1, 1) and vol_cfg.filter_shape is not None:
        vol_cfg.filer_shape = tuple(
            augmentation.input_size_for_rotated_output(
                vol_cfg.filter_shape, voxel_size
            )
        )

  return config


def _postprocess_augmented_data(
    ds: tf.data.Dataset, config: InputConfig
) -> tf.data.Dataset:
  """Center-crops augmented data if necessary."""
  if config.augmentation.rotation not in ('2d', '3d'):
    return ds

  for name, vol_cfg in config.volumes.items():
    if ds.element_spec[name][1:4] != vol_cfg.load_shape[::-1]:
      shape = vol_cfg.load_shape

      def _update_array(x, name=name, shape=shape):
        setattr(x, name, mask.crop(x[name], (0, 0, 0), shape))
        return x

      ds = ds.map(
          _update_array, num_parallel_calls=tf.data.experimental.AUTOTUNE
      )

  return ds


def get_path_str(paths: dict[str, str]) -> str:
  return ','.join('%s:%s' % (k, v) for k, v in paths.items())


def _load_data(
    ex: dict[str, tf.Tensor], config: InputConfig
) -> dict[str, tf.Operation | tf.Tensor]:
  """Pulls volumetric data from storage."""
  ret = dict(ex)
  dtype_remap = {tf.uint64: tf.int64}

  for name, vol in config.volumes.items():
    if vol.oob_mask:
      ret[name] = inputs.make_oob_mask(
          ex['coord'],
          ex['volname'],
          shape=vol.load_shape,
          volinfo_map_string=get_path_str(vol.paths),
      )
    else:
      raise NotImplementedError

  return ret


def _apply_augmentations(
    data: dict[str, tf.Tensor],
    config: InputConfig,
    voxel_size: tuple[float, float, float],
) -> dict[str, tf.Tensor]:
  """Applies augmentations to all loaded volumetric data."""
  if (config.augmentation.contrast_factor_range) or (
      config.augmentation.brightness_factor_range
  ):
    logging.info('Applying contrast or brightness data augmentation.')
    data['em'] = augmentation.random_contrast_brightness_adjustment(
        data['em'],
        data['seg'],
        config.augmentation.contrast_factor_range,
        config.augmentation.brightness_factor_range,
        config.augmentation.apply_adjustment_to,
    )

  transform_axes = augmentation.PermuteAndReflect(
      rank=5,
      permutable_axes=config.augmentation.permutable_axes,
      reflectable_axes=config.augmentation.reflectable_axes,
  )

  for name in config.volumes.keys():
    tf.debugging.assert_rank(data[name], 5)
    data[name] = transform_axes(data[name])

  rot_mtx = None
  if config.augmentation.rotation == '2d':
    rot_mtx = augmentation.random_2d_rotation_matrix()
  elif config.augmentation.rotation == '3d':
    rot_mtx = augmentation.random_3d_rotation_matrix()

  if rot_mtx is not None:
    assert voxel_size is not None
    for name in config.volumes.keys():
      arr = data[name]
      if arr.dtype == tf.int64:
        arr = tf.cond(
            tf.reduce_any(arr > np.iinfo(np.int32).max),  #
            lambda arr=arr: inputs.make_labels_contiguous(arr),  #
            lambda arr=arr: arr,
        )
        arr = tf.cast(arr, tf.int32)

      if arr.shape.as_list() != [1, 1, 1, 1, 1]:
        data[name] = augmentation.apply_rotation(arr, rot_mtx, voxel_size)

  return data


def sample_coordinates(
    config: InputConfig,
    rng_seed: int | None = None,
) -> tf.data.Dataset:
  """Builds a dataset of coordinates at which to sample data.

  Args:
    config: Configuration of the input data.
    rng_seed: The random number generator seed to be used for determinism.

  Returns:
    Dataset with coordinates. Every item is guaranteed to have at least
    the following members:
      'coord': [1, 3] XYZ int64 array
      'volname': [1] string array
  """
  if config.sampling.bounding_boxes:
    boxes_cfg = []
    volume_names = []
    # Compile boxes and volumes strings.
    for vol_name, volume_boxes in config.sampling.bounding_boxes.items():
      volume_names.append(vol_name)
      boxes_cfg.append(volume_boxes)
    ds = inputs.sample_patch_coordinates(
        boxes_cfg, volume_names, rng_seed=rng_seed
    )
  elif config.sampling.bag_coords:
    raise NotImplementedError('bag file reading not supported yet.')
  elif config.sampling.arrayrecord_coords:

    def _make_source(pattern):
      return array_record.ArrayRecordDataSource(
          sorted(tf.io.gfile.glob(pattern))
      )

    if isinstance(config.sampling.arrayrecord_coords, str):
      sources = _make_source(config.sampling.arrayrecord_coords)
      weights = [1.0]
    else:
      sources, weights = [], []
      for pattern, weight in config.sampling.arrayrecord_coords.items():
        sources.append(_make_source(pattern))
        weights.append(weight)

    def _tf_load(idx, source):
      return tf.numpy_function(
          lambda x, src=source: src[x], [idx], [tf.string], stateful=False
      )[0]

    def _sample_indices(source, seed):
      rng = np.random.default_rng(seed)
      ds = tf.data.Dataset.from_tensor_slices(rng.permutation(len(source)))
      ds = ds.map(lambda x, src=source: _tf_load(x, source=src))
      return ds

    all_ds = [_sample_indices(s, rng_seed) for s in sources]

    weights = np.array(weights)
    weights = weights.astype(float) / weights.sum()
    ds = tf.data.Dataset.sample_from_datasets(all_ds, weights, seed=rng_seed)
    ds = ds.map(inputs.parse_tf_coords, deterministic=True)
  else:
    raise ValueError('No sampling scheme specified.')

  return ds


def load_and_augment_subvolumes(
    config: InputConfig,
    rng_seed: int | None = None,
    transform_locations: (
        Callable[[tf.data.Dataset, InputConfig], tf.data.Dataset] | None
    ) = None,
    ds: tf.data.Dataset | None = None,
) -> tf.data.Dataset:
  """Loads data and applies augmentations.

  Args:
    config: Configuration of the input data.
    rng_seed: The random number generator seed to be used for determinism.
    transform_locations: An optional funcion which will be called on loaded or
      generated coordinates. It allows to transform those coordinates before the
      data is loaded from the volumestore. The input dataset is guaranteed to
      contain the "coord". "coord" field indicates the center of the subvolume
      which will be loaded. Example use-case: Shifting the loaded subvolumes so
      that center (it can for example contain ground truth data) is shifted from
      the center of the patch.
    ds: Optional Dataset providing the coordinates to load. If specified,
      config.sampling is ignored.

  Returns:
    The loaded dataset.
  """
  if ds is None:
    ds = sample_coordinates(config, rng_seed)

  if transform_locations is not None:
    ds = transform_locations(ds, config)

  for vol in config.volumes.values():
    if vol.filter_shape is not None:
      ds = ds.filter(
          ft.partial(
              inputs.filter_oob,
              volinfo_map_string=get_path_str(vol.paths),
              patch_size=vol.filter_shape,
          )
      )

    ds = ds.filter(
        ft.partial(
            inputs.filter_oob,
            volinfo_map_string=get_path_str(vol.paths),
            patch_size=vol.load_shape,
            use_bboxes=False,
        )
    )

  # Actually load subvolumes.
  # pylint:disable=g-long-lambda
  ds = ds.interleave(
      lambda x: tf.data.Dataset.from_tensors(x).map(
          ft.partial(_load_data, config=config)
      ),
      cycle_length=config.read_parallelism,
      num_parallel_calls=config.read_parallelism,
      deterministic=(rng_seed is not None),
  )

  # Apply augmentations.
  ds = ds.interleave(
      lambda x: tf.data.Dataset.from_tensors(x).map(
          ft.partial(_apply_augmentations, config=config, voxel_size=voxel_size)
      ),
      cycle_length=config.augment_parallelism,
      num_parallel_calls=config.augment_parallelism,
      deterministic=(rng_seed is not None),
  )

  ds = _postprocess_augmented_data(ds, orig_config)
  return ds
