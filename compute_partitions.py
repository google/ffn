#!/usr/bin/env python

r"""Computes the partition map for a segmentation.

For every labeled voxel of the input volume, computes the fraction of identically
labeled voxels within a neighborhood of radius `lom_radius`, and then quantizes
that number according to `thresholds`.

Sample invocation:
  python compute_partitions.py \
      --input_volume third_party/neuroproof_examples/training_sample2/groundtruth.h5:stack \
      --output_volume af.h5:af \
      --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
      --lom_radius 16,16,16 \
      --min_size 10000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from ffn.inference import segmentation
from ffn.inference import storage
from ffn.utils import bounding_box

import h5py
import numpy as np
from scipy.ndimage import filters

FLAGS = flags.FLAGS

flags.DEFINE_string('input_volume', None,
                    'Segmentation volume as <volume_path>:<dataset>, where'
                    'volume_path points to a HDF5 volume.')
flags.DEFINE_string('output_volume', None,
                    'Volume in which to save the partition map, as '
                    '<volume_path>:<dataset>.')
flags.DEFINE_list('thresholds', None,
                  'List of activation voxel fractions used for partitioning.')
flags.DEFINE_list('lom_radius', None,
                  'Local Object Mask (LOM) radii as (x, y, z).')
flags.DEFINE_list('id_whitelist', None,
                  'Whitelist of object IDs for which to compute the partition '
                  'numbers.')
flags.DEFINE_list('exclusion_regions', None,
                  'List of (x, y, z, r) tuples specifying spherical regions to '
                  'mark as excluded (i.e. set the output value to 255).')
flags.DEFINE_string('mask_configs', None,
                    'MaskConfigs proto in text foramt. Any locations where at '
                    'least one voxel of the LOM is masked will be marked as '
                    'excluded.')
flags.DEFINE_integer('min_size', 10000,
                     'Minimum number of voxels for a segment to be considered for '
                     'partitioning.')


def _summed_volume_table(val):
  """Computes a summed volume table of 'val'."""
  val = val.astype(np.int32)
  svt = val.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
  return np.pad(svt, [[1, 0], [1, 0], [1, 0]], mode='constant')


def _query_summed_volume(svt, diam):
  """Queries a summed volume table.

  Operates in 'VALID' mode, i.e. only computes the sums for voxels where the
  full diam // 2 context is available.

  Args:
    svt: summed volume table (see _summed_volume_table)
    diam: diameter (z, y, x tuple) of the area within which to compute sums

  Returns:
    sum of all values within a diam // 2 radius (under L1 metric) of every voxel
    in the array from which 'svt' was built.
  """
  return (
      svt[diam[0]:, diam[1]:, diam[2]:] - svt[diam[0]:, diam[1]:, :-diam[2]] -
      svt[diam[0]:, :-diam[1], diam[2]:] - svt[:-diam[0], diam[1]:, diam[2]:] +
      svt[:-diam[0], :-diam[1], diam[2]:] + svt[:-diam[0], diam[1]:, :-diam[2]]
      + svt[diam[0]:, :-diam[1], :-diam[2]] -
      svt[:-diam[0], :-diam[1], :-diam[2]])


def load_mask(mask_configs, box, lom_diam_zyx):
  if mask_configs is None:
    return None

  mask = storage.build_mask(mask_configs.masks, box.start[::-1],
                            box.size[::-1])
  svt = _summed_volume_table(mask)
  mask = _query_summed_volume(svt, lom_diam_zyx) >= 1
  return mask


def compute_partitions(seg_array,
                       thresholds,
                       lom_radius,
                       id_whitelist=None,
                       exclusion_regions=None,
                       mask_configs=None,
                       min_size=10000):
  """Computes quantized fractions of active voxels in a local object mask.

  Args:
    thresholds: list of activation voxel fractions to use for partitioning.
    lom_radius: LOM radii as [x, y, z]
    id_whitelist: (optional) whitelist of object IDs for which to compute the
        partition numbers
    exclusion_regions: (optional) list of x, y, z, r tuples specifying regions
        to mark as excluded (with 255). The regions are spherical, with
        (x, y, z) definining the center of the sphere and 'r' specifying its
        radius. All values are in voxels.
    mask_configs: (optional) MaskConfigs proto; any locations where at least
        one voxel of the LOM is masked will be marked as excluded (255).

  Returns:
    tuple of:
      corner of output subvolume as (x, y, z)
      uint8 ndarray of active fraction voxels
  """
  seg_array = segmentation.clear_dust(seg_array, min_size=min_size)
  assert seg_array.ndim == 3

  lom_radius = np.array(lom_radius)
  lom_radius_zyx = lom_radius[::-1]
  lom_diam_zyx = 2 * lom_radius_zyx + 1

  def _sel(i):
    if i == 0:
      return slice(None)
    else:
      return slice(i, -i)

  valid_sel = [_sel(x) for x in lom_radius_zyx]
  output = np.zeros(seg_array[valid_sel].shape, dtype=np.uint8)
  corner = lom_radius

  if exclusion_regions is not None:
    sz, sy, sx = output.shape
    hz, hy, hx = np.mgrid[:sz, :sy, :sx]

    hz += corner[2]
    hy += corner[1]
    hx += corner[0]

    for x, y, z, r in exclusion_regions:
      mask = (hx - x)**2 + (hy - y)**2 + (hz - z)**2 <= r**2
      output[mask] = 255

  labels = set(np.unique(seg_array))
  logging.info('Labels to process: %d', len(labels))

  if id_whitelist is not None:
    labels &= set(id_whitelist)

  mask = load_mask(mask_configs,
                   bounding_box.BoundingBox(
                       start=(0, 0, 0), size=seg_array.shape[::-1]),
                   lom_diam_zyx)
  if mask is not None:
    output[mask] = 255

  fov_volume = np.prod(lom_diam_zyx)
  for l in labels:
    # Don't create a mask for the background component.
    if l == 0:
      continue

    object_mask = (seg_array == l)

    svt = _summed_volume_table(object_mask)
    active_fraction = _query_summed_volume(svt, lom_diam_zyx) / fov_volume
    assert active_fraction.shape == output.shape

    # Drop context that is only necessary for computing the active fraction
    # (i.e. one LOM radius in every direction).
    object_mask = object_mask[valid_sel]

    # TODO(mjanusz): Use np.digitize here.
    for i, th in enumerate(thresholds):
      output[object_mask & (active_fraction < th) & (output == 0)] = i + 1

    output[object_mask & (active_fraction >= thresholds[-1]) &
           (output == 0)] = len(thresholds) + 1

    logging.info('Done processing %d', l)

  logging.info('Nonzero values: %d', np.sum(output > 0))

  return corner, output


def adjust_bboxes(bboxes, lom_radius):
  ret = []

  for bbox in bboxes:
    bbox = bbox.adjusted_by(start=lom_radius, end=-lom_radius)
    if np.all(bbox.size > 0):
      ret.append(bbox)

  return ret


def main(argv):
  del argv  # Unused.
  path, dataset = FLAGS.input_volume.split(':')
  with h5py.File(path) as f:
    segmentation = f[dataset]
    bboxes = []
    for name, v in segmentation.attrs.items():
      if name.startswith('bounding_boxes'):
        for bbox in v:
          bboxes.append(bounding_box.BoundingBox(bbox[0], bbox[1]))

    if not bboxes:
      bboxes.append(
          bounding_box.BoundingBox(
              start=(0, 0, 0), size=segmentation.shape[::-1]))

    shape = segmentation.shape
    lom_radius = [int(x) for x in FLAGS.lom_radius]
    corner, partitions = compute_partitions(
        segmentation[...], [float(x) for x in FLAGS.thresholds], lom_radius,
        FLAGS.id_whitelist, FLAGS.exclusion_regions, FLAGS.mask_configs,
        FLAGS.min_size)

  bboxes = adjust_bboxes(bboxes, np.array(lom_radius))

  path, dataset = FLAGS.output_volume.split(':')
  with h5py.File(path, 'w') as f:
    ds = f.create_dataset(dataset, shape=shape, dtype=np.uint8, fillvalue=255,
                          chunks=True, compression='gzip')
    s = partitions.shape
    ds[corner[2]:corner[2] + s[0],
       corner[1]:corner[1] + s[1],
       corner[0]:corner[0] + s[2]] = partitions
    ds.attrs['bounding_boxes'] = [(b.start, b.size) for b in bboxes]
    ds.attrs['partition_counts'] = np.array(np.unique(partitions,
                                                      return_counts=True))


if __name__ == '__main__':
  flags.mark_flag_as_required('input_volume')
  flags.mark_flag_as_required('output_volume')
  flags.mark_flag_as_required('thresholds')
  flags.mark_flag_as_required('lom_radius')
  app.run(main)
