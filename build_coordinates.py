#!/usr/bin/env python

"""Builds a TFRecord file of coordinates for training.

Use ./compute_partitions.py to generate data for --partition_volumes.
Note that the volume names you provide in --partition_volumes will
have to match the volume labels you pass to the training script.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from absl import app
from absl import flags
from absl import logging

import h5py
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_list('partition_volumes', None,
                  'Partition volumes as '
                  '<volume_name>:<volume_path>:<dataset>, where volume_path '
                  'points to a HDF5 volume, and <volume_name> is an arbitrary '
                  'label that will have to also be used during training.')
flags.DEFINE_string('coordinate_output', None,
                    'Path to a TF Record file in which to save the '
                    'coordinates.')
flags.DEFINE_list('margin', None, '(z, y, x) tuple specifying the '
                  'number of voxels adjacent to the border of the volume to '
                  'exclude from sampling. This should normally be set to the '
                  'radius of the FFN training FoV (i.e. network FoV radius '
                  '+ deltas.')


IGNORE_PARTITION = 255


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
  del argv  # Unused.

  totals = defaultdict(int)  # partition -> voxel count
  indices = defaultdict(list)  # partition -> [(vol_id, 1d index)]

  vol_labels = []
  vol_shapes = []
  mz, my, mx = [int(x) for x in FLAGS.margin]

  for i, partvol in enumerate(FLAGS.partition_volumes):
    name, path, dataset = partvol.split(':')
    with h5py.File(path, 'r') as f:
      partitions = f[dataset][mz:-mz, my:-my, mx:-mx]
      vol_shapes.append(partitions.shape)
      vol_labels.append(name)

      uniques, counts = np.unique(partitions, return_counts=True)
      for val, cnt in zip(uniques, counts):
        if val == IGNORE_PARTITION:
          continue

        totals[val] += cnt
        indices[val].extend(
            [(i, flat_index) for flat_index in
             np.flatnonzero(partitions == val)])

  logging.info('Partition counts:')
  for k, v in totals.items():
    logging.info(' %d: %d', k, v)

  logging.info('Resampling and shuffling coordinates.')

  max_count = max(totals.values())
  indices = np.concatenate(
      [np.resize(np.random.permutation(v), (max_count, 2)) for
       v in indices.values()], axis=0)
  np.random.shuffle(indices)

  logging.info('Saving coordinates.')
  record_options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  with tf.python_io.TFRecordWriter(FLAGS.coordinate_output,
                                   options=record_options) as writer:
    for i, coord_idx in indices:
      z, y, x = np.unravel_index(coord_idx, vol_shapes[i])

      coord = tf.train.Example(features=tf.train.Features(feature=dict(
          center=_int64_feature([mx + x, my + y, mz + z]),
          label_volume_name=_bytes_feature(vol_labels[i].encode('utf-8'))
      )))
      writer.write(coord.SerializeToString())


if __name__ == '__main__':
  flags.mark_flag_as_required('margin')
  flags.mark_flag_as_required('coordinate_output')
  flags.mark_flag_as_required('partition_volumes')

  app.run(main)
