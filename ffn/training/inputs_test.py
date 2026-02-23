# Copyright 2026 Google Inc.
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

import functools as ft
import os


from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.common import tuples
from connectomics.volume import metadata
from ffn.training import inputs
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class _FilterOobTestBase:

  _volinfo_map_string: str

  def test_filter_oob_in_bounds(self):
    coord = tf.constant([[50, 50, 50]], dtype=tf.int64)
    volname = tf.constant(['testvol'], dtype=tf.string)
    item = {'coord': coord, 'volname': volname}
    result = inputs.filter_oob(
        item, self._volinfo_map_string, patch_size=[10, 10, 10]
    )
    with tf.Session() as sess:
      self.assertTrue(sess.run(result))

  def test_filter_oob_out_of_bounds(self):
    coord = tf.constant([[0, 0, 0]], dtype=tf.int64)
    volname = tf.constant(['testvol'], dtype=tf.string)
    item = {'coord': coord, 'volname': volname}
    result = inputs.filter_oob(
        item, self._volinfo_map_string, patch_size=[10, 10, 10]
    )
    with tf.Session() as sess:
      self.assertFalse(sess.run(result))

  def test_filter_oob_in_dataset_filter(self):
    ds = tf.data.Dataset.from_tensors({
        'coord': tf.constant([[50, 50, 50]], dtype=tf.int64),
        'volname': tf.constant(['testvol'], dtype=tf.string),
    })
    ds = ds.filter(
        ft.partial(
            inputs.filter_oob,
            volinfo_map_string=self._volinfo_map_string,
            patch_size=[10, 10, 10],
        )
    )
    iterator = tf.data.make_one_shot_iterator(ds)
    item = iterator.get_next()
    with tf.Session() as sess:
      result = sess.run(item)
      np.testing.assert_array_equal(result['coord'], [[50, 50, 50]])


class FilterOobMetadataJsonTest(_FilterOobTestBase, absltest.TestCase):

  def setUp(self):
    super().setUp()
    tf.reset_default_graph()

    self._tmpdir = self.create_tempdir().full_path
    meta = metadata.VolumeMetadata(
        path='none',
        volume_size=tuples.XYZ(100, 100, 100),
        pixel_size=tuples.XYZ(8, 8, 30),
        bounding_boxes=[
            bounding_box.BoundingBox(start=(0, 0, 0), size=(100, 100, 100))
        ],
    )
    self._metadata_path = os.path.join(self._tmpdir, 'metadata.json')
    with open(self._metadata_path, 'w') as f:
      f.write(meta.to_json())

    self._volinfo_map_string = f'testvol:{self._metadata_path}'


if __name__ == '__main__':
  absltest.main()
