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

from absl.testing import absltest
from ffn.training import tracker
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class TrackerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tf.reset_default_graph()

  def test_tracker_rendering(self):
    eval_shape = [32, 32, 32]
    shifts = [(0, 0, 0)]
    eval_tracker = tracker.EvalTracker(eval_shape, shifts)
    eval_tracker.sess = tf.Session()
    eval_tracker.sess.run(tf.global_variables_initializer())

    labels = np.zeros([1] + eval_shape + [1], dtype=np.float32)
    predicted = np.zeros([1] + eval_shape + [1], dtype=np.float32)
    weights = np.zeros([1] + eval_shape + [1], dtype=np.float32)

    # Check rendering with volume name
    eval_tracker.add_patch(
        labels,
        predicted,
        weights,
        coord=np.array([0, 0, 0]),
        volume_name='test_volume',
    )

    eval_tracker.to_tf()
    summaries = eval_tracker.get_summaries()

    # Verify that we got image summaries
    image_summaries = [s for s in summaries if s.HasField('image')]
    self.assertNotEmpty(image_summaries)

    # Check specifically for the tags we expect
    tags = [s.tag for s in image_summaries]
    self.assertIn('final_xy/0', tags)
    self.assertIn('final_xz/0', tags)
    self.assertIn('final_yz/0', tags)

  def test_tracker_rendering_no_volume_name(self):
    eval_shape = [32, 32, 32]
    shifts = [(0, 0, 0)]
    eval_tracker = tracker.EvalTracker(eval_shape, shifts)
    eval_tracker.sess = tf.Session()
    eval_tracker.sess.run(tf.global_variables_initializer())

    labels = np.zeros([1] + eval_shape + [1], dtype=np.float32)
    predicted = np.zeros([1] + eval_shape + [1], dtype=np.float32)
    weights = np.zeros([1] + eval_shape + [1], dtype=np.float32)

    # Check rendering without volume name
    eval_tracker.add_patch(
        labels, predicted, weights, coord=np.array([0, 0, 0])
    )

    eval_tracker.to_tf()
    summaries = eval_tracker.get_summaries()

    # Verify that we got image summaries
    image_summaries = [s for s in summaries if s.HasField('image')]
    self.assertNotEmpty(image_summaries)


if __name__ == '__main__':
  absltest.main()
