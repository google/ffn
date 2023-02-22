# coding=utf-8
# Copyright 2020-2023 Google Inc.
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
from ffn.utils import decision_point
import numpy as np


class DecisionPointTest(absltest.TestCase):

  def test_find_decision_point(self):
    # 2 segments
    seg = np.zeros((100, 80, 60), dtype=np.uint64)
    seg[:40, :, :] = 1
    seg[60:, :, :] = 2
    points = decision_point.find_decision_points(seg, (1, 1, 1))
    self.assertIn((1, 2), points)
    self.assertLen(points, 1)

    dist, point = points[(1, 2)]
    self.assertEqual(dist, 10)
    self.assertEqual(point.tolist(), [29, 39, 49])

    # 3 segments
    seg = np.zeros((1, 100, 100), dtype=np.uint64)
    seg[0, :20, :20] = 1
    seg[0, :20:, -20:] = 2
    seg[0, -20:, 40:60] = 3
    points = decision_point.find_decision_points(seg, (1, 1, 1))
    self.assertIn((1, 2), points)
    self.assertIn((1, 3), points)
    self.assertIn((2, 3), points)
    self.assertLen(points, 3)
    self.assertEqual(points[(1, 2)][1].tolist(), [49, 9, 0])
    self.assertEqual(points[(1, 3)][1].tolist(), [29, 49, 0])
    self.assertEqual(points[(2, 3)][1].tolist(), [69, 49, 0])

    # Same as above, but distance restricted so that only the 1-2
    # decision point is found.
    points = decision_point.find_decision_points(
        seg, (1, 1, 1), max_distance=30.0)
    self.assertIn((1, 2), points)
    self.assertLen(points, 1)


if __name__ == '__main__':
  absltest.main()
