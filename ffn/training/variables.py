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
"""Customized variables for tracking ratios, rates, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.google as tf


class FractionTracker(object):
  """Helper for tracking fractions."""

  def __init__(self, name='fraction'):
    # Values are: total, hits.
    self.var = tf.get_variable(name, [2], tf.int64,
                               tf.constant_initializer([0, 0]), trainable=False)

  def record_miss(self):
    return self.var.assign_add([1, 0])

  def record_hit(self):
    return self.var.assign_add([1, 1])

  def get_hit_rate(self):
    total = self.var[0]
    hits = self.var[1]
    hit_rate = (tf.cast(hits, tf.float32) /
                tf.maximum(tf.constant(1, dtype=tf.float32),
                           tf.cast(total, tf.float32)))

    with tf.control_dependencies([hit_rate]):
      update_var = self.var.assign_add([-total, -hits])
    with tf.control_dependencies([update_var]):
      return tf.identity(hit_rate)
