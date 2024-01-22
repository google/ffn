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

import numpy as np
import tensorflow.compat.v1 as tf


class FractionTracker:
  """Helper for tracking fractions."""

  def __init__(self, name: str = 'fraction'):
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


class DistributionTracker:
  """Helper for tracking distributions."""

  def __init__(self, num_classes: int, name: str = 'distribution'):
    self.num_classes = num_classes
    self.var = tf.get_variable(
        name, [num_classes],
        tf.int64,
        tf.constant_initializer([0] * num_classes),
        trainable=False)

  def record_class(self, class_id, count=1):
    return self.var.assign_add(
        tf.one_hot(class_id, self.num_classes, dtype=tf.int64) * count)

  def record_classes(self, labels):
    delta = tf.math.bincount(
        labels,
        minlength=self.num_classes,
        maxlength=self.num_classes,
        dtype=tf.int64)
    return self.var.assign_add(delta)

  def get_rates(self, reset=True):
    """Queries the class frequencies.

    Args:
      reset: whether to reset the class counters to 0 after query

    Returns:
      TF op for class frequencies
    """
    total = tf.reduce_sum(self.var)
    rates = tf.cast(self.var, tf.float32) / tf.maximum(
        tf.constant(1, dtype=tf.float32), tf.cast(total, tf.float32))
    if not reset:
      return rates

    with tf.control_dependencies([rates]):
      update_var = self.var.assign_add(-self.var)
    with tf.control_dependencies([update_var]):
      return tf.identity(rates)


def get_and_reset_value(var):
  readout = var + 0
  with tf.control_dependencies([readout]):
    update_var = var.assign_add(-readout)
  with tf.control_dependencies([update_var]):
    return tf.identity(readout)


class TFSyncVariable:
  """A local variable which can be periodically synchronized to a TF one."""

  def __init__(self, name, shape, dtype):
    self._value = np.zeros(shape, dtype=dtype.as_numpy_dtype)
    self._tf_var = tf.get_variable(
        name,
        shape,
        dtype,
        tf.constant_initializer(self.value),
        trainable=False)
    self._update_placeholder = tf.placeholder(
        dtype, shape, name='plc_%s' % name)
    self._to_tf = self._tf_var.assign_add(self._update_placeholder)
    self._from_tf = get_and_reset_value(self._tf_var)
    self.tf_value = None

  @property
  def from_tf(self):
    return self._from_tf

  @property
  def value(self):
    return self._value

  def to_tf(self, ops, feed_dict):
    ops.append(self._to_tf)
    feed_dict[self._update_placeholder] = self._value
    self._value = np.zeros_like(self._value)

  def reset(self):
    self._value = np.zeros_like(self._value)
