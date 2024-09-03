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

"""Pure numpy adaptation of the FFN tracker.

This makes the training script independent from TF, other than for the
input pipeline.
"""

from ffn.training import tracker
import numpy as np


class Variable:
  """Variable keeping its value as a numpy array."""

  def __init__(self, shape, dtype):
    self._value = np.zeros(shape, dtype=dtype.as_numpy_dtype)

  @property
  def tf_value(self):
    return self._value

  @property
  def from_tf(self):
    return None

  @property
  def value(self):
    return self._value

  def to_tf(self, ops, feed_dict):
    pass

  def reset(self):
    self._value[:] = 0.


class EvalTracker(tracker.EvalTracker):
  """Eval tracker using numpy variables."""

  def _add_tf_var(self, name, shape, dtype):
    v = Variable(shape, dtype)
    setattr(self, name, v)
    self._tf_vars.append(v)
    return v

  def to_tf(self):
    pass

  def from_tf(self):
    pass
