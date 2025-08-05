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
"""Helpers for inference jobs."""

import contextlib
import json
import threading
import time

import numpy as np
import skimage.exposure

from . import storage


# Names retained for compatibility with the MR interface.
# TODO(mjanusz): Drop this requirement or provide a wrapper class for the MR
# counter so that this is no longer necessary.
# pylint: disable=invalid-name
class StatCounter:
  """Stat counter with a MR counter interface."""

  def __init__(self, update, name, parent=None):
    """Initializes the counter.

    Args:
      update: callable taking no arguments; will be called when the counter is
        incremented
      name: name of the counter to use for streamz
      parent: optional StatCounter object to which to propagate any updates of
        the current counter
    """
    self._counter = 0
    self._update = update
    self._lock = threading.Lock()
    self._parent = parent

  def Increment(self):
    self.IncrementBy(1)

  def IncrementBy(self, x, export=True):
    """Increments the counter value by 'x'.

    Args:
      x: value to increment by
      export: whether to also increment the streamz counter
    """
    with self._lock:
      self._counter += int(x)
      self._update()

    if self._parent is not None:
      self._parent.IncrementBy(x)

  def Set(self, x, export=True):
    """Sets the counter value to 'x'.

    Args:
      x: value to set the counter to
    """
    x_old = self._counter
    x_diff = x - x_old
    self.IncrementBy(x_diff, export=export)

  def __repr__(self):
    return 'StatCounter(total=%g)' % (self.value)

  @property
  def value(self):
    return self._counter


# pylint: enable=invalid-name

MSEC_IN_SEC = 1000


class Counters:
  """Container for counters."""

  def __init__(self, parent=None):
    self._lock = threading.Lock()  # for self._counters
    self.reset()
    self.parent = parent

  def reset(self):
    with self._lock:
      self._counters = {}
    self._last_update = 0

  def __getitem__(self, name: str) -> StatCounter:
    return self.get(name)

  def get(self, name: str, **kwargs) -> StatCounter:
    """Retrieves the counter associated with the provided name.

    If the counter does not exist, it will be created.

    Args:
      name: counter name
      **kwargs: forwarded to _make_counter

    Returns:
      a counter corresponding to the specified name
    """
    with self._lock:
      if name not in self._counters:
        self._counters[name] = self._make_counter(name, **kwargs)
      return self._counters[name]

  def __iter__(self):
    return iter(self._counters.items())

  def _make_counter(self, name: str, **kwargs) -> StatCounter:
    del kwargs
    return StatCounter(self.update_status, name)

  def update_status(self):
    pass

  def get_sub_counters(self):
    return Counters(self)

  def dump(self, filename: str):
    with storage.atomic_file(filename, 'w') as fd:
      for name, counter in sorted(self._counters.items()):
        fd.write('%s: %d\n' % (name, counter.value))

  def dumps(self) -> str:
    state = {name: counter.value for name, counter in self._counters.items()}
    return json.dumps(state)

  def loads(self, encoded_state: str):
    state = json.loads(encoded_state)
    for name, value in state.items():
      # Do not set the exported counters. Otherwise after computing
      # the temporal differences in pcon large spikes will be shown.
      self[name].Set(value, export=False)


@contextlib.contextmanager
def timer_counter(
    counters: Counters, name: str, export=True, increment: int = 1
):
  """Creates a counter tracking time spent in the context.

  Args:
    counters: Counters object
    name: counter name
    export: whether to export counter via streamz
    increment: value by which to increment the underlying counter

  Yields:
    tuple of two counters, to track number of calls and time spent on them,
    respectively
  """
  assert isinstance(counters, Counters)
  counter = counters.get(name + '-calls', export=export)
  timer = counters.get(name + '-time-ms', export=export)
  start_time = time.time()
  try:
    yield timer, counter
  finally:
    counter.IncrementBy(increment)
    dt = (time.time() - start_time) * MSEC_IN_SEC
    timer.IncrementBy(dt)


class TimedIter:
  """Wraps an iterator with a timing counter."""

  def __init__(self, it, counters, counter_name):
    self.it = it
    self.counters = counters
    self.counter_name = counter_name

  def __iter__(self):
    return self

  def __next__(self):
    with timer_counter(self.counters, self.counter_name):
      ret = next(self.it)
    return ret

  def next(self):
    return self.__next__()


def match_histogram(image, lut, mask=None):
  """Changes the intensity distribution of a 3d image.

  The distrubution is changed so that it matches a reference
  distribution, for which a lookup table was produced by
  `compute_histogram_lut`.

  Args:
    image: (z, y, x) ndarray with the source image
    lut: lookup table from `compute_histogram_lut`
    mask: optional Boolean mask defining areas that are NOT to be considered for
      CDF calculation after applying CLAHE

  Returns:
    None; `image` is modified in place
  """
  for z in range(image.shape[0]):
    clahe_slice = skimage.exposure.equalize_adapthist(image[z, ...])
    clahe_slice = (clahe_slice * 255).astype(np.uint8)

    valid_slice = clahe_slice
    if mask is not None:
      valid_slice = valid_slice[np.logical_not(mask[z, ...])]

    if valid_slice.size == 0:
      continue

    cdf, bins = skimage.exposure.cumulative_distribution(valid_slice)
    cdf = np.array(cdf.tolist() + [1.0])
    bins = np.array(bins.tolist() + [255])
    image[z, ...] = lut[
        (cdf[np.searchsorted(bins, clahe_slice)] * 255).astype(np.uint8)
    ]


def compute_histogram_lut(image):
  """Computes the inverted CDF of image intensity.

  Args:
    image: 2d numpy array containing the image

  Returns:
    a 256-element numpy array representing a lookup table `lut`,
    such that lut[uniform_image] will transform `uniform_image` with
    a uniform intensity distribution to have an intensity distribution
    matching `image`.
  """
  cdf, bins = skimage.exposure.cumulative_distribution(image)
  lut = np.zeros(256, dtype=np.uint8)
  for i in range(0, 256):
    lut[i] = bins[np.searchsorted(cdf, i / 255.0)]

  return lut
