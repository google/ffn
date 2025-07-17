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
"""Policies for choosing starting points for FFNs.

Seed policies are iterable objects yielding (z, y, x) tuples identifying
points at which the FFN will attempt to create a segment.
"""

import itertools
import threading
from typing import Any, Sequence
import weakref

from absl import logging
import edt
import numpy as np
from scipy import ndimage
import skimage
import skimage.feature
import skimage.morphology

from . import storage


class BaseSeedPolicy:
  """Base class for seed policies."""

  def __init__(self, canvas, **kwargs):
    """Initializes the policy.

    Args:
      canvas: inference Canvas object; simple policies use this to access basic
        geometry information such as the shape of the subvolume; more complex
        policies can access the raw image data, etc.
      **kwargs: other keyword arguments
    """
    logging.info('Deleting unused BaseSeedPolicy kwargs: %s', kwargs)
    del kwargs

    # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
    self.canvas = weakref.proxy(canvas)
    self.coords: np.ndarray = None  # shape: [N, 3] (zyx order in the last dim)
    self.idx = 0

  def init_coords(self):
    raise NotImplementedError()

  def __iter__(self):
    return self

  def __next__(self):
    """Returns the next seed point as (z, y, x).

    Does initial filtering of seed points to exclude locations that are
    too close to the image border.

    Returns:
      (z, y, x) tuples.

    Raises:
      StopIteration when the seeds are exhausted.
    """
    if self.coords is None:
      self.init_coords()

      if self.coords is None:
        raise StopIteration()

      if self.coords.size:
        margin = np.array(self.canvas.margin)[np.newaxis, ...]
        # Do early filtering of clearly invalid locations (too close to image
        # borders) as late filtering might be expensive.
        self.coords = self.coords[np.all(
            (self.coords - margin >= 0) &
            (self.coords + margin < self.canvas.shape),
            axis=1), :]

    while self.idx < self.coords.shape[0]:
      curr = self.coords[self.idx, :]
      self.idx += 1
      return tuple(curr)  # z, y, x

    raise StopIteration()

  def next(self):
    return self.__next__()

  def get_state(self, previous=False):
    """Returns a pickleable state for this seeding policy.

    Args:
      previous: if True, indicates that a state for the already consumed seed,
        and so an in-progress segment, is being requested
    """
    if previous:
      return self.coords, max(0, self.idx - 1)
    else:
      return self.coords, self.idx

  def set_state(self, state):
    self.coords, self.idx = state

  # This is done as an optimization to avoid generating seeds that are
  # immediately going to be rejected. Keep this in sync with
  # restrictor.is_valid_{seed|pos} and canvas.is_valid_pos.
  def get_exclusion_mask(self):
    """Returns a mask of voxels that are considered invalid for seeds."""
    mask = self.canvas.segmentation > 0
    if self.canvas.restrictor is not None:
      # This mask prevents any FOV movement into the masked area,
      # including the initial step (seed).
      if self.canvas.restrictor.mask is not None:
        mask |= self.canvas.restrictor.mask

      if self.canvas.restrictor.seed_mask is not None:
        mask |= self.canvas.restrictor.seed_mask

    return mask


def _find_peaks(distances, **kwargs):
  # Use a specifc seed for the noise so that results are reproducible
  # regardless of what happens before the policy is called.
  rng = np.random.RandomState(seed=42)
  idxs = skimage.feature.peak_local_max(
      distances + rng.rand(*distances.shape) * 1e-4, **kwargs)
  return idxs


class PolicyPeaks(BaseSeedPolicy):
  """Attempts to find points away from edges in the image.

  Runs a 3d Sobel filter to detect edges in the raw data, followed
  by a distance transform and peak finding to identify seed points.
  """

  # Limit the number of concurrent threads computing the local maxima. This
  # operation causes temporary increase in memory usage.
  _sem = threading.Semaphore(4)

  def init_coords(self):
    logging.info('peaks: starting')

    # Edge detection.
    edges = ndimage.generic_gradient_magnitude(
        self.canvas.image.astype(np.float32), ndimage.sobel)

    # Adaptive thresholding.
    sigma = 49.0 / 6.0
    thresh_image = np.zeros(edges.shape, dtype=np.float32)
    ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
    filt_edges = edges > thresh_image

    del edges, thresh_image

    mask = self.get_exclusion_mask()

    # This prevents a border effect where the large amount of masked area
    # screws up the distance transform below.
    if self.canvas.restrictor is not None:
      if self.canvas.restrictor.mask is not None:
        filt_edges[self.canvas.restrictor.mask] = 1
      if self.canvas.restrictor.seed_mask is not None:
        filt_edges[self.canvas.restrictor.seed_mask] = 1

    if np.all(filt_edges == 1):
      return

    with PolicyPeaks._sem:
      logging.info('peaks: filtering done')
      dt = edt.edt(
          1 - filt_edges,
          anisotropy=self.canvas.voxel_size_zyx).astype(np.float32)
      logging.info('peaks: edt done')

      dt[mask] = -1
      dt[~np.isfinite(dt)] = -1

      idxs = _find_peaks(dt, min_distance=3, threshold_abs=0, threshold_rel=0)

      # After skimage upgrade to 0.13.0 peak_local_max returns peaks in
      # descending order, versus ascending order previously.  Sort ascending to
      # maintain historic behavior.
      idxs = np.array(sorted((z, y, x) for z, y, x in idxs))

      logging.info('peaks: found %d local maxima', idxs.shape[0])
      self.coords = idxs


class PolicyPeaks2d(BaseSeedPolicy):
  """Attempts to find points away from edges at each 2d slice of image.

  Runs a 2d Sobel filter to detect edges in each 2d slice of
  raw data (specified by z index), followed by 2d distance transform
  and peak finding to identify seed points.
  """

  def __init__(self, canvas, min_distance=7, threshold_abs=2.5,
               sort_cmp='ascending', **kwargs):
    """Initialize settings.

    For compatibility with original version, min_distance=3, threshold_abs=0,
    sort=False.

    Args:
      canvas: inference Canvas object.
      min_distance: forwarded to peak_local_max.
      threshold_abs: forwarded to peak_local_max.
      sort_cmp: the cmp function to use for sorting seed coordinates.
      **kwargs: forwarded to base.
    """
    super().__init__(canvas, **kwargs)
    self.min_distance = min_distance
    self.threshold_abs = threshold_abs
    self.sort_reverse = sort_cmp.strip().lower().startswith('de')

  def init_coords(self):
    logging.info('2d peaks: starting')

    # Loop over 2d slices.
    for z in range(self.canvas.image.shape[0]):
      image_2d = (self.canvas.image[z, :, :]).astype(np.float32)

      # Edge detection.
      edges = ndimage.generic_gradient_magnitude(image_2d, ndimage.sobel)

      # Adaptive thresholding.
      sigma = 49.0 / 6.0
      thresh_image = np.zeros(edges.shape, dtype=np.float32)
      ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
      filt_edges = edges > thresh_image

      del edges, thresh_image

      # Prevent border effect
      if (self.canvas.restrictor is not None and
          self.canvas.restrictor.mask is not None):
        filt_edges[self.canvas.restrictor.mask[z, :, :]] = 1

      # Distance transform
      dt = edt.edt(1 - filt_edges).astype(np.float32)

      idxs = _find_peaks(
          dt,
          min_distance=self.min_distance,
          threshold_abs=self.threshold_abs,
          threshold_rel=0)

      # TODO(phli): Not sure why this was using image_2d instead of dt for
      # peaks.  Fix and wire back in.
      # if self.sort:
      #   # Visit the seeds in order of greatest to least distance from edge.
      #   peakvals = image_2d[tuple(idxs.transpose())]
      #   idxs = idxs[np.argsort(-peakvals), :]

      zs = np.full((idxs.shape[0], 1), z, dtype=np.int64)
      idxs = np.concatenate((zs, idxs), axis=1)

      # Update self.coords with indices found at this z index
      logging.info('2d peaks: found %d local maxima at z index %d',
                   idxs.shape[0], z)
      self.coords = np.concatenate((self.coords, idxs)) if z != 0 else idxs

    self.coords = np.array(
        sorted([(z, y, x) for z, y, x in self.coords],
               reverse=self.sort_reverse))

    logging.info('2d peaks: found %d total local maxima', self.coords.shape[0])


class PolicyFillEmptySpace(BaseSeedPolicy):
  """Selects points in unsegmented parts of the image.

  Seed points are created in a local maxima of the distance transform
  of the segmenetation.

  Use this policy to try to fill fragments of the smallest branches in an
  otherwise complete segmentation in datasets with little to no ECS.
  """

  def init_coords(self):
    logging.info('fill_empty: starting')

    dt = edt.edt(self.canvas.segmentation == 0).astype(np.float32)

    # Set absolute threshold to <1 to avoid generating seeds in areas that are
    # already segmented, where dt >= 1. This also helps to avoid slow execution
    # caused by https://github.com/scikit-image/scikit-image/issues/5161
    idxs = _find_peaks(dt, min_distance=2, threshold_abs=0.5, threshold_rel=0)

    logging.info('fill_empty: found %d local maxima', idxs.shape[0])
    self.coords = np.array(sorted((z, y, x) for z, y, x in idxs))


class PolicyMax(BaseSeedPolicy):
  """All points in the image, in descending order of intensity."""

  def init_coords(self):
    idxs = np.mgrid[[slice(0, x) for x in self.canvas.image.shape]]
    sort_idx = np.argsort(self.canvas.image.flat)[::-1]
    self.coords = np.array(list(zip(*[idx.flat[sort_idx] for idx in idxs])))


class PolicyMaxPeaks(BaseSeedPolicy):
  """Local peaks of intensity."""

  def init_coords(self):
    img = self.canvas.image.astype(np.float32).copy()
    mask = self.get_exclusion_mask()
    img[mask] = 0
    idxs = _find_peaks(img, min_distance=3, threshold_abs=0, threshold_rel=0)
    self.coords = np.array(sorted((z, y, x) for z, y, x in idxs))


class PolicyImagePeaks3D2D(BaseSeedPolicy):
  """3d image peaks followed by 2d image peaks."""

  def __init__(self, canvas, min_distance_2d=2, min_distance_3d=4, **kwargs):
    super().__init__(canvas, **kwargs)
    self._min_distance_2d = min_distance_2d
    self._min_distance_3d = min_distance_3d

  def init_coords(self):
    img = self.canvas.image
    coords3d = []
    if self._min_distance_3d >= 0:
      coords3d = np.array(
          skimage.feature.peak_local_max(
              img, min_distance=self._min_distance_3d)).tolist()

    coords2d = []
    if self._min_distance_2d >= 0:
      for z in range(img.shape[0]):
        coordinates = skimage.feature.peak_local_max(
            img[z, ...], min_distance=self._min_distance_2d)
        for y, x in coordinates:
          coords2d.append((z, y, x))

    self.coords = np.array(coords3d + coords2d)


class PolicyImagePeaks2DDisk(BaseSeedPolicy):
  """2d image peaks with disk footprint."""

  def __init__(self, canvas, min_distance_2d=3, threshold_rel=0.5,
               disk_radius=1, **kwargs):
    super().__init__(canvas, **kwargs)
    self._min_distance_2d = min_distance_2d
    self._threshold_rel = threshold_rel
    self._disk_radius = disk_radius

  def init_coords(self):
    img = self.canvas.image

    coords2d_disk = []
    footprint = skimage.morphology.disk(radius=self._disk_radius)
    for z in range(img.shape[0]):
      coordinates = skimage.feature.peak_local_max(
          img[z, ...],
          min_distance=self._min_distance_2d,
          p_norm=2,
          threshold_rel=self._threshold_rel,
          exclude_border=True,
          footprint=footprint)
      for y, x in coordinates:
        coords2d_disk.append((z, y, x))

    self.coords = np.array(coords2d_disk)


class PolicyGrid3d(BaseSeedPolicy):
  """Points distributed on a uniform 3d grid."""

  def __init__(self,
               canvas,
               step=16,
               offsets=(0, 8, 4, 12, 2, 10, 14),
               **kwargs):
    super().__init__(canvas, **kwargs)
    self.step = step
    self.offsets = offsets

  def init_coords(self):
    coords = []
    for offset in self.offsets:
      for z in range(offset, self.canvas.image.shape[0], self.step):
        for y in range(offset, self.canvas.image.shape[1], self.step):
          for x in range(offset, self.canvas.image.shape[2], self.step):
            coords.append((z, y, x))
    self.coords = np.array(coords)


class PolicyGrid2d(BaseSeedPolicy):
  """Points distributed on a uniform 2d grid."""

  def __init__(self,
               canvas,
               step=16,
               offsets=(0, 8, 4, 12, 2, 6, 10, 14),
               **kwargs):
    super().__init__(canvas, **kwargs)
    self.step = step
    self.offsets = offsets

  def init_coords(self):
    coords = []
    for offset in self.offsets:
      for z in range(self.canvas.image.shape[0]):
        for y in range(offset, self.canvas.image.shape[1], self.step):
          for x in range(offset, self.canvas.image.shape[2], self.step):
            coords.append((z, y, x))
    self.coords = np.array(coords)


class PolicyInvertOrigins(BaseSeedPolicy):
  """Reverse order of the seed locations used in a previous segmentation run."""

  def __init__(self, canvas, corner=None, segmentation_dir=None, **kwargs):
    super().__init__(canvas, **kwargs)
    self.corner = corner
    self.segmentation_dir = segmentation_dir

  def init_coords(self):
    origins_to_invert = storage.load_origins(self.segmentation_dir,
                                             self.corner)
    points = origins_to_invert.items()
    points.sort(reverse=True)
    self.coords = np.array([origin_info.start_zyx for _, origin_info
                            in points])


class PolicyDenseSeeds(BaseSeedPolicy):
  """Dense seeds from thresholded image after optional erosion."""

  def __init__(self, canvas: Any, threshold: float = 0.5, num_erosions: int = 0,
               invert: bool = False, **kwargs):
    super().__init__(canvas, **kwargs)
    self._threshold = threshold
    self._num_erosions = num_erosions
    self._invert = invert

  def init_coords(self):
    img = self.canvas.image

    x = np.array(img > self._threshold).astype(bool)
    if self._invert:
      x = ~x
    for _ in range(self._num_erosions):
      x = skimage.morphology.binary_erosion(x)
    coords = np.where(x)

    self.coords = np.array(coords).T


class ReverseCoords(BaseSeedPolicy):
  """Wraps another policy and just reverses the seed order."""

  def __init__(self, canvas, policy_to_reverse: str, **policy_kwargs):
    super().__init__(canvas)
    policy_cls = globals()[policy_to_reverse]
    self._policy = policy_cls(canvas, **policy_kwargs)

  def init_coords(self):
    self.coords = np.array(list(self._policy)[::-1])


class SequentialPolicies(BaseSeedPolicy):
  """Applies policies sequentially."""

  def __init__(self, canvas, policies: Sequence[tuple[str, dict[str, Any]]],
               **kwargs):
    """Initializes the policies.

    Args:
      canvas: inference Canvas object
      policies: sequence of policies to chain together. Each entry is a tuple
        of size two; the name of the policy, followed by a keyword dict.
      **kwargs: other keyword arguments.
    """
    del kwargs
    super().__init__(canvas)
    self._policies = []
    for seed_policy, seed_policy_kwargs in policies:
      policy_cls = globals()[seed_policy]
      self._policies.append(policy_cls(canvas, **seed_policy_kwargs))

  def init_coords(self):
    self.coords = np.array(list(itertools.chain(*self._policies)))

  def get_state(self, previous=False):
    """Returns a pickleable state for this seeding policy.

    Args:
      previous: if True, indicates that a state for the already consumed seed,
        and so an in-progress segment, is being requested
    """
    states = []
    for policy in self._policies:
      states.append(policy.get_state(previous=previous))
    return states

  def set_state(self, state):
    for s, policy in zip(state, self._policies):
      policy.set_state(s)
