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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import weakref

import numpy as np
from scipy import ndimage
import skimage
import skimage.feature
from skimage import morphology
import networkx as nx
from scipy.special import expit
from skan import skeleton_to_csgraph

from . import storage


class BaseSeedPolicy(object):
  """Base class for seed policies."""

  def __init__(self, canvas, **kwargs):
    """Initializes the policy.

    Args:
      canvas: inference Canvas object; simple policies use this to access
          basic geometry information such as the shape of the subvolume;
          more complex policies can access the raw image data, etc.
      **kwargs: other keyword arguments
    """
    del kwargs
    # TODO(mjanusz): Remove circular reference between Canvas and seed policies.
    self.canvas = weakref.proxy(canvas)
    # JG: self.coords will be an ndarray of shape [n_coords, 3]
    # representing (z, y, x) locations in the image to use as seeds.
    self.coords = None
    self.idx = 0

  def _init_coords(self):
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
      self._init_coords()

    while self.idx < self.coords.shape[0]:
      curr = self.coords[self.idx, :]
      self.idx += 1

      # TODO(mjanusz): Get rid of this.
      # Do early filtering of clearly invalid locations (too close to image
      # borders) as late filtering might be expensive.
      if (np.all(curr - self.canvas.margin >= 0) and
          np.all(curr + self.canvas.margin < self.canvas.shape)):
        return tuple(curr)  # z, y, x

    raise StopIteration()

  def next(self):
    return self.__next__()

  def get_state(self):
    return self.coords, self.idx

  def set_state(self, state):
    self.coords, self.idx = state


class PolicyPeaks(BaseSeedPolicy):
  """Attempts to find points away from edges in the image.

  Runs a 3d Sobel filter to detect edges in the raw data, followed
  by a distance transform and peak finding to identify seed points.
  """

  def _init_coords(self):
    logging.info('peaks: starting')

    # Edge detection.
    edges = ndimage.generic_gradient_magnitude(
        self.canvas.image.astype(np.float32),
        ndimage.sobel)

    # Adaptive thresholding.
    sigma = 49.0 / 6.0
    thresh_image = np.zeros(edges.shape, dtype=np.float32)
    ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
    filt_edges = edges > thresh_image

    del edges, thresh_image

    # This prevents a border effect where the large amount of masked area
    # screws up the distance transform below.
    if (self.canvas.restrictor is not None and
        self.canvas.restrictor.mask is not None):
      filt_edges[self.canvas.restrictor.mask] = 1

    logging.info('peaks: filtering done')
    dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)
    logging.info('peaks: edt done')

    # Use a specifc seed for the noise so that results are reproducible
    # regardless of what happens before the policy is called.
    state = np.random.get_state()
    np.random.seed(42)
    idxs = skimage.feature.peak_local_max(
        dt + np.random.random(dt.shape) * 1e-4,
        indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
    np.random.set_state(state)

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

    Args:
      canvas: inference Canvas object.
      min_distance: forwarded to peak_local_max.
      threshold_abs: forwarded to peak_local_max.
      sort_cmp: 'ascending' or 'descending' for sorting seed coordinates.
      **kwargs: forwarded to base.

    For compatibility with original version, min_distance=3, threshold_abs=0,
    sort=False.
    """
    super(PolicyPeaks2d, self).__init__(canvas, **kwargs)
    self.min_distance = min_distance
    self.threshold_abs = threshold_abs
    self.sort_reverse = sort_cmp.strip().lower().startswith('de')

  def _init_coords(self):
    logging.info('2d peaks: starting')

    # Loop over 2d slices.
    for z in range(self.canvas.image.shape[0]):
      image_2d = (self.canvas.image[z, :, :]).astype(np.float32)

      # Edge detection.
      edges = ndimage.generic_gradient_magnitude(
          image_2d, ndimage.sobel)

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
      dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)

      # Use a specifc seed for the noise so that results are reproducible
      # regardless of what happens before the policy is called.
      state = np.random.get_state()
      np.random.seed(42)
      idxs = skimage.feature.peak_local_max(
          dt + np.random.random(dt.shape) * 1e-4,
          indices=True, min_distance=3, threshold_abs=0, threshold_rel=0)
      zs = np.full((idxs.shape[0], 1), z, dtype=np.int64)
      idxs = np.concatenate((zs, idxs), axis=1)
      np.random.set_state(state)

      # Update self.coords with indices found at this z index
      logging.info('2d peaks: found %d local maxima at z index %d',
                   idxs.shape[0], z)
      self.coords = np.concatenate((self.coords, idxs)) if z != 0 else idxs

    self.coords = np.array(
        sorted([(z, y, x) for z, y, x in self.coords], reverse=self.sort_reverse))

    logging.info('2d peaks: found %d total local maxima', self.coords.shape[0])


class PolicyMax(BaseSeedPolicy):
  """All points in the image, in descending order of intensity."""

  def _init_coords(self):
    idxs = np.mgrid[[slice(0, x) for x in self.canvas.image.shape]]
    sort_idx = np.argsort(self.canvas.image.flat)[::-1]
    self.coords = np.array(zip(*[idx.flat[sort_idx] for idx in idxs]))


class PolicyGrid3d(BaseSeedPolicy):
  """Points distributed on a uniform 3d grid."""

  def __init__(self, canvas, step=16, offsets=(0, 8, 4, 12, 2, 10, 14),
               **kwargs):
    super(PolicyGrid3d, self).__init__(canvas, **kwargs)
    self.step = step
    self.offsets = offsets

  def _init_coords(self):
    self.coords = []
    for offset in self.offsets:
      for z in range(offset, self.canvas.image.shape[0], self.step):
        for y in range(offset, self.canvas.image.shape[1], self.step):
          for x in range(offset, self.canvas.image.shape[2], self.step):
            self.coords.append((z, y, x))
    self.coords = np.array(self.coords)


class PolicyGrid2d(BaseSeedPolicy):
  """Points distributed on a uniform 2d grid."""

  def __init__(self, canvas, step=16, offsets=(0, 8, 4, 12, 2, 6, 10, 14),
               **kwargs):
    super(PolicyGrid2d, self).__init__(canvas, **kwargs)
    self.step = step
    self.offsets = offsets

  def _init_coords(self):
    self.coords = []
    for offset in self.offsets:
      for z in range(self.canvas.image.shape[0]):
        for y in range(offset, self.canvas.image.shape[1], self.step):
          for x in range(offset, self.canvas.image.shape[2], self.step):
            self.coords.append((z, y, x))
    self.coords = np.array(self.coords)


class PolicyInvertOrigins(BaseSeedPolicy):

  def __init__(self, canvas, corner=None, segmentation_dir=None,
               **kwargs):
    super(PolicyInvertOrigins, self).__init__(canvas, **kwargs)
    self.corner = corner
    self.segmentation_dir = segmentation_dir

  def _init_coords(self):
    origins_to_invert = storage.load_origins(self.segmentation_dir,
                                             self.corner)
    points = origins_to_invert.items()
    points.sort(reverse=True)
    self.coords = np.array([origin_info.start_zyx for _, origin_info
                            in points])


class SeedPolicyWithSaver(BaseSeedPolicy):
    """A seed policy which also optionally saves the seed at specified intervals."""
    def __init__(self, canvas, save_history_every=None, **kwargs):
        super(SeedPolicyWithSaver, self).__init__(canvas, **kwargs)
        self.save_history_every = save_history_every
        # Seed history is an ndarray of shape [z, y, x, iter_num].
        self.seed_history = np.expand_dims(np.zeros_like(self.canvas.image), -1)
        self.coord_history = None

    def save_seed_history(self):
        self.seed_history = np.concatenate((self.seed_history,
                                            np.expand_dims(self.canvas.seed, -1)),
                                           axis=-1)

    def _check_save_history(self):
        """Check whether seeds should be saved at this iteration; if so, save them."""
        if not self.save_history_every:
            return
        if self.idx % self.save_history_every == 0:
            self.save_seed_history()


class ManualSeedPolicy(SeedPolicyWithSaver):
    """Use a manually-specified set of seeds."""
    def __init__(self, canvas, save_history_every=None):
      logging.info("ManualSeedPolicy.__init__()")
      super(ManualSeedPolicy, self).__init__(canvas, save_history_every, **kwargs)

    def _init_coords(self):
        # TODO(jpgard): collect these from user; temporarily these are hard-coded.
        coords = [(0, 4521, 3817),  # soma center
                  # (0, 3416, 248),  # bad seed in dark cloud for testing
                  (0, 3477, 3936),
                  (0, 3744, 4413),
                  (0, 2937, 5097),
                  (0, 3428, 3795),
                  (0, 3486, 3945),
                  (0, 3548, 4053),
                  (0, 3610, 4178),
                  (0, 3651, 4282),
                  (0, 3662, 4333),
                  (0, 3634, 4384),
                  (0, 3522, 4375),
                  (0, 3345, 4359),
                  (0, 2384, 5579),  # axon
                  (0, 2205, 5702),  # axon
                  (0, 1962, 5901),  # axon
                  (0, 1713, 6111),  # axon
                  (0, 1484, 6274),  # junction of axon branches
                  (0, 1025, 6452),  # in axon branches
                  (0, 822, 6521),  # in axon branches
                  (0, 773, 6560),  # in axon branches
                  (0, 402, 6453),  # point in "twist" of axon branches
                  (0, 1167, 6423),  # in axon branches
                  (0, 1197, 6627),  # in axon branches
                  (0, 1074, 6735),  # in axon branches
                  (0, 1428, 6948),  # in axon branches
                  ]
        logging.info('ManualSeedPolicy: starting with coords {}'.format(coords))
        self.coords = np.array(coords)

    def __next__(self):
        """Returns the next seed point as (z, y, x).

        Returns:
          (z, y, x) tuples.

        Raises:
          StopIteration when the seeds are exhausted.
        """
        if self.coords is None:
            self._init_coords()

        while self.idx < self.coords.shape[0]:
            self._check_save_history()
            curr = self.coords[self.idx, :]
            self.idx += 1
            logging.info("ManualSeedPolicy processing seed: {}".format(curr))
            return tuple(curr)  # z, y, x

        raise StopIteration()

class TipTracerSeedPolicy(SeedPolicyWithSaver):

    def __init__(self, canvas, save_history_every=None, skeletonization_threshold=0.5,
                 **kwargs):
        """
        At each iteration, add the tips of the trace to the list of seeds.

        :param canvas: Canvas object
        each step (this is used to threshold the predicted probabilities in canvas).
        :param save_seeds_every: save canvas at this interval; this consumes a lot of
        memory for large arrays but is useful for debugging and visualization of
        inference results. Note that the results are only written to disk after
        completion of the Runner.run(), not after each iteration.
        :param kwargs: other kwargs passed to BaseSeedPolicy constructor.
        """

        super(TipTracerSeedPolicy, self).__init__(canvas, **kwargs)
        self.skeletonization_threshold = skeletonization_threshold
        self.save_seed_every = save_history_every
        self.skeleton_history = np.expand_dims(np.zeros_like(self.canvas.image), -1)

    def _init_coords(self):
        """Initialize array of seed coordinates in (z, y, x) format."""
        coords = [(0, 4521, 3817, 0),  # soma center
                  ]
        self.coords = np.array(coords)

    def _check_save_skeleton(self, skel):
        """Check whether skeleton should be saved, and save it."""
        if not self.save_history_every:
            return
        if self.idx % self.save_history_every == 0:
            # save the skeleton
            if len(skel.shape) == 2:
                # skel is a 2D array; need to add a z-axis
                skel_exp = np.expand_dims(skel, 0)
            skel_exp = np.expand_dims(skel_exp, -1)
            self.skeleton_history = np.concatenate(
                (
                    self.skeleton_history,
                    skel_exp),
                axis=-1
            )

    def __next__(self):
        """Update the list of seeds and return the next seed point as (z, y, x).

        Applies skeletonization to the existing canvas, extracts leaf nodes of the
        current skeleton, and adds the coordinates of leaf nodes to the list of seeds.

        Returns:
          (z, y, x) tuples.

        Raises:
          StopIteration when the seeds are exhausted.
        """
        if self.coords is None:
            self._init_coords()

        if self.idx > 0:  # Only extract new tips after inference has run at least once.

            logging.info("TipTracerSeedPolicy skeletonizing and extracting seeds")
            # Transform logits to probabilities, apply threshold, and skeletonize to
            # extract the locations of leaf nodes ("tips")
            c_t = expit(np.squeeze(self.canvas.seed))
            c_t = np.nan_to_num(c_t)
            c_t = (c_t >= self.skeletonization_threshold).astype(np.uint8)
            s_t = morphology.skeletonize(c_t)
            self._check_save_skeleton(s_t)
            g_t, c_t, _ = skeleton_to_csgraph(s_t)
            g_t = nx.from_scipy_sparse_matrix(g_t)
            # Find largest connected component and extract leaf nodes.
            Gc = max(nx.connected_component_subgraphs(g_t), key=len)
            leaf_node_ids = [node_id for node_id, node_degree
                             in nx.degree(Gc, Gc.nodes())
                             if node_degree == 1
                             ]
            # Add the leaf nodes as new seeds
            logging.info("adding {} nodes to coords at iteration {}".format(
                len(leaf_node_ids), self.idx
            ))
            new_seeds = c_t[leaf_node_ids, :].astype(int)
            new_seeds = np.hstack(
                (np.zeros((len(leaf_node_ids), 1), dtype=int),  # fix z-coordinate at zero
                 new_seeds,
                 np.full((len(leaf_node_ids), 1), fill_value=self.idx, dtype=int)
                 )
            )
            self.coords = np.vstack((self.coords, new_seeds))
            self._check_save_history()

        # while self.idx < self.coords.shape[0]:
        while self.idx < 20:
            curr = self.coords[self.idx, :3]
            self.idx += 1
            return tuple(curr)

        raise StopIteration()
