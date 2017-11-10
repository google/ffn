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
"""Functions for computing consensus between FFN segmentations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import logging
import os
import numpy as np
import psutil


from . import consensus_pb2
from . import storage
from . import segmentation


def compute_consensus_for_segmentations(v1, v2, request):
  """Computes consensus between two segmentations.

  Args:
    v1: 1st segmentation as a 3d integer ndarray
    v2: 2nd segmentation as a 3d integer ndarray
    request: ConsensusRequest proto

  Returns:
    3d consensus segmentation array (integer ndarray)

  Raises:
    ValueError: if an unsupported consensus type is requested
  """
  if request.type == consensus_pb2.ConsensusRequest.CONSENSUS_SPLIT:
    # We reduce the data type size after split consensus, which creates new
    # segments, so we do not know a priori how many bits will be needed for
    # the IDs.
    segmentation.split_segmentation_by_intersection(v1, v2,
                                                    request.split_min_size)
    v1 = segmentation.reduce_id_bits(v1)
  else:
    raise ValueError('Unsupported mode: %s' % request.type)

  return v1


def compute_consensus(corner, request):
  """Computes consensus segmentation between two FFN subvolumes.

  Args:
    corner: lower corner of the subvolume for which to compute consensus as
        a (z, y, x) tuple
    request: ConsensusRequest proto

  Returns:
    tuple of:
      consensus segmentation as a z, y, x uint numpy array
      origin dictionary for the consensus segmentation (keys are segment IDs,
          values are tuples of: (seed location in x, y, z;
          number of FFN iterations used to produce the segment;
          wall clock time in seconds used for inference).
  """
  process = psutil.Process(os.getpid())
  logging.info('consensus: mem[start] = %d MiB',
               process.memory_info()[0] / 2**20)

  v1, v1_origins = storage.load_segmentation_from_source(request.segmentation1,
                                                         corner)
  logging.info('consensus: v1 data loaded')
  v2, _, = storage.load_segmentation_from_source(request.segmentation2, corner)
  logging.info('consensus: v2 data loaded')

  logging.info('consensus: mem[data loaded] = %d MiB',
               process.memory_info()[0] / 2**20)

  v1 = compute_consensus_for_segmentations(v1, v2, request)

  # Retain origin information for the segments that remain.
  relabeled_origins = {}
  for seg_id in np.unique(v1):
    if seg_id == 0:
      continue
    if seg_id in v1_origins:
      relabeled_origins[seg_id] = v1_origins[seg_id]

  return v1, relabeled_origins
