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
"""Functions for resegmentation result analysis."""

import re

import google3
import numpy as np
from scipy import ndimage

from google3.pyglib import gfile
from google3.pyglib import logging

from google3.research.neuromancer.segmentation.ffn import resegmentation_pb2
from google3.research.neuromancer.segmentation.ffn import storage
from google3.research.neuromancer.segmentation.python import pywrapsegment_util


class InvalidBaseSegmentatonError(Exception):
  pass


class IncompleteResegmentationError(Exception):
  pass


def compute_iou(reseg):
  """Computes the Jaccard index for two objects.

  Args:
    reseg: 4d boolean ndarray of mask for two objects over which to compute
        the JI, shape: [2, z, y, x]

  Returns:
    Jaccard index between two objects
  """
  return (np.sum(reseg[0, ...] & reseg[1, ...]) /
          float(np.sum(np.max(reseg, axis=0))))


def evaluate_segmentation_result(reseg, dels, moves, delta, analysis_r,
                                 seg1, seg2, sampling, result):
  """Computes statistics comparing resegmentation to original segmentation.

  Args:
    reseg: 3d Boolean array defining the mask of the object created in
        resegmentation, shape: [z, y, x]
    dels: list of numbers of voxels marked as deleted; every item in the list
        corresponds to an inference call of the FFN
    moves: array of network FoV locations (z, y, x) visited when creating the
        current object, shape: [n, 3]
    delta: (z, y, x) offset of the analysis subvolume within the resegmentation
        subvolume.
    analysis_r: (z, y, x) radius of the analysis subvolume
    seg1: binary map of the original segment A, shape: [z, y, x]
    seg2: binary map of the original segment B, shape: [z, y, x]
    sampling: (z, y, x) size of the voxel of the resegmentation object in nm
    result: SegmentResult proto to populate with statistics
  """
  result.max_edt = float(
      ndimage.distance_transform_edt(reseg, sampling=sampling).max())
  if moves.size > 0:
    corner0_zyx = np.array(delta)
    corner1_zyx = np.array(delta) + 2 * np.array(analysis_r)
    mask = np.all(
        (moves >= corner0_zyx[np.newaxis, ...]) &
        (moves <= corner1_zyx[np.newaxis, ...]),
        axis=1)
    result.deleted_voxels = long(np.sum(dels[mask]))

  result.num_voxels = long(np.sum(reseg))
  result.segment_a_consistency = float(
      np.sum(reseg[seg1])) / np.sum(seg1)
  result.segment_b_consistency = float(
      np.sum(reseg[seg2])) / np.sum(seg2)


def parse_resegmentation_filename(filename):
  logging.info('processing: %s', filename)
  id1, id2, x, y, z = [
      long(t) for t in
      re.search(r'(\d+)-(\d+)_at_(\d+)_(\d+)_(\d+)', filename).groups()]
  return id1, id2, x, y, z


def evaluate_endpoint_resegmentation(filename, seg_volstore,
                                     resegmentation_radius,
                                     threshold=0.5):
  """Evaluates endpoint resegmentation.

  Args:
    filename: path to the file containing resegmentation results
    seg_volstore: VolumeStore object with the original segmentation
    resegmentation_radius: (z, y, x) radius of the resegmentation subvolume
    threshold: threshold at which to create objects from the predicted
        object map

  Returns:
    EndpointResegmentationResult proto

  Raises:
    InvalidBaseSegmentatonError: when no base segmentation object with the
        expected ID matches the resegmentation data
  """
  id1, _, x, y, z = parse_resegmentation_filename(filename)

  result = resegmentation_pb2.EndpointSegmentationResult()
  result.id = id1
  start = result.start
  start.x, start.y, start.z = x, y, z

  sr = result.segmentation_radius
  sr.z, sr.y, sr.x = resegmentation_radius

  with gfile.Open(filename, 'r') as f:
    data = np.load(f)
    prob = storage.dequantize_probability(data['probs'])
    prob = np.nan_to_num(prob)  # nans indicate unvisited voxels

  sr = result.segmentation_radius
  orig_seg = seg_volstore[0,
                          (z - sr.z):(z + sr.z + 1),
                          (y - sr.y):(y + sr.y + 1),
                          (x - sr.x):(x + sr.x + 1)][0, ...]
  seg1 = orig_seg == id1
  if not np.any(seg1):
    raise InvalidBaseSegmentatonError()

  new_seg = prob[0, ...] >= threshold
  result.num_voxels = int(np.sum(new_seg))

  overlaps = pywrapsegment_util.ComputeOverlapCounts(
      orig_seg.ravel(), new_seg.astype(np.uint64).ravel())
  for k, v in overlaps.items():
    old, new = k
    if not new:
      continue

    result.overlaps[old].num_overlapping = v
    result.overlaps[old].num_original = int(np.sum(orig_seg == old))

    if old == id1:
      result.source.CopyFrom(result.overlaps[old])

  return result


def evaluate_pair_resegmentation(filename, seg_volstore,
                                 resegmentation_radius,
                                 analysis_radius,
                                 threshold=0.5):
  """Evaluates segment pair resegmentation.

  Args:
    filename: path to the file containing resegmentation results
    seg_volstore: VolumeStore object with the original segmentation
    resegmentation_radius: (z, y, x) radius of the resegmentation subvolume
    analysis_radius: (z, y, x) radius of the subvolume in which to perform
        analysis
    threshold: threshold at which to create objects from the predicted
        object map

  Returns:
    PairResegmentationResult proto

  Raises:
    IncompleteResegmentationError: when the resegmentation data does not
        represent two finished segments
    InvalidBaseSegmentatonError: when no base segmentation object with the
        excepted ID matches the resegmentation data
  """
  id1, id2, x, y, z = parse_resegmentation_filename(filename)

  result = resegmentation_pb2.PairResegmentationResult()
  result.id_a, result.id_b = id1, id2
  p = result.point
  p.x, p.y, p.z = x, y, z

  sr = result.segmentation_radius
  sr.z, sr.y, sr.x = resegmentation_radius

  with gfile.Open(filename, 'r') as f:
    data = np.load(f)
    prob = storage.dequantize_probability(data['probs'])
    prob = np.nan_to_num(prob)  # nans indicate unvisited voxels
    dels = data['deletes']
    moves = data['histories']  # z, y, x
    start_points = data['start_points']  # x, y, z

  if prob.shape[0] != 2:
    raise IncompleteResegmentationError()

  assert prob.ndim == 4

  # Corner of the resegmentation subvolume in the global coordinate system.
  corner = np.array([p.x - sr.x, p.y - sr.y, p.z - sr.z])

  # In case of multiple segmentation attempts, the last recorded start
  # point is the one we care about.
  origin_a = np.array(start_points[0][-1], dtype=np.int) + corner
  origin_b = np.array(start_points[1][-1], dtype=np.int) + corner
  oa = result.eval.from_a.origin
  oa.x, oa.y, oa.z = origin_a
  ob = result.eval.from_b.origin
  ob.x, ob.y, ob.z = origin_b

  # Record basic infromation about the resegmentation run.
  analysis_r = np.array(analysis_radius)
  r = result.eval.radius
  r.z, r.y, r.x = analysis_r

  seg = seg_volstore[0,
                     (z - analysis_r[0]):(z + analysis_r[0] + 1),
                     (y - analysis_r[1]):(y + analysis_r[1] + 1),
                     (x - analysis_r[2]):(x + analysis_r[2] + 1)][0, ...]
  seg1 = seg == id1
  seg2 = seg == id2
  result.eval.num_voxels_a = int(np.sum(seg1))
  result.eval.num_voxels_b = int(np.sum(seg2))

  if result.eval.num_voxels_a == 0 or result.eval.num_voxels_b == 0:
    raise InvalidBaseSegmentatonError()

  # Record information about the size of the original segments.
  sampling = (seg_volstore.info.pixelsize.z,
              seg_volstore.info.pixelsize.y,
              seg_volstore.info.pixelsize.x)
  result.eval.max_edt_a = float(
      ndimage.distance_transform_edt(seg1, sampling=sampling).max())
  result.eval.max_edt_b = float(
      ndimage.distance_transform_edt(seg2, sampling=sampling).max())

  # Offset of the analysis subvolume within the resegmentation subvolume.
  delta = np.array(resegmentation_radius) - analysis_r
  prob = prob[:,
              delta[0]:(delta[0] + 2 * analysis_r[0] + 1),
              delta[1]:(delta[1] + 2 * analysis_r[1] + 1),
              delta[2]:(delta[2] + 2 * analysis_r[2] + 1)]
  reseg = prob >= threshold
  result.eval.iou = compute_iou(reseg)

  # Record information about the size of the reconstructed segments.
  evaluate_segmentation_result(
      reseg[0, ...], dels[0], moves[0], delta, analysis_r, seg1, seg2,
      sampling, result.eval.from_a)
  evaluate_segmentation_result(
      reseg[1, ...], dels[1], moves[1], delta, analysis_r, seg1, seg2,
      sampling, result.eval.from_b)

  return result
