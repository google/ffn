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
"""Helper functions for resegmentation.

Resegmentation is local segmentation targeted to specific points in an already
segmented volume. The results of resegmentation can be compared to the original
segments in order to perform object agglomeration.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import os

import numpy as np

from scipy import ndimage
from scipy.special import expit

from tensorflow import gfile

from . import storage
from .inference_utils import timer_counter


def get_starting_location(dists, exclusion_radius):
  z, y, x = np.unravel_index(np.argmax(dists), tuple(dists.shape))
  # Mark area around the new point as 'excluded' by clearing the distance
  # map around it.
  er = exclusion_radius
  dists[max(z - er.z, 0):z + er.z + 1,
        max(y - er.y, 0):y + er.y + 1,
        max(x - er.x, 0):x + er.x + 1] = 0
  return z, y, x


def get_target_path(request, point_num):
  """Computes the output path for a specific point.

  Args:
    request: ResegmentationRequest proto
    point_num: index of the point of interest within the proto

  Returns:
    path to the output file where resegmentation results will be saved
  """
  # Prepare the output directory.
  output_dir = request.output_directory

  id_a = request.points[point_num].id_a
  id_b = request.points[point_num].id_b

  if request.subdir_digits > 1:
    m = hashlib.md5()
    m.update(str(id_a))
    m.update(str(id_b))
    output_dir = os.path.join(output_dir, m.hexdigest()[:request.subdir_digits])
  gfile.MakeDirs(output_dir)

  # Terminate early if the output already exists.
  dp = request.points[point_num].point
  target_path = os.path.join(output_dir, '%d-%d_at_%d_%d_%d.npz' % (
      id_a, id_b, dp.x, dp.y, dp.z))
  if gfile.Exists(target_path):
    logging.info('Output already exists: %s', target_path)
    return

  return target_path


def get_canvas(point, radius, runner):
  """Creates an FFN Canvas.

  Args:
    point: decision point as (z, y, x)
    radius: radius around decision point as (z, y, x)
    runner: inference Runner object

  Returns:
    inference Canvas object
  """

  origin = np.array(point)
  radius = np.array(radius)
  corner = origin - radius
  subvol_size = radius * 2 + 1
  end = subvol_size + corner

  if (np.any(corner < 0) or
      runner.init_seg_volstore.size.z <= end[0] or
      runner.init_seg_volstore.size.y <= end[1] or
      runner.init_seg_volstore.size.x <= end[2]):
    logging.error('Not enough context for: %d, %d, %d; corner: %r; end: %r',
                  point[2], point[1], point[0], corner, end)
    return None, None

  return runner.make_canvas(corner, subvol_size, keep_history=True)


def process_point(request, runner, point_num):
  """Runs resegmentation for a specific point.

  Args:
    request: ResegmentationRequest proto
    runner: inference Runner object
    point_num: index of the point of interest within the proto
  """
  with timer_counter(runner.counters, 'resegmentation'):
    target_path = get_target_path(request, point_num)
    if target_path is None:
      return

    curr = request.points[point_num]
    point = curr.point
    point = point.z, point.y, point.x
    radius = (request.radius.z, request.radius.y, request.radius.x)
    canvas, alignment = get_canvas(point, radius, runner)
    if canvas is None:
      logging.warning('Could not get a canvas object.')
      return

    def unalign_prob(prob):
      return alignment.align_and_crop(
          canvas.corner_zyx,
          prob,
          alignment.corner,
          alignment.size,
          forward=False)

    is_shift = (canvas.restrictor is not None and
                np.any(canvas.restrictor.shift_mask))
    is_endpoint = not curr.HasField('id_b')

    seg_a = canvas.segmentation == canvas.local_id(curr.id_a)
    size_a = np.sum(seg_a)

    if is_endpoint:
      size_b = -1
      todo = [seg_a]
    else:
      seg_b = canvas.segmentation == canvas.local_id(curr.id_b)
      size_b = np.sum(seg_b)
      todo = [seg_a, seg_b]

    if size_a == 0 or size_b == 0:
      logging.warning('Segments (%d, %d) local ids (%d, %d) not found in input '
                      'at %r.  Current values are: %r.',
                      curr.id_a, curr.id_b, canvas.local_id(curr.id_a),
                      canvas.local_id(curr.id_b), point,
                      np.unique(canvas.segmentation))
      canvas._deregister_client()  # pylint:disable=protected-access
      return

    if is_endpoint:
      canvas.seg_prob[:] = 0.0
      canvas.segmentation[:] = 0
    else:
      # Clear the two segments in question, but keep everything else as
      # context.
      canvas.segmentation[seg_a] = 0
      canvas.segmentation[seg_b] = 0
      canvas.seg_prob[seg_a] = 0.0
      canvas.seg_prob[seg_b] = 0.0

    transformed_point = alignment.transform(np.array([point]).T)
    tz, ty, tx = transformed_point[:, 0]
    oz, oy, ox = canvas.corner_zyx
    tz -= oz
    ty -= oy
    tx -= ox

    # First index enumerates the original segments. Second index,
    # when present, enumerates segmentation attempts.
    raw_probs = []
    probs = []
    deletes = []
    histories = []
    start_points = [[], []]

    if request.HasField('analysis_radius'):
      ar = request.analysis_radius
      analysis_box = bounding_box.BoundingBox(
          start=(radius[2] - ar.x,
                 radius[1] - ar.y,
                 radius[0] - ar.z),
          size=(2 * ar.x + 1, 2 * ar.y + 1, 2 * ar.z + 1))
    else:
      analysis_box = bounding_box.BoundingBox(
          (0, 0, 0), canvas.image.shape[::-1])

    options = request.inference.inference_options
    for i, seg in enumerate(todo):
      logging.info('processing object %d', i)

      with timer_counter(canvas.counters, 'edt'):
        ps = runner.init_seg_volstore.info.pixelsize
        dists = ndimage.distance_transform_edt(seg, sampling=(ps.z, ps.y, ps.x))
        # Do not seed where not enough context is available.
        dists[:canvas.margin[0], :, :] = 0
        dists[:, :canvas.margin[1], :] = 0
        dists[:, :, :canvas.margin[2]] = 0
        dists[-canvas.margin[0]:, :, :] = 0
        dists[:, -canvas.margin[1]:, :] = 0
        dists[:, :, -canvas.margin[2]:] = 0
        canvas.log_info('EDT computation done')

      # Optionally exclude a region around the decision point from seeding.
      if request.HasField('init_exclusion_radius'):
        ier = request.init_exclusion_radius
        dists[tz - ier.z:tz + ier.z + 1,
              ty - ier.y:ty + ier.y + 1,
              tx - ier.x:tx + ier.x + 1] = 0

      seg_prob = None
      recovered = False

      for _ in range(request.max_retry_iters):
        z0, y0, x0 = get_starting_location(dists, request.exclusion_radius)
        if not seg[z0, y0, x0]:
          continue

        canvas.log_info('.. starting segmentation at (xyz): %d %d %d',
                        x0, y0, z0)
        canvas.segment_at((z0, y0, x0))
        seg_prob = expit(canvas.seed)
        start_points[i].append((x0, y0, z0))

        # Check if we recovered an acceptable fraction of the initial segment
        # in which the seed was located.
        recovered = True

        crop_seg = seg[analysis_box.to_slice()]
        crop_prob = seg_prob[analysis_box.to_slice()]
        start_size = np.sum(crop_seg)
        segmented_voxels = np.sum((crop_prob >= options.segment_threshold) &
                                  crop_seg)
        if request.segment_recovery_fraction > 0:
          if segmented_voxels / start_size >= request.segment_recovery_fraction:
            break
        elif segmented_voxels >= options.min_segment_size:
          break

        recovered = False

      # Store resegmentation results.
      if seg_prob is not None:
        qprob = storage.quantize_probability(seg_prob)
        raw_probs.append(qprob)
        probs.append(unalign_prob(qprob))
        deletes.append(np.array(canvas.history_deleted))
        histories.append(np.array(canvas.history))

      if request.terminate_early:
        if not recovered:
          break
        if (request.segment_recovery_fraction > 0 and i == 0 and
            len(todo) > 1):
          seg2 = todo[1]
          crop_seg = seg2[analysis_box.to_slice()]
          size2 = np.sum(crop_seg)
          segmented_voxels2 = np.sum(
              (crop_prob >= options.segment_threshold) & crop_seg)

          if segmented_voxels2 / size2 < request.segment_recovery_fraction:
            break

  canvas.log_info('saving results to %s', target_path)
  with storage.atomic_file(target_path) as fd:
    np.savez_compressed(fd,
                        probs=np.array(probs),
                        raw_probs=np.array(raw_probs),
                        deletes=np.array(deletes),
                        histories=np.array(histories),
                        start_points=start_points,
                        request=request.SerializeToString(),
                        counters=canvas.counters.dumps(),
                        corner_zyx=canvas.corner_zyx,
                        is_shift=is_shift)
  canvas.log_info('.. save complete')
  # Cannot `del canvas` here in Python 2 -- deleting an object referenced
  # in a nested scope is a syntax error.
  canvas._deregister_client()  # pylint:disable=protected-access


def process(request, runner):
  num_points = len(request.points)
  for i in range(num_points):
    logging.info('processing %d/%d', i, num_points)
    process_point(request, runner, i)

