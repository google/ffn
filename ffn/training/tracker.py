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
"""Utilities for tracking and reporting the training status."""

import collections
import enum
import io
from typing import Any, Sequence

from absl import logging
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from scipy import special
import tensorflow.compat.v1 as tf

from . import mask
from . import variables


if tf.executing_eagerly():
  tf.compat.v2.experimental.numpy.experimental_enable_numpy_behavior()


class MoveType(enum.IntEnum):
  CORRECT = 0
  MISSED = 1
  SPURIOUS = 2


class VoxelType(enum.IntEnum):
  TOTAL = 0
  MASKED = 1


class PredictionType(enum.IntEnum):
  TP = 0
  TN = 1
  FP = 2
  FN = 3


class FovStat(enum.IntEnum):
  TOTAL_VOXELS = 0
  MASKED_VOXELS = 1
  WEIGHTS_SUM = 2


class EvalTracker:
  """Tracks eval results over multiple training steps."""

  def __init__(
      self, eval_shape: list[int], shifts: Sequence[tuple[int, int, int]]
  ):
    # TODO(mjanusz): Remove this TFv1 code once no longer used.
    if not tf.executing_eagerly():
      self.eval_labels = tf.compat.v1.placeholder(
          tf.float32, [1] + eval_shape + [1], name='eval_labels'
      )
      self.eval_preds = tf.compat.v1.placeholder(
          tf.float32, [1] + eval_shape + [1], name='eval_preds'
      )
      self.eval_weights = tf.compat.v1.placeholder(
          tf.float32, [1] + eval_shape + [1], name='eval_weights'
      )
      self.eval_loss = tf.reduce_mean(
          self.eval_weights
          * tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.eval_preds, labels=self.eval_labels
          )
      )
      self.sess = None
    self.eval_threshold = special.logit(0.9)
    self._eval_shape = eval_shape  # zyx
    self._define_tf_vars(shifts)
    self._patch_count = 0

    self.reset()

  def _add_tf_var(self, name, shape, dtype):
    v = variables.TFSyncVariable(name, shape, dtype)
    setattr(self, name, v)
    self._tf_vars.append(v)
    return v

  def _define_tf_vars(self, fov_shifts: Sequence[tuple[int, int, int]]):
    """Defines TFSyncVariables."""
    self._tf_vars = []
    self._add_tf_var('moves', [3], tf.int64)
    self._add_tf_var('loss', [1], tf.float32)
    self._add_tf_var('num_voxels', [2], tf.int64)
    self._add_tf_var('num_patches', [1], tf.int64)
    self._add_tf_var('prediction_counts', [4], tf.int64)
    self._add_tf_var('fov_stats', [3], tf.float32)

    radii = set([int(np.linalg.norm(s)) for s in fov_shifts])
    radii.add(0)
    self.moves_by_r = {}
    for r in radii:
      self.moves_by_r[r] = self._add_tf_var('moves_%d' % r, [3], tf.int64)

  def to_tf(self):
    ops = []
    feed_dict = {}

    for var in self._tf_vars:
      var.to_tf(ops, feed_dict)

    assert self.sess is not None
    self.sess.run(ops, feed_dict)

  def from_tf(self):
    ops = [var.from_tf for var in self._tf_vars]
    assert self.sess is not None
    values = self.sess.run(ops)

    for value, var in zip(values, self._tf_vars):
      var.tf_value = value

  def reset(self):
    """Resets status of the tracker."""
    self.images_xy = collections.deque(maxlen=16)
    self.images_xz = collections.deque(maxlen=16)
    self.images_yz = collections.deque(maxlen=16)
    self.meshes = collections.deque(maxlen=16 * 3)
    for var in self._tf_vars:
      var.reset()

  def track_weights(self, weights: np.ndarray):
    self.fov_stats.value[FovStat.TOTAL_VOXELS] += weights.size
    self.fov_stats.value[FovStat.MASKED_VOXELS] += np.sum(weights == 0.0)
    self.fov_stats.value[FovStat.WEIGHTS_SUM] += np.sum(weights)

  def record_move(
      self, wanted: bool, executed: bool, offset_xyz: Sequence[int]
  ):
    """Records an FFN FOV move."""
    r = int(np.linalg.norm(offset_xyz))
    assert r in self.moves_by_r, '%d not in %r' % (
        r,
        list(self.moves_by_r.keys()),
    )

    if wanted:
      if executed:
        self.moves.value[MoveType.CORRECT] += 1
        self.moves_by_r[r].value[MoveType.CORRECT] += 1
      else:
        self.moves.value[MoveType.MISSED] += 1
        self.moves_by_r[r].value[MoveType.MISSED] += 1
    elif executed:
      self.moves.value[MoveType.SPURIOUS] += 1
      self.moves_by_r[r].value[MoveType.SPURIOUS] += 1

  def slice_image(
      self,
      coord: np.ndarray,
      labels: np.ndarray,
      predicted: np.ndarray,
      weights: np.ndarray,
      slice_axis: int,
      volume_name: str | bytes | Sequence[Any] | np.ndarray | None = None,
  ) -> tf.Summary.Value:
    """Builds a tf.Summary showing a slice of an object mask.

    The object mask slice is shown side by side with the corresponding
    ground truth mask.

    Args:
      coord: [1, 3] xyz coordinate as ndarray
      labels: ndarray of ground truth data, shape [1, z, y, x, 1]
      predicted: ndarray of predicted data, shape [1, z, y, x, 1]
      weights: ndarray of loss weights, shape [1, z, y, x, 1]
      slice_axis: axis in the middle of which to place the cutting plane for
        which the summary image will be generated, valid values are 2 ('x'), 1
        ('y'), and 0 ('z').
      volume_name: name of the volume to be displayed on the image.

    Returns:
      tf.Summary.Value object with the image.
    """
    zyx = list(labels.shape[1:-1])
    selector = [0, slice(None), slice(None), slice(None), 0]
    selector[slice_axis + 1] = zyx[slice_axis] // 2
    selector = tuple(selector)  # for numpy indexing

    del zyx[slice_axis]
    h, w = zyx

    buf = io.BytesIO()
    labels = (labels[selector] * 255).astype(np.uint8)
    predicted = (predicted[selector] * 255).astype(np.uint8)
    weights = (weights[selector] * 255).astype(np.uint8)

    im = PIL.Image.fromarray(
        np.repeat(
            np.concatenate([labels, predicted, weights], axis=1)[
                ..., np.newaxis
            ],
            3,
            axis=2,
        ),
        'RGB',
    )
    draw = PIL.ImageDraw.Draw(im)

    x, y, z = coord.squeeze()
    text = f'{x},{y},{z}'
    if volume_name is not None:
      if (
          isinstance(volume_name, (list, tuple, np.ndarray))
          and len(volume_name) == 1
      ):
        volume_name = volume_name[0]

      if isinstance(volume_name, bytes):
        volume_name = volume_name.decode('utf-8')

      text += f'\n{volume_name}'

    try:

      # font = PIL.ImageFont.load_default()
    except (IOError, ValueError):
      font = PIL.ImageFont.load_default()

    draw.text((1, 1), text, fill='rgb(255,64,64)', font=font)
    del draw

    im.save(buf, 'PNG')

    axis_names = 'zyx'
    axis_names = axis_names.replace(axis_names[slice_axis], '')

    return tf.Summary.Value(
        tag='final_%s' % axis_names[::-1],
        image=tf.Summary.Image(
            height=h,
            width=w * 3,
            colorspace=3,  # RGB
            encoded_image_string=buf.getvalue(),
        ),
    )

  def add_patch(
      self,
      labels: np.ndarray,
      predicted: np.ndarray,
      weights: np.ndarray,
      coord: np.ndarray | None = None,
      image_summaries: bool = True,
      volume_name: str | None = None,
  ):
    """Evaluates single-object segmentation quality."""

    predicted = mask.crop_and_pad(predicted, (0, 0, 0), self._eval_shape)
    weights = mask.crop_and_pad(weights, (0, 0, 0), self._eval_shape)
    labels = mask.crop_and_pad(labels, (0, 0, 0), self._eval_shape)

    if not tf.executing_eagerly():
      assert self.sess is not None
      (loss,) = self.sess.run(
          [self.eval_loss],
          {
              self.eval_labels: labels,
              self.eval_preds: predicted,
              self.eval_weights: weights,
          },
      )
    else:
      loss = tf.reduce_mean(
          weights
          * tf.nn.sigmoid_cross_entropy_with_logits(
              logits=predicted, labels=labels
          )
      )

    self.loss.value[:] += loss
    self.num_voxels.value[VoxelType.TOTAL] += labels.size
    self.num_voxels.value[VoxelType.MASKED] += np.sum(weights == 0.0)

    pred_mask = predicted >= self.eval_threshold
    true_mask = labels > 0.5
    pred_bg = np.logical_not(pred_mask)
    true_bg = np.logical_not(true_mask)

    self.prediction_counts.value[PredictionType.TP] += np.sum(
        pred_mask & true_mask
    )
    self.prediction_counts.value[PredictionType.TN] += np.sum(pred_bg & true_bg)
    self.prediction_counts.value[PredictionType.FP] += np.sum(
        pred_mask & true_bg
    )
    self.prediction_counts.value[PredictionType.FN] += np.sum(
        pred_bg & true_mask
    )
    self.num_patches.value[:] += 1

    if image_summaries:
      predicted = special.expit(predicted)
      self.images_xy.append(
          self.slice_image(coord, labels, predicted, weights, 0, volume_name)
      )
      self.images_xz.append(
          self.slice_image(coord, labels, predicted, weights, 1, volume_name)
      )
      self.images_yz.append(
          self.slice_image(coord, labels, predicted, weights, 2, volume_name)
      )

  def _compute_classification_metrics(self, prediction_counts, prefix):
    """Computes standard classification metrics."""
    tp = prediction_counts.tf_value[PredictionType.TP]
    fp = prediction_counts.tf_value[PredictionType.FP]
    tn = prediction_counts.tf_value[PredictionType.TN]
    fn = prediction_counts.tf_value[PredictionType.FN]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    if precision > 0 or recall > 0:
      f1 = 2.0 * precision * recall / (precision + recall)
    else:
      f1 = 0.0

    return [
        tf.Summary.Value(
            tag='%s/accuracy' % prefix,
            simple_value=(tp + tn) / max(tp + tn + fp + fn, 1),
        ),
        tf.Summary.Value(tag='%s/precision' % prefix, simple_value=precision),
        tf.Summary.Value(tag='%s/recall' % prefix, simple_value=recall),
        tf.Summary.Value(
            tag='%s/specificity' % prefix, simple_value=tn / max(tn + fp, 1)
        ),
        tf.Summary.Value(tag='%s/f1' % prefix, simple_value=f1),
    ]

  def get_summaries(self) -> list[tf.Summary.Value]:
    """Gathers tensorflow summaries into single list."""

    self.from_tf()
    if not self.num_voxels.tf_value[VoxelType.TOTAL]:
      return []

    for images in self.images_xy, self.images_xz, self.images_yz:
      for i, summary in enumerate(images):
        summary.tag += '/%d' % i

    total_moves = sum(self.moves.tf_value)
    move_summaries = []
    for mt in MoveType:
      move_summaries.append(
          tf.Summary.Value(
              tag='moves/all/%s' % mt.name.lower(),
              simple_value=self.moves.tf_value[mt] / total_moves,
          )
      )

    summaries = (
        [
            tf.Summary.Value(
                tag='fov/masked_voxel_fraction',
                simple_value=(
                    self.fov_stats.tf_value[FovStat.MASKED_VOXELS]
                    / self.fov_stats.tf_value[FovStat.TOTAL_VOXELS]
                ),
            ),
            tf.Summary.Value(
                tag='fov/average_weight',
                simple_value=(
                    self.fov_stats.tf_value[FovStat.WEIGHTS_SUM]
                    / self.fov_stats.tf_value[FovStat.TOTAL_VOXELS]
                ),
            ),
            tf.Summary.Value(
                tag='masked_voxel_fraction',
                simple_value=(
                    self.num_voxels.tf_value[VoxelType.MASKED]
                    / self.num_voxels.tf_value[VoxelType.TOTAL]
                ),
            ),
            tf.Summary.Value(
                tag='eval/patch_loss',
                simple_value=self.loss.tf_value[0]
                / self.num_patches.tf_value[0],
            ),
            tf.Summary.Value(
                tag='eval/patches', simple_value=self.num_patches.tf_value[0]
            ),
            tf.Summary.Value(tag='moves/total', simple_value=total_moves),
        ]
        + move_summaries
        + (
            list(self.meshes)
            + list(self.images_xy)
            + list(self.images_xz)
            + list(self.images_yz)
        )
    )

    summaries.extend(
        self._compute_classification_metrics(self.prediction_counts, 'eval/all')
    )

    for r, r_moves in self.moves_by_r.items():
      total_moves = sum(r_moves.tf_value)
      summaries.extend([
          tf.Summary.Value(
              tag='moves/r=%d/correct' % r,
              simple_value=r_moves.tf_value[MoveType.CORRECT] / total_moves,
          ),
          tf.Summary.Value(
              tag='moves/r=%d/spurious' % r,
              simple_value=r_moves.tf_value[MoveType.SPURIOUS] / total_moves,
          ),
          tf.Summary.Value(
              tag='moves/r=%d/missed' % r,
              simple_value=r_moves.tf_value[MoveType.MISSED] / total_moves,
          ),
          tf.Summary.Value(
              tag='moves/r=%d/total' % r, simple_value=total_moves
          ),
      ])

    return summaries
