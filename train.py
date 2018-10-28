# Copyright 2017-2018 Google Inc.
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
"""Driver script for FFN training.

The FFN is first run on single seed point. The mask prediction for that seed
point is then used to train subsequent steps of the FFN by moving the field
of view in a way dependent on the initial predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
from io import BytesIO
from functools import partial
import itertools
import json
import logging
import os
import random
import time

import h5py
import numpy as np

import PIL
import PIL.Image

import six

from scipy.special import expit
from scipy.special import logit
import tensorflow as tf

from absl import app
from absl import flags
from tensorflow import gfile

from ffn.inference import movement
from ffn.training import mask
from ffn.training.import_util import import_symbol
from ffn.training import inputs
from ffn.training import augmentation
# Necessary so that optimizer flags are defined.
# pylint: disable=unused-import
from ffn.training import optimizer
# pylint: enable=unused-import

FLAGS = flags.FLAGS

# Options related to training data.
flags.DEFINE_string('train_coords', None,
                    'Glob for the TFRecord of training coordinates.')
flags.DEFINE_string('data_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing uint8 '
                    'image data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('label_volumes', None,
                    'Comma-separated list of <volume_name>:<volume_path>:'
                    '<dataset>, where volume_name need to match the '
                    '"label_volume_name" field in the input example, '
                    'volume_path points to HDF5 volumes containing int64 '
                    'label data, and `dataset` is the name of the dataset '
                    'from which data will be read.')
flags.DEFINE_string('model_name', None,
                    'Name of the model to train. Format: '
                    '[<packages>.]<module_name>.<model_class>, if packages is '
                    'missing "ffn.training.models" is used as default.')
flags.DEFINE_string('model_args', None,
                    'JSON string with arguments to be passed to the model '
                    'constructor.')

# Training infra options.
flags.DEFINE_string('train_dir', '/tmp',
                    'Path where checkpoints and other data will be saved.')
flags.DEFINE_string('master', '', 'Network address of the master.')
flags.DEFINE_integer('batch_size', 4, 'Number of images in a batch.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of tasks in the ps job.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train for.')
flags.DEFINE_integer('replica_step_delay', 300,
                     'Require the model to reach step number '
                     '<replica_step_delay> * '
                     '<replica_id> before starting training on a given '
                     'replica.')
flags.DEFINE_integer('summary_rate_secs', 120,
                     'How often to save summaries (in seconds).')

# FFN training options.
flags.DEFINE_float('seed_pad', 0.05,
                   'Value to use for the unknown area of the seed.')
flags.DEFINE_float('threshold', 0.9,
                   'Value to be reached or exceeded at the new center of the '
                   'field of view in order for the network to inspect it.')
flags.DEFINE_enum('fov_policy', 'fixed', ['fixed', 'max_pred_moves'],
                  'Policy to determine where to move the field of the '
                  'network. "fixed" tries predefined offsets specified by '
                  '"model.shifts". "max_pred_moves" moves to the voxel with '
                  'maximum mask activation within a plane perpendicular to '
                  'one of the 6 Cartesian directions, offset by +/- '
                  'model.deltas from the current FOV position.')
# TODO(mjanusz): Implement fov_moves > 1 for the 'fixed' policy.
flags.DEFINE_integer('fov_moves', 1,
                     'Number of FOV moves by "model.delta" voxels to execute '
                     'in every dimension. Currently only works with the '
                     '"max_pred_moves" policy.')
flags.DEFINE_boolean('shuffle_moves', True,
                     'Whether to randomize the order of the moves used by the '
                     'network with the "fixed" policy.')

flags.DEFINE_float('image_mean', None,
                   'Mean image intensity to use for input normalization.')
flags.DEFINE_float('image_stddev', None,
                   'Image intensity standard deviation to use for input '
                   'normalization.')
flags.DEFINE_list('image_offset_scale_map', None,
                  'Optional per-volume specification of mean and stddev. '
                  'Every entry in the list is a colon-separated tuple of: '
                  'volume_label, offset, scale.')

flags.DEFINE_list('permutable_axes', ['1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be permuted '
                  'in order to augment the training data.')

flags.DEFINE_list('reflectable_axes', ['0', '1', '2'],
                  'List of integers equal to a subset of [0, 1, 2] specifying '
                  'which of the [z, y, x] axes, respectively, may be reflected '
                  'in order to augment the training data.')

FLAGS = flags.FLAGS


class EvalTracker(object):
  """Tracks eval results over multiple training steps."""

  def __init__(self, eval_shape):
    self.eval_labels = tf.placeholder(
        tf.float32, [1] + eval_shape + [1], name='eval_labels')
    self.eval_preds = tf.placeholder(
        tf.float32, [1] + eval_shape + [1], name='eval_preds')
    self.eval_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.eval_preds, labels=self.eval_labels))
    self.reset()
    self.eval_threshold = logit(0.9)
    self.sess = None
    self._eval_shape = eval_shape

  def reset(self):
    """Resets status of the tracker."""
    self.loss = 0
    self.num_patches = 0
    self.tp = 0
    self.tn = 0
    self.fn = 0
    self.fp = 0
    self.total_voxels = 0
    self.masked_voxels = 0
    self.images_xy = deque(maxlen=16)
    self.images_xz = deque(maxlen=16)
    self.images_yz = deque(maxlen=16)

  def slice_image(self, labels, predicted, weights, slice_axis):
    """Builds a tf.Summary showing a slice of an object mask.

    The object mask slice is shown side by side with the corresponding
    ground truth mask.

    Args:
      labels: ndarray of ground truth data, shape [1, z, y, x, 1]
      predicted: ndarray of predicted data, shape [1, z, y, x, 1]
      weights: ndarray of loss weights, shape [1, z, y, x, 1]
      slice_axis: axis in the middle of which to place the cutting plane
          for which the summary image will be generated, valid values are
          2 ('x'), 1 ('y'), and 0 ('z').

    Returns:
      tf.Summary.Value object with the image.
    """
    zyx = list(labels.shape[1:-1])
    selector = [0, slice(None), slice(None), slice(None), 0]
    selector[slice_axis + 1] = zyx[slice_axis] // 2
    selector = tuple(selector)

    del zyx[slice_axis]
    h, w = zyx

    buf = BytesIO()
    labels = (labels[selector] * 255).astype(np.uint8)
    predicted = (predicted[selector] * 255).astype(np.uint8)
    weights = (weights[selector] * 255).astype(np.uint8)

    im = PIL.Image.fromarray(np.concatenate([labels, predicted,
                                             weights], axis=1), 'L')
    im.save(buf, 'PNG')

    axis_names = 'zyx'
    axis_names = axis_names.replace(axis_names[slice_axis], '')

    return tf.Summary.Value(
        tag='final_%s' % axis_names[::-1],
        image=tf.Summary.Image(
            height=h, width=w * 3, colorspace=1,  # greyscale
            encoded_image_string=buf.getvalue()))

  def add_patch(self, labels, predicted, weights,
                coord=None, volname=None, patches=None):
    """Evaluates single-object segmentation quality."""

    predicted = mask.crop_and_pad(predicted, (0, 0, 0), self._eval_shape)
    weights = mask.crop_and_pad(weights, (0, 0, 0), self._eval_shape)
    labels = mask.crop_and_pad(labels, (0, 0, 0), self._eval_shape)
    loss, = self.sess.run([self.eval_loss], {self.eval_labels: labels,
                                             self.eval_preds: predicted})
    self.loss += loss
    self.total_voxels += labels.size
    self.masked_voxels += np.sum(weights == 0.0)

    pred_mask = predicted >= self.eval_threshold
    true_mask = labels > 0.5
    pred_bg = np.logical_not(pred_mask)
    true_bg = np.logical_not(true_mask)

    self.tp += np.sum(pred_mask & true_mask)
    self.fp += np.sum(pred_mask & true_bg)
    self.fn += np.sum(pred_bg & true_mask)
    self.tn += np.sum(pred_bg & true_bg)
    self.num_patches += 1

    predicted = expit(predicted)
    self.images_xy.append(self.slice_image(labels, predicted, weights, 0))
    self.images_xz.append(self.slice_image(labels, predicted, weights, 1))
    self.images_yz.append(self.slice_image(labels, predicted, weights, 2))

  def get_summaries(self):
    """Gathers tensorflow summaries into single list."""

    if not self.total_voxels:
      return []

    precision = self.tp / max(self.tp + self.fp, 1)
    recall = self.tp / max(self.tp + self.fn, 1)

    for images in self.images_xy, self.images_xz, self.images_yz:
      for i, summary in enumerate(images):
        summary.tag += '/%d' % i

    summaries = (
        list(self.images_xy) + list(self.images_xz) + list(self.images_yz) + [
            tf.Summary.Value(tag='masked_voxel_fraction',
                             simple_value=(self.masked_voxels /
                                           self.total_voxels)),
            tf.Summary.Value(tag='eval/patch_loss',
                             simple_value=self.loss / self.num_patches),
            tf.Summary.Value(tag='eval/patches',
                             simple_value=self.num_patches),
            tf.Summary.Value(tag='eval/accuracy',
                             simple_value=(self.tp + self.tn) / (
                                 self.tp + self.tn + self.fp + self.fn)),
            tf.Summary.Value(tag='eval/precision',
                             simple_value=precision),
            tf.Summary.Value(tag='eval/recall',
                             simple_value=recall),
            tf.Summary.Value(tag='eval/specificity',
                             simple_value=self.tn / max(self.tn + self.fp, 1)),
            tf.Summary.Value(tag='eval/f1',
                             simple_value=(2.0 * precision * recall /
                                           (precision + recall)))
        ])

    return summaries


def run_training_step(sess, model, fetch_summary, feed_dict):
  """Runs one training step for a single FFN FOV."""
  ops_to_run = [model.train_op, model.global_step, model.logits]

  if fetch_summary is not None:
    ops_to_run.append(fetch_summary)

  results = sess.run(ops_to_run, feed_dict)
  step, prediction = results[1:3]

  if fetch_summary is not None:
    summ = results[-1]
  else:
    summ = None

  return prediction, step, summ


def fov_moves():
  # Add one more move to get a better fill of the evaluation area.
  if FLAGS.fov_policy == 'max_pred_moves':
    return FLAGS.fov_moves + 1
  return FLAGS.fov_moves


def train_labels_size(model):
  return (np.array(model.pred_mask_size) +
          np.array(model.deltas) * 2 * fov_moves())


def train_eval_size(model):
  return (np.array(model.pred_mask_size) +
          np.array(model.deltas) * 2 * FLAGS.fov_moves)


def train_image_size(model):
  return (np.array(model.input_image_size) +
          np.array(model.deltas) * 2 * fov_moves())


def train_canvas_size(model):
  return (np.array(model.input_seed_size) +
          np.array(model.deltas) * 2 * fov_moves())


def _get_offset_and_scale_map():
  if not FLAGS.image_offset_scale_map:
    return {}

  ret = {}
  for vol_def in FLAGS.image_offset_scale_map:
    vol_name, offset, scale = vol_def.split(':')
    ret[vol_name] = float(offset), float(scale)

  return ret


def _get_reflectable_axes():
  return [int(x) + 1 for x in FLAGS.reflectable_axes]


def _get_permutable_axes():
  return [int(x) + 1 for x in FLAGS.permutable_axes]


def define_data_input(model, queue_batch=None):
  """Adds TF ops to load input data."""

  label_volume_map = {}
  for vol in FLAGS.label_volumes.split(','):
    volname, path, dataset = vol.split(':')
    label_volume_map[volname] = h5py.File(path)[dataset]

  image_volume_map = {}
  for vol in FLAGS.data_volumes.split(','):
    volname, path, dataset = vol.split(':')
    image_volume_map[volname] = h5py.File(path)[dataset]

  if queue_batch is None:
    queue_batch = FLAGS.batch_size

  # Fetch sizes of images and labels
  label_size = train_labels_size(model)
  image_size = train_image_size(model)

  label_radii = (label_size // 2).tolist()
  label_size = label_size.tolist()
  image_radii = (image_size // 2).tolist()
  image_size = image_size.tolist()

  # Fetch a single coordinate and volume name from a queue reading the
  # coordinate files or from saved hard/important examples
  coord, volname = inputs.load_patch_coordinates(FLAGS.train_coords)

  # Load object labels (segmentation).
  labels = inputs.load_from_numpylike(
      coord, volname, label_size, label_volume_map)

  label_shape = [1] + label_size[::-1] + [1]
  labels = tf.reshape(labels, label_shape)

  loss_weights = tf.constant(np.ones(label_shape, dtype=np.float32))

  # Load image data.
  patch = inputs.load_from_numpylike(
      coord, volname, image_size, image_volume_map)
  data_shape = [1] + image_size[::-1] + [1]
  patch = tf.reshape(patch, shape=data_shape)

  if ((FLAGS.image_stddev is None or FLAGS.image_mean is None) and
      not FLAGS.image_offset_scale_map):
    raise ValueError('--image_mean, --image_stddev or --image_offset_scale_map '
                     'need to be defined')

  # Convert segmentation into a soft object mask.
  lom = tf.logical_and(
      labels > 0,
      tf.equal(labels, labels[0,
                              label_radii[2],
                              label_radii[1],
                              label_radii[0],
                              0]))
  labels = inputs.soften_labels(lom)

  # Apply basic augmentations.
  transform_axes = augmentation.PermuteAndReflect(
      rank=5, permutable_axes=_get_permutable_axes(),
      reflectable_axes=_get_reflectable_axes())
  labels = transform_axes(labels)
  patch = transform_axes(patch)
  loss_weights = transform_axes(loss_weights)

  # Normalize image data.
  patch = inputs.offset_and_scale_patches(
      patch, volname[0],
      offset_scale_map=_get_offset_and_scale_map(),
      default_offset=FLAGS.image_mean,
      default_scale=FLAGS.image_stddev)

  # Create a batch of examples. Note that any TF operation before this line
  # will be hidden behind a queue, so expensive/slow ops can take advantage
  # of multithreading.
  patches, labels, loss_weights = tf.train.shuffle_batch(
      [patch, labels, loss_weights], queue_batch,
      num_threads=max(1, FLAGS.batch_size // 2),
      capacity=32 * FLAGS.batch_size,
      min_after_dequeue=4 * FLAGS.batch_size,
      enqueue_many=True)

  return patches, labels, loss_weights, coord, volname


def prepare_ffn(model):
  """Creates the TF graph for an FFN."""
  shape = [FLAGS.batch_size] + list(model.pred_mask_size[::-1]) + [1]

  model.labels = tf.placeholder(tf.float32, shape, name='labels')
  model.loss_weights = tf.placeholder(tf.float32, shape, name='loss_weights')
  model.define_tf_graph()


def fixed_offsets(model, seed, fov_shifts=None):
  """Generates offsets based on a fixed list."""
  for off in itertools.chain([(0, 0, 0)], fov_shifts):
    if model.dim == 3:
      is_valid_move = seed[:,
                           seed.shape[1] // 2 + off[2],
                           seed.shape[2] // 2 + off[1],
                           seed.shape[3] // 2 + off[0],
                           0] >= logit(FLAGS.threshold)
    else:
      is_valid_move = seed[:,
                           seed.shape[1] // 2 + off[1],
                           seed.shape[2] // 2 + off[0],
                           0] >= logit(FLAGS.threshold)

    if not is_valid_move:
      continue

    yield off


def max_pred_offsets(model, seed):
  """Generates offsets with the policy used for inference."""
  # Always start at the center.
  queue = deque([(0, 0, 0)])
  done = set()

  train_image_radius = train_image_size(model) // 2
  input_image_radius = np.array(model.input_image_size) // 2

  while queue:
    offset = queue.popleft()

    # Drop any offsets that would take us beyond the image fragment we
    # loaded for training.
    if np.any(np.abs(np.array(offset)) + input_image_radius >
              train_image_radius):
      continue

    # Ignore locations that were visited previously.
    quantized_offset = (
        offset[0] // max(model.deltas[0], 1),
        offset[1] // max(model.deltas[1], 1),
        offset[2] // max(model.deltas[2], 1))

    if quantized_offset in done:
      continue

    done.add(quantized_offset)

    yield offset

    # Look for new offsets within the updated seed.
    curr_seed = mask.crop_and_pad(seed, offset, model.pred_mask_size[::-1])
    todos = sorted(
        movement.get_scored_move_offsets(
            model.deltas[::-1],
            curr_seed[0, ..., 0],
            threshold=logit(FLAGS.threshold)), reverse=True)
    queue.extend((x[2] + offset[0],
                  x[1] + offset[1],
                  x[0] + offset[2]) for _, x in todos)


def get_example(load_example, eval_tracker, model, get_offsets):
  """Generates individual training examples.

  Args:
    load_example: callable returning a tuple of image and label ndarrays
                  as well as the seed coordinate and volume name of the example
    eval_tracker: EvalTracker object
    model: FFNModel object
    get_offsets: iterable of (x, y, z) offsets to investigate within the
        training patch

  Yields:
    tuple of:
      seed array, shape [1, z, y, x, 1]
      image array, shape [1, z, y, x, 1]
      label array, shape [1, z, y, x, 1]
  """
  seed_shape = train_canvas_size(model).tolist()[::-1]

  while True:
    full_patches, full_labels, loss_weights, coord, volname = load_example()
    # Always start with a clean seed.
    seed = logit(mask.make_seed(seed_shape, 1, pad=FLAGS.seed_pad))

    for off in get_offsets(model, seed):
      predicted = mask.crop_and_pad(seed, off, model.input_seed_size[::-1])
      patches = mask.crop_and_pad(full_patches, off, model.input_image_size[::-1])
      labels = mask.crop_and_pad(full_labels, off, model.pred_mask_size[::-1])
      weights = mask.crop_and_pad(loss_weights, off, model.pred_mask_size[::-1])

      # Necessary, since the caller is going to update the array and these
      # changes need to be visible in the following iterations.
      assert predicted.base is seed
      yield predicted, patches, labels, weights

    eval_tracker.add_patch(
        full_labels, seed, loss_weights, coord, volname, full_patches)


def get_batch(load_example, eval_tracker, model, batch_size, get_offsets):
  """Generates batches of training examples.

  Args:
    load_example: callable returning a tuple of image and label ndarrays
                  as well as the seed coordinate and volume name of the example
    eval_tracker: EvalTracker object
    model: FFNModel object
    batch_size: desidred batch size
    get_offsets: iterable of (x, y, z) offsets to investigate within the
        training patch

  Yields:
    tuple of:
      seed array, shape [b, z, y, x, 1]
      image array, shape [b, z, y, x, 1]
      label array, shape [b, z, y, x, 1]

    where 'b' is the batch_size.
  """
  def _batch(iterable):
    for batch_vals in iterable:
      # `batch_vals` is sequence of `batch_size` tuples returned by the
      # `get_example` generator, to which we apply the following transformation:
      #   [(a0, b0), (a1, b1), .. (an, bn)] -> [(a0, a1, .., an),
      #                                         (b0, b1, .., bn)]
      # (where n is the batch size) to get a sequence, each element of which
      # represents a batch of values of a given type (e.g., seed, image, etc.)
      yield zip(*batch_vals)

  # Create a separate generator for every element in the batch. This generator
  # will automatically advance to a different training example once the allowed
  # moves for the current location are exhausted.
  for seeds, patches, labels, weights in _batch(six.moves.zip(
      *[get_example(load_example, eval_tracker, model, get_offsets) for _
        in range(batch_size)])):

    batched_seeds = np.concatenate(seeds)

    yield (batched_seeds, np.concatenate(patches), np.concatenate(labels),
           np.concatenate(weights))

    # batched_seed is updated in place with new predictions by the code
    # calling get_batch. Here we distribute these updated predictions back
    # to the buffer of every generator.
    for i in range(batch_size):
      seeds[i][:] = batched_seeds[i, ...]


def save_flags():
  gfile.MakeDirs(FLAGS.train_dir)
  with gfile.Open(os.path.join(FLAGS.train_dir,
                               'flags.%d' % time.time()), 'w') as f:
    for mod, flag_list in FLAGS.flags_by_module_dict().items():
      if (mod.startswith('google3.research.neuromancer.tensorflow') or
          mod.startswith('/')):
        for flag in flag_list:
          f.write('%s\n' % flag.serialize())


def train_ffn(model_cls, **model_kwargs):
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):
      # The constructor might define TF ops/placeholders, so it is important
      # that the FFN is instantiated within the current context.
      model = model_cls(**model_kwargs)
      eval_shape_zyx = train_eval_size(model).tolist()[::-1]

      eval_tracker = EvalTracker(eval_shape_zyx)
      load_data_ops = define_data_input(model, queue_batch=1)
      prepare_ffn(model)
      merge_summaries_op = tf.summary.merge_all()

      if FLAGS.task == 0:
        save_flags()

      summary_writer = None
      saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
      scaffold = tf.train.Scaffold(saver=saver)
      with tf.train.MonitoredTrainingSession(
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          save_summaries_steps=None,
          save_checkpoint_secs=300,
          config=tf.ConfigProto(
              log_device_placement=False, allow_soft_placement=True),
          checkpoint_dir=FLAGS.train_dir,
          scaffold=scaffold) as sess:

        eval_tracker.sess = sess
        step = int(sess.run(model.global_step))

        if FLAGS.task > 0:
          # To avoid early instabilities when using multiple replicas, we use
          # a launch schedule where new replicas are brought online gradually.
          logging.info('Delaying replica start.')
          while step < FLAGS.replica_step_delay * FLAGS.task:
            time.sleep(5.0)
            step = int(sess.run(model.global_step))
        else:
          summary_writer = tf.summary.FileWriterCache.get(FLAGS.train_dir)
          summary_writer.add_session_log(
              tf.summary.SessionLog(status=tf.summary.SessionLog.START), step)

        fov_shifts = list(model.shifts)  # x, y, z
        if FLAGS.shuffle_moves:
          random.shuffle(fov_shifts)

        policy_map = {
            'fixed': partial(fixed_offsets, fov_shifts=fov_shifts),
            'max_pred_moves': max_pred_offsets
        }
        batch_it = get_batch(lambda: sess.run(load_data_ops),
                             eval_tracker, model, FLAGS.batch_size,
                             policy_map[FLAGS.fov_policy])

        t_last = time.time()

        while not sess.should_stop() and step < FLAGS.max_steps:
          # Run summaries periodically.
          t_curr = time.time()
          if t_curr - t_last > FLAGS.summary_rate_secs and FLAGS.task == 0:
            summ_op = merge_summaries_op
            t_last = t_curr
          else:
            summ_op = None

          seed, patches, labels, weights = next(batch_it)

          updated_seed, step, summ = run_training_step(
              sess, model, summ_op,
              feed_dict={
                  model.loss_weights: weights,
                  model.labels: labels,
                  model.input_patches: patches,
                  model.input_seed: seed,
              })

          # Save prediction results in the original seed array so that
          # they can be used in subsequent steps.
          mask.update_at(seed, (0, 0, 0), updated_seed)

          # Record summaries.
          if summ is not None:
            logging.info('Saving summaries.')
            summ = tf.Summary.FromString(summ)

            # Compute a loss over the whole training patch (i.e. more than a
            # single-step field of view of the network). This quantifies the
            # quality of the final object mask.
            summ.value.extend(eval_tracker.get_summaries())
            eval_tracker.reset()

            assert summary_writer is not None
            summary_writer.add_summary(summ, step)

      if summary_writer is not None:
        summary_writer.flush()


def main(argv=()):
  del argv  # Unused.
  model_class = import_symbol(FLAGS.model_name)
  # Multiply the task number by a value large enough that tasks starting at a
  # similar time cannot end up with the same seed.
  seed = int(time.time() + FLAGS.task * 3600 * 24)
  logging.info('Random seed: %r', seed)
  random.seed(seed)

  train_ffn(model_class, batch_size=FLAGS.batch_size,
            **json.loads(FLAGS.model_args))


if __name__ == '__main__':
  flags.mark_flag_as_required('train_coords')
  flags.mark_flag_as_required('data_volumes')
  flags.mark_flag_as_required('label_volumes')
  flags.mark_flag_as_required('model_name')
  flags.mark_flag_as_required('model_args')
  app.run(main)
