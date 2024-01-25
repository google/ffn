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

from functools import partial
import json
import logging
import os
import random
import time
from typing import Optional

from absl import app
from absl import flags
from ffn.training import augmentation
from ffn.training import examples
from ffn.training import inputs
from ffn.training import model as ffn_model
# Necessary so that optimizer flags are defined.
from ffn.training import optimizer  # pylint: disable=unused-import
from ffn.training import tracker
from ffn.training.import_util import import_symbol
import h5py
import numpy as np
from scipy import special
import tensorflow.compat.v1 as tf
from tensorflow.io import gfile


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


def run_training_step(sess: tf.Session, model: ffn_model.FFNModel,
                      fetch_summary: Optional[tf.Operation],
                      feed_dict: dict[str, np.ndarray]):
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


def fov_moves() -> int:
  # Add one more move to get a better fill of the evaluation area.
  if FLAGS.fov_policy == 'max_pred_moves':
    return FLAGS.fov_moves + 1
  return FLAGS.fov_moves


def train_labels_size(info: ffn_model.ModelInfo) -> np.ndarray:
  return (np.array(info.pred_mask_size) +
          np.array(info.deltas) * 2 * fov_moves())


def train_eval_size(info: ffn_model.ModelInfo) -> np.ndarray:
  return (np.array(info.pred_mask_size) +
          np.array(info.deltas) * 2 * FLAGS.fov_moves)


def train_image_size(info: ffn_model.ModelInfo) -> np.ndarray:
  return (np.array(info.input_image_size) +
          np.array(info.deltas) * 2 * fov_moves())


def train_canvas_size(info: ffn_model.ModelInfo) -> np.ndarray:
  return (np.array(info.input_seed_size) +
          np.array(info.deltas) * 2 * fov_moves())


def _get_offset_and_scale_map() -> dict[str, tuple[float, float]]:
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


def prepare_ffn(model: ffn_model.FFNModel):
  """Creates the TF graph for an FFN."""
  shape = [FLAGS.batch_size] + list(model.info.pred_mask_size[::-1]) + [1]

  model.labels = tf.placeholder(tf.float32, shape, name='labels')
  model.loss_weights = tf.placeholder(tf.float32, shape, name='loss_weights')
  model.define_tf_graph()


def save_flags():
  gfile.makedirs(FLAGS.train_dir)
  with gfile.GFile(
      os.path.join(FLAGS.train_dir, 'flags.%d' % time.time()), 'w') as f:
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
      eval_shape_zyx = train_eval_size(model.info).tolist()[::-1]

      eval_tracker = tracker.EvalTracker(eval_shape_zyx, model.shifts)
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

        train_image_radius = train_image_size(model.info) // 2
        input_image_radius = np.array(model.info.input_image_size) // 2
        policy_map = {
            'fixed':
                partial(
                    examples.fixed_offsets,
                    fov_shifts=fov_shifts,
                    threshold=special.logit(FLAGS.threshold)),
            'max_pred_moves':
                partial(
                    examples.max_pred_offsets,
                    max_radius=train_image_radius - input_image_radius,
                    threshold=special.logit(FLAGS.threshold)),
            'no_step':
                examples.no_offsets
        }
        policy_fn = policy_map[FLAGS.fov_policy]

        def _make_ffn_example():
          return examples.get_example(
              lambda: sess.run(load_data_ops),
              eval_tracker,
              model.info,
              policy_fn,
              FLAGS.seed_pad,
              seed_shape=tuple(train_canvas_size(model.info).tolist()[::-1]))

        batch_it = examples.BatchExampleIter(_make_ffn_example, eval_tracker,
                                             FLAGS.batch_size, model.info)

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

          eval_tracker.to_tf()
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
          batch_it.update_seeds(updated_seed)

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
