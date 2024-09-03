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

"""Input pipeline for FFN models."""

from concurrent import futures
import functools
import threading
from typing import Any, Callable

from absl import logging
from connectomics.common import utils
from ffn.input import volume
from ffn.jax import tracker
from ffn.training import examples
from ffn.training import inputs
from ffn.training import model as ffn_model
import jax
import jmp
import ml_collections
import numpy as np
import tensorflow as tf


# TODO(mjanusz): Check if this is still required.
tf.config.threading.set_inter_op_parallelism_threads(128)

Dataset = tf.data.Dataset


def _check_thread_success(future: futures.Future[Any], err_msg: str = ''):
  e = future.exception()

  if e is None:
    return

  logging.error(err_msg)
  raise e


def load_examples(
    config: ml_collections.ConfigDict,
    rng: jax.Array,
    load_shape: volume.TupleXYZ,
) -> tuple[grain.TfMixtureDataLoader | tf.data.Dataset, int]:
  """Loads a single training example."""

  if config.get('train_coords'):
    train_coords = config.train_coords
    # ConfigDict keys cannot contain dots, so we allow specifying the dictionary
    # as an iterable of key-value pairs.
    if isinstance(train_coords, tuple) or isinstance(train_coords, list):
      train_coords = {k: v for k, v in train_coords}

    sampling = volume.SamplingConfig(vsi_coords=train_coords)
  else:
    train_coords = config.arrayrec_coords
    if isinstance(train_coords, tuple) or isinstance(train_coords, list):
      train_coords = {k: v for k, v in train_coords}

    sampling = volume.SamplingConfig(arrayrecord_coords=train_coords)
  # The shape of the loaded data has to be symmetric so that if we apply
  # random reflections, the center point remains in the center.
  effective_load_shape = tuple((np.array(load_shape) // 2 * 2 + 1).tolist())

  cfg = volume.InputConfig(
      sampling=sampling,
      volumes={
          'em': volume.VolumeConfig(
              config.em_volumes,
              load_shape=effective_load_shape,
              filter_shape=effective_load_shape,
          ),
          # Note that it is actually valid for the patch to extend beyond the
          # bounding box of the label volume, so we only check that the center
          # voxel is within the labeled area.
          'seg': volume.VolumeConfig(
              config.seg_volumes,
              load_shape=effective_load_shape,
              filter_shape=(1, 1, 1),
          ),
          'oob': volume.VolumeConfig(
              config.seg_volumes, load_shape=effective_load_shape, oob_mask=True
          ),
      },
      # TODO(mjanusz): Add support for rotation and simulated missing sections.
      augmentation=volume.AugmentationConfig(
          permutable_axes=config.permutable_axes,
          reflectable_axes=config.reflectable_axes,
          contrast_factor_range=config.contrast_factor_range,
          brightness_factor_range=config.brightness_factor_range,
          apply_adjustment_to=config.apply_adjustment_to,
      ),
  )

  if config.loss_mask_volumes:
    cfg.volumes['loss_mask'] = volume.VolumeConfig(
        config.loss_mask_volumes,
        load_shape=effective_load_shape,
        default_value=1 if config.loss_mask_invert else 0,
    )

    if hasattr(config, 'loss_mass_relabel'):
      cfg.volumes['loss_mask'].relabel_maps = config.loss_mask_relabel

  def _add_ffn_data(ex: volume.Example) -> volume.Example:
    weights = tf.cast(ex['oob'], tf.float32)

    if config.loss_mask_volumes:
      if config.loss_mask_invert:
        loss_mask = tf.equal(ex['loss_mask'], 0)
      else:
        loss_mask = ex['loss_mask'] > 0

      weights *= 1.0 - tf.cast(loss_mask, tf.float32)

    seg = ex['seg']
    center_val = seg[
        0,  #
        seg.shape[1] // 2,  #
        seg.shape[2] // 2,  #
        seg.shape[3] // 2,  #
        0,
    ]
    lom = tf.logical_and(seg > 0, tf.equal(seg, center_val))
    labels = inputs.soften_labels(lom)

    lx, ly, lz = load_shape
    emt = tf.cast(ex['em'][:, :lz, :ly, :lx, :], tf.float32)
    if config.image_clip_value_max > 0.0:
      emt = tf.clip_by_value(emt, 0.0, config.image_clip_value_max)

    return dict(
        ex,
        weights=weights[:, :lz, :ly, :lx, :],
        labels=labels[:, :lz, :ly, :lx, :],
        patches=(emt - config.image_mean) / config.image_stddev,
    )

  batch_size = config.per_device_batch_size * jax.local_device_count()

  if cfg.sampling.vsi_coords:
    num_examples = getattr(config, 'train_num_coords', 100_000_000)
    ds = volume.load_and_augment_subvolumes(cfg, int(np.array(rng)[0]))
    ds = ds.map(_add_ffn_data)

    options = tf.data.Options()
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
    options.threading.private_threadpool_size = 256
    options.threading.max_intra_op_parallelism = 1
    # Temporary workaround. See b/179292577.
    options.experimental_external_state_policy = (
        tf.data.experimental.ExternalStatePolicy.WARN
    )
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(config.tf_data_prefetch_size)
  else:
    ds, num_examples = volume.grain_load_and_augment_subvolumes(
        cfg, np.array(rng), _add_ffn_data, batch_size
    )
  return ds, num_examples


def create_dataset(
    config: ml_collections.ConfigDict,
    seed: jax.Array,
    load_shape: volume.TupleXYZ,
    data_service_address: str | None = None,
) -> tuple[Dataset, int]:
  """Creates a dataset for training.

  Args:
    config: Configuration to use.
    seed: PRNGKey for seeding operations in the training dataset.
    load_shape: XYZ shape of the data patch to load from volumestore.
    data_service_address: Unsupported.

  Returns:
    Training dataset and the total number of examples.
  """
  if data_service_address is not None:
    raise NotImplementedError(
        'Support for tf.data service not implemented yet.'
    )

  ds, num_total_examples = load_examples(config, seed, load_shape)
  return ds, num_total_examples


class BatchDictExampleIter(examples.BatchExampleIter):
  """Replaces tuples with dicts."""

  def __next__(self):
    seeds, patches, labels, weights = super().__next__()
    return {'seed': seeds, 'label': labels, 'patch': patches, 'weight': weights}


class MixingBatchExampleIter(BatchDictExampleIter):
  """Like BatchDictExampleIter but with more examples in parallel.

  The total number of examples held in memory at a time is given by
  num_batches * batch_size. A full batch is randomly selected out
  of these examples at every training stap. This reduces correlations
  between training batches in consecutive steps, and makes it possible
  to prefetch data in the background.
  """

  # pylint: disable=super-init-not-called
  def __init__(
      self,
      example_generator_fn: Callable[[], examples.ExampleGenerator],
      eval_tracker: tracker.EvalTracker,
      batch_size: int,
      num_batches: int,
      model_info: ffn_model.ModelInfo,
      batch_prefetch: int = 16,
      jmp_policy: jmp.Policy | None = None,
  ):
    """Constructor.

    Args:
      example_generator_fn: function returning a generator of single training
        examples
      eval_tracker: FFN eval tracker object
      batch_size: number of examples per batch
      num_batches: number of batches to hold in memory
      model_info: FFN model info
      batch_prefetch: number of batches to prefetch
      jmp_policy: Optional Jax policy for mixed precision training
    """
    assert num_batches > 1
    self._eval_tracker = eval_tracker
    self._seeds: list[np.ndarray] = []
    # List of indices of self._generators that generated the current
    # batch.
    self._current_idx: list[int] = []
    self._batch_size = batch_size
    self._info = model_info
    self._jmp_policy = jmp_policy

    # Loading of individual training examples.
    self._generators = [
        example_generator_fn() for _ in range(batch_size * num_batches)
    ]
    self._tpe = futures.ThreadPoolExecutor(
        max_workers=batch_size * batch_prefetch
    )
    self._fs_lock = threading.Lock()
    self._fs = set()
    for i, gen in enumerate(self._generators):
      self._fs.add(self._tpe.submit(lambda gen=gen, i=i: (i, next(gen))))

    # Prefetching of complete batches.
    self._batch_tpe = futures.ThreadPoolExecutor(max_workers=batch_prefetch)
    self._batch_fs = set()
    for i in range(batch_prefetch):
      self._batch_fs.add(self._batch_tpe.submit(self._generate_batch))

    self._seed_update_tpe = futures.ThreadPoolExecutor(max_workers=4)

  def _generate_batch(self):
    """Returns a batch of training examples."""
    seeds, patches, labels, weights, batch_ids = [], [], [], [], []

    while len(batch_ids) < self._batch_size:
      with self._fs_lock:
        for f in futures.as_completed(self._fs):
          self._fs.remove(f)
          i, (seed, patch, label, weight) = f.result()
          seeds.append(seed)
          patches.append(patch)
          labels.append(label)
          weights.append(weight)
          batch_ids.append(i)

          if len(batch_ids) == self._batch_size:
            break

    batched_seeds = np.concatenate(seeds)
    batched_weights = np.concatenate(weights)
    batched_labels = np.concatenate(labels)
    batched_patches = np.concatenate(patches)

    if self._jmp_policy is not None:
      batched_patches = self._jmp_policy.cast_to_compute(batched_patches)
      batched_seeds = self._jmp_policy.cast_to_compute(batched_seeds)

    return (
        batch_ids,
        seeds,
        batched_seeds,
        batched_weights,
        batched_labels,
        batched_patches,
    )

  def __next__(self):
    # The time reported here indicates how long the training script had to
    # wait to get a new batch of examples. It should ideally be ~0.
    with utils.report_time('MixingBatchExampleIter'):
      f = next(futures.as_completed(self._batch_fs))

    self._batch_fs.remove(f)
    self._batch_fs.add(self._batch_tpe.submit(self._generate_batch))

    (
        self._current_idx,
        self._seeds,
        batched_seeds,
        batched_weights,
        batched_labels,
        batched_patches,
    ) = f.result()
    self._eval_tracker.track_weights(batched_weights)
    return {
        'seed': batched_seeds,
        'label': batched_labels,
        'patch': batched_patches,
        'weight': batched_weights,
    }

  def update_seeds(self, batched_seeds: np.ndarray | jax.Array):
    """Propagates data from `batched_seeds` back to the example generators."""

    def _update(
        seeds: list[np.ndarray],
        batched_seeds: np.ndarray | jax.Array,
        current: list[int],
    ):
      # Transfer data from device to host if using a JAX array.
      batched_seeds = np.array(batched_seeds)
      # Fold batch dimensions back to a single one.
      batched_seeds = np.reshape(
          batched_seeds, [-1] + list(batched_seeds.shape[-4:])
      )

      dx = self._info.input_seed_size[0] - self._info.pred_mask_size[0]
      dy = self._info.input_seed_size[1] - self._info.pred_mask_size[1]
      dz = self._info.input_seed_size[2] - self._info.pred_mask_size[2]

      for i, _ in enumerate(current):
        if dz == 0 and dy == 0 and dx == 0:
          seeds[i][:] = batched_seeds[i, ...]
        else:
          seeds[i][
              :,  #
              dz // 2 : -(dz - dz // 2),  #
              dy // 2 : -(dy - dy // 2),  #
              dx // 2 : -(dx - dx // 2),  #
              :,
          ] = batched_seeds[i, ...]

      with self._fs_lock:
        for gen_idx in current:
          gen = self._generators[gen_idx]
          self._fs.add(
              self._tpe.submit(lambda gen=gen, i=gen_idx: (i, next(gen)))
          )

    # Distribute data asynchronously.
    update_future = self._seed_update_tpe.submit(
        _update, self._seeds, batched_seeds, self._current_idx
    )
    update_future.add_done_callback(
        functools.partial(
            _check_thread_success, err_msg='Error while updating seeds.'
        )
    )


class UnbatchIter:
  """Fetches batches from a tf.data iterator and returns elements one by one.

  Iterating over tf.data appears to incur some overhead, so it's faster to
  pull complete batches and unpack them here.

  The input arrays expected to be shaped [b, z, y, x, c], while the output
  arrays are going to have shape [z, y, x, c].
  """

  def __init__(self, batch_iter: tf.data.Iterator):
    self._batch_iter = batch_iter
    self._batch = None
    self._idx = 0
    self._lock = threading.Lock()

  def __iter__(self):
    return self

  def __next__(self):
    with self._lock:
      if self._batch is None:
        self._idx = 0

        # The time reported here will reflect delays caused by the tf.data
        # pipeline. This does not impact training speed as long as
        # MixingBatchExampleIter time is close to 0.
        with utils.report_time('tf_data_input'):
          ex = next(self._batch_iter)

        # Convert from EagerTensor to numpy.
        self._batch = (
            np.array(ex['patches']),
            np.array(ex['labels']),
            np.array(ex['weights']),
            np.array(ex['coord']),
            np.array(ex['volname']),
        )

      ret = [x[self._idx] for x in self._batch]

      self._idx += 1
      if self._batch[0].shape[0] == self._idx:
        self._batch = None

      return ret


def get_batch_iter(
    data_iter: tf.data.Iterator,
    eval_tracker: tracker.EvalTracker,
    policy_fn: examples.GetOffsets,
    model_info: ffn_model.ModelInfo,
    config: ml_collections.ConfigDict,  #
    seed_shape: tuple[int, int, int],
    batch_size: int,
    jmp_policy: jmp.Policy | None = None,
) -> BatchDictExampleIter:
  """Creates an iterator over batches of training examples."""

  # Pull single examples from the TF DS iterator.
  unbatched_iter = UnbatchIter(data_iter)

  def _load_example():
    return next(unbatched_iter)

  # Pull training examples from the source (_load_example) and generate
  # FFN training examples (when FOV movements are made, there will be more
  # than 1 training example per data item loaded from the input).
  def _make_example():
    return examples.get_example(
        _load_example,
        eval_tracker,
        model_info,
        policy_fn,
        config.seed_pad,
        seed_shape=seed_shape,
    )

  # Instantiate multiple generators (_make_examples), and batch their outputs
  # as they become available.
  if config.mix_num_batches == 1:
    return BatchDictExampleIter(
        _make_example, eval_tracker, batch_size, model_info
    )
  else:
    return MixingBatchExampleIter(
        _make_example,
        eval_tracker,
        batch_size,
        config.mix_num_batches,
        model_info,
        config.host_num_batch_prefetch,
        jmp_policy=jmp_policy,
    )
