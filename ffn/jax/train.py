# Copyright 2024 Google Inc.
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

"""Training script for FFN models."""

import collections
import functools as ft
import os
import random
import time
from typing import Any, Sequence, TypeVar

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from connectomics.jax import training
from etils import epath
from ffn.jax import input_pipeline
from ffn.jax import tracker
from ffn.training import examples
from ffn.training import model as ffn_model
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import jmp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
from scipy import special
from t5x.checkpoints import DatasetArgs
from t5x.checkpoints import DatasetCheckpointHandler
import tensorflow as tf

from connectomics.jax.models import util as model_util


class TrainState(flax.struct.PyTreeNode):  # pytype: disable=invalid-function-definition  # dataclass_transform
  step: int
  opt_state: optax.OptState
  params: flax.core.FrozenDict[str, Any]
  batch_stats: Any
  ema_params: Any = None


DataIterator = TypeVar(
    'DataIterator',
    tf.data.Iterator,
)


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: jax.Array,
    input_shape: Sequence[int],
) -> tuple[nn.Module, optax.Schedule, optax.GradientTransformation, TrainState]:
  """Instantiates and initializes the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.

  Returns:
    The initialized TrainState with the optimizer.
  """
  model = model_util.model_from_config(config)
  rng = {'params': rng, 'dropout': jax.random.PRNGKey(1)}
  variables = model.init(rng, jnp.ones(input_shape))
  params = variables['params']

  parameter_overview.log_parameter_overview(params)
  tx, lr = training.get_optimizer(config)

  return (
      model,
      lr,
      tx,
      TrainState(
          step=0,
          opt_state=tx.init(params),
          batch_stats=variables.get('batch_stats', None),
          params=params,
          ema_params=params if config.get('ema_decay', 0.0) > 0.0 else None,
      ),
  )


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')
  loss_std: metrics.Std.from_output('loss')
  learning_rate: metrics.LastValue.from_output('learning_rate')


def _updated_seed(seed: jnp.ndarray, update: jnp.ndarray) -> jnp.ndarray:
  """Applies the additive `update` to `seed`."""
  dz = seed.shape[-4] - update.shape[-4]
  dy = seed.shape[-3] - update.shape[-3]
  dx = seed.shape[-2] - update.shape[-2]

  logging.log_first_n(
      logging.INFO, 'Updating seed: %r with: %r', 1, seed.shape, update.shape
  )

  if dz == 0 and dy == 0 and dx == 0:
    return seed + update
  else:
    raise ValueError(
        'Currently only models with same input and output shapes are supported.'
    )

  return seed + jnp.pad(
      update,
      [
          [0, 0],  #
          [dz // 2, dz - dz // 2],  #
          [dy // 2, dy - dy // 2],
          [dx // 2, dx - dx // 2],
          [0, 0],
      ],
  )


def train_step(
    model: nn.Module,
    state: TrainState,
    schedule: optax.Schedule,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jax.Array],  #
    config: ml_collections.ConfigDict,
    dropout_rng: jax.Array,
    jmp_policy: jmp.Policy | None = None,
    loss_scale: jmp.LossScale = jmp.NoOpLossScale(),
) -> tuple[TrainState, metrics.Collection, jax.Array, jmp.LossScale | None]:
  """Performs a single training step.

  Args:
    model: Module to compute predictions.
    state: Current training state. Updated training state will be returned.
    schedule: optax learning rate schedule.
    optimizer: optax optimizer.
    batch: Training inputs for this step.
    config: Configuration for model.
    dropout_rng: RNG key for dropout.
    jmp_policy: Jax policy for mixed precision training.
    loss_scale: Loss scaling policy.

  Returns:
    tuple of: updated state, dictionary with metrics, updated part of the
    seed, updated loss scaling policy.
  """
  step = state.step + 1
  dropout_rng = jax.random.fold_in(dropout_rng, step)

  def loss_fn(params):
    variables = {'params': params}
    if state.batch_stats is not None:
      variables['batch_stats'] = state.batch_stats

    data = jnp.concatenate((batch['patch'], batch['seed']), axis=-1)
    kwargs = {}
    if 'transformer' in config.model_class or 'mixer' in config.model_class:
      kwargs['train'] = True

    logits, new_variables = model.apply(
        variables, data, mutable=True, rngs={'dropout': dropout_rng}, **kwargs
    )

    if config.additive_seed_update:
      logits = _updated_seed(batch['seed'], logits)

    loss = optax.sigmoid_binary_cross_entropy(logits, batch['label'])
    # NOTE: When using float16s, overflows occur so we anyways use float32 here.
    loss = jnp.mean(loss * batch['weight'], dtype=jnp.float32)

    loss = loss_scale.scale(loss)
    return loss, (new_variables.get('batch_stats', None), logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  params_copy = state.params
  if jmp_policy is not None:
    params_copy = jmp_policy.cast_to_compute(params_copy)

  (loss, (new_batch_stats, logits)), grad = grad_fn(params_copy)

  # Compute average gradient across multiple workers.
  if jmp_policy is not None:
    grad = jmp_policy.cast_to_param(grad)

  grad = loss_scale.unscale(grad)
  updates, new_opt_state = optimizer.update(grad, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)

  # Dynamic loss scaling needs adjustment in order to actually be dynamic.
  if config.skip_nonfinite_updates or config.dynamic_loss_scale:
    grads_finite = jmp.all_finite(grad)
    loss_scale = loss_scale.adjust(grads_finite)

    new_params, new_opt_state = jmp.select_tree(
        grads_finite,
        (new_params, new_opt_state),
        (state.params, state.opt_state),
    )

  ema_decay = config.get('ema_decay', 0.0)
  new_ema_params = state.ema_params
  if ema_decay > 0.0:
    if new_ema_params is None:
      new_ema_params = new_params
    else:
      decay = jnp.array(ema_decay, dtype=jnp.float32)
      decay = jnp.where(state.step == 0, 0.0, decay)
      new_ema_params = optax.incremental_update(
          new_params, new_ema_params, step_size=1.0 - decay
      )

  new_state = state.replace(  # pytype: disable=attribute-error
      step=step,
      params=new_params,
      opt_state=new_opt_state,
      batch_stats=new_batch_stats,
      ema_params=new_ema_params,
  )

  lr = schedule(state.opt_state.count)  # pytype: disable=attribute-error
  metrics_update = TrainMetrics.single_from_model_output(
      loss=loss, learning_rate=lr
  )

  return new_state, metrics_update, logits, loss_scale


def fov_moves(config: ml_collections.ConfigDict) -> int:
  if config.fov_policy == 'max_pred_moves':
    # Add one more move to get a better fill of the evaluation area.
    return config.fov_moves + 1
  else:
    return config.fov_moves


def train_image_size(
    info: ffn_model.ModelInfo, config: ml_collections.ConfigDict
) -> np.ndarray:
  return np.array(info.input_image_size) + np.array(
      info.deltas
  ) * 2 * fov_moves(config)


def train_canvas_size(
    info: ffn_model.ModelInfo, config: ml_collections.ConfigDict
) -> np.ndarray:
  return np.array(info.input_seed_size) + np.array(info.deltas) * 2 * fov_moves(
      config
  )


def train_eval_size(
    info: ffn_model.ModelInfo, config: ml_collections.ConfigDict
) -> np.ndarray:
  return np.array(info.pred_mask_size) + np.array(info.deltas) * 2 * fov_moves(
      config
  )


def build_shifts(
    config: ml_collections.ConfigDict,
) -> list[tuple[int, int, int]]:
  """Builds a sequence of FOV shifts for the network."""
  shifts = []
  d = config.deltas
  m = config.fov_moves
  for dx in range(-m * d[0], m * d[0] + 1, max(d[0], 1)):
    for dy in range(-m * d[1], m * d[1] + 1, max(d[1], 1)):
      for dz in range(-m * d[2], m * d[2] + 1, max(d[2], 1)):
        if dx == 0 and dy == 0 and dz == 0:
          continue
        shifts.append((dx, dy, dz))

  if config.shuffle_fov_moves:
    move_by_r = collections.defaultdict(list)
    for x, y, z in shifts:
      r = abs(x) + abs(y) + abs(z)
      move_by_r[r].append((x, y, z))

    # For multi-step moves, it is important to ensure that the
    # locations closer to the center of the seed are covered
    # before more distant ones..
    shifts = []
    for r, moves in sorted(move_by_r.items()):
      random.shuffle(moves)
      shifts.extend(moves)

  return shifts


def get_policy(
    fov_shifts: list[tuple[int, int, int]],
    info: ffn_model.ModelInfo,
    config: ml_collections.ConfigDict,
) -> examples.GetOffsets:
  """Returns a FOV movement policy function."""
  train_image_radius = train_image_size(info, config) // 2
  input_image_radius = np.array(info.input_image_size) // 2
  policy_map = {
      'fixed': ft.partial(
          examples.fixed_offsets,
          fov_shifts=fov_shifts,
          threshold=special.logit(config.threshold),
      ),
      'fixed_window': ft.partial(
          examples.fixed_offsets_window,
          fov_shifts=fov_shifts,
          threshold=special.logit(config.threshold),
          radius=8,
      ),
      'max_pred_moves': ft.partial(
          examples.max_pred_offsets,
          max_radius=train_image_radius - input_image_radius,
          threshold=special.logit(config.threshold),
      ),
      'no_step': examples.no_offsets,
  }
  return policy_map[config.fov_policy]


def _get_tf_writer(writers) -> metric_writers.SummaryWriter | None:
  # pylint:disable=protected-access
  for writer in writers:
    assert isinstance(writer, metric_writers.AsyncWriter)
    if isinstance(writer._writer, metric_writers.SummaryWriter):
      return writer._writer
  # pylint:enable=protected-access


def _get_ocp_args(
    train_iter: DataIterator, restore: bool = True
) -> DataIterator | ocp.args.CheckpointArgs:
  if isinstance(train_iter, tf.data.Iterator):
    return DatasetArgs(train_iter)


def _make_ckpt_args(state, train_iter: DataIterator) -> ocp.args.CheckpointArgs:
  return ocp.args.Composite(
      train_state=ocp.args.StandardSave(state),
      train_iter=_get_ocp_args(train_iter, restore=False),
  )


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str,
    data_service_address: str | None = None,
):
  """Main training loop."""
  workdir = epath.Path(workdir)
  workdir.mkdir(parents=True, exist_ok=True)

  rng = training.get_rng(config.seed)

  info = ffn_model.ModelInfo(
      deltas=config.deltas,
      pred_mask_size=config.fov_size,
      input_seed_size=config.fov_size,
      input_image_size=config.fov_size,
  )

  # Set up FFN FOV movement.
  fov_shifts = build_shifts(config)
  policy_fn = get_policy(fov_shifts, info, config)

  # Build input pipeline.
  rng, data_rng = jax.random.split(rng)
  data_seed = int(
      jax.random.randint(data_rng, [], minval=0, maxval=np.iinfo(np.int32).max)
  )
  random.seed(data_seed)

  train_ds, num_total_examples = input_pipeline.create_dataset(
      config,
      data_rng,
      load_shape=tuple(train_image_size(info, config)),
      data_service_address=data_service_address,
  )
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  logging.info('train_elem_shape=%r', train_iter.element_spec['em'].shape)  # pytype:disable=attribute-error

  # batch, z, y, x, (image, seed)
  input_shape = [1] + np.array(info.input_image_size).tolist()[::-1] + [2]

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, schedule, optimizer, state = create_train_state(
      config, model_rng, input_shape=input_shape
  )
  rng, dropout_rng = jax.random.split(rng)

  item_handlers = {}
  if isinstance(train_iter, tf.data.Iterator):
    item_handlers = {'train_iter': DatasetCheckpointHandler('ckpt', True)}

  # Checkpointing init.
  checkpoint_dir = epath.Path(workdir) / 'checkpoints'
  checkpoint_manager = ocp.CheckpointManager(
      checkpoint_dir,
      item_names=('train_state', 'train_iter'),
      item_handlers=item_handlers,
      options=ocp.CheckpointManagerOptions(
          save_interval_steps=config.checkpoint_every_steps
      ),
  )
  checkpointed_state = {'train_state': state, 'train_iter': train_iter}
  latest_step = checkpoint_manager.latest_step()
  # If an initial checkpoint is provided and the checkpointing library does not
  # report a 'latest' checkpoint, then we are starting a new experiment.
  # Otherwise an existing experiment is being resumed (e.g. after the training
  # task being preempted) and the latest checkpoint should take precedence.
  if config.init_from_cpoint and latest_step is None:
    handler = ocp.StandardCheckpointHandler()
    train_state_path = epath.Path(config.init_from_cpoint) / 'train_state'
    train_iter_path = epath.Path(config.init_from_cpoint) / 'train_iter'

    if isinstance(train_iter, tf.data.Iterator):
      iter_handler = item_handlers['train_iter']
      args = DatasetArgs(train_iter)

    checkpointed_state['train_state'] = handler.restore(
        train_state_path, args=ocp.args.StandardRestore(state)
    )
    checkpointed_state['train_iter'] = iter_handler.restore(
        train_iter_path, args=args
    )
    logging.info('Initializing training from %r', config.init_from_cpoint)
  elif latest_step is not None:
    checkpointed_state = checkpoint_manager.restore(
        latest_step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardRestore(state),
            train_iter=_get_ocp_args(train_iter),
        ),
    )
    logging.info('Restored checkpoint for step %d', latest_step)

  if latest_step is None:
    logging.info('Starting training from scratch.')
    # Save input config to CNS in addition to XM.
    if jax.process_index() == 0:
      with tf.io.gfile.GFile(
          tf.io.gfile.join(workdir, 'config.json'), 'w'
      ) as f:
        f.write(config.to_json_best_effort() + '\n')

  # Data partitioning, if recovered from checkpoint, can be incompatible
  # with the current setup. Avoid the problem by moving the state to the
  # host.
  state = jax.tree.map(np.array, checkpointed_state['train_state'])
  train_iter = checkpointed_state['train_iter']
  initial_step = int(state.step) + 1

  global_batch_size = config.per_device_batch_size * jax.device_count()
  host_batch_size = config.per_device_batch_size * jax.local_device_count()

  # Upper bound. The real number will be lower as not all steps are
  # taken for every example.
  steps_per_epoch = (
      num_total_examples // global_batch_size * (len(fov_shifts) + 1)
  )
  num_train_steps = steps_per_epoch * config.num_epochs
  logging.info(
      'num_train_steps=%d, steps_per_epoch=%d', num_train_steps, steps_per_epoch
  )

  # Mixed precision settings.
  jmp_policy = jmp.get_policy(config.mp_policy) if config.mp_policy else None
  loss_scale = jmp.NoOpLossScale()
  if config.loss_scale > 0:
    if config.dynamic_loss_scale:
      loss_scale = flax_utils.replicate(
          jmp.DynamicLossScale(jnp.asarray(float(config.loss_scale)))
      )
    else:
      loss_scale = jmp.StaticLossScale(config.loss_scale)

  # Shard batch across devices.
  mesh = Mesh(np.array(jax.devices()), ('batch',))
  batch_sharding = NamedSharding(mesh, P('batch'))
  replicate_sharding = NamedSharding(mesh, P())
  logging.info('Device mesh: %r', mesh)

  def train_fn(state, batch, loss_scale):
    return train_step(
        model=model,
        config=config,
        schedule=schedule,
        optimizer=optimizer,
        jmp_policy=jmp_policy,
        loss_scale=loss_scale,
        dropout_rng=dropout_rng,
        batch=batch,
        state=state,
    )

  shard_in = (
      replicate_sharding,  # state
      batch_sharding,  # data
      replicate_sharding,  # loss scale
  )
  shard_out = (
      replicate_sharding,  # state
      replicate_sharding,  # metrics
      batch_sharding,  # logits
      replicate_sharding,  # loss scale
  )
  p_train_step = jax.jit(train_fn, in_shardings=shard_in,
                         out_shardings=shard_out)

  # Initialize summary writer.
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  if initial_step == 1:
    writer.write_hparams({
        k: v
        for k, v in config.items()
        if isinstance(v, (bool, float, int, str))
    })

  logging.info('Starting training loop at step %d.', initial_step)
  hooks = []
  report_progress = training.ReportProgress(
      global_batch_size, num_train_steps=num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks.append(report_progress)

  eval_shape_zyx = train_eval_size(info, config).tolist()[::-1]
  eval_tracker = tracker.EvalTracker(eval_shape_zyx, fov_shifts)

  batch_iter = input_pipeline.get_batch_iter(
      train_iter,
      eval_tracker,
      policy_fn,
      info,
      config,
      seed_shape=tuple(train_canvas_size(info, config).tolist()[::-1]),
      batch_size=host_batch_size,
      jmp_policy=jmp_policy,
  )

  train_metrics = None
  shutdown_request = False
  timings = collections.defaultdict(list)

  def postprocess_batch(batch):
    # Unpack batch dim into (device, batch).
    def _reshape(x):
      x = np.asarray(x)
      per_device_data = np.split(x, len(mesh.local_devices), axis=0)

      on_dev = jax.device_put(per_device_data, mesh.local_devices)
      global_shape = (
          len(batch_sharding.device_set) * config.per_device_batch_size,
      ) + per_device_data[0].shape[1:]
      return jax.make_array_from_single_device_arrays(
          global_shape, batch_sharding, on_dev
      )

    return jax.tree.map(
        _reshape,
        {
            'patch': batch['patch'],
            'seed': batch['seed'],
            'label': batch['label'],
            'weight': batch['weight'],
        },
    )

  with metric_writers.ensure_flushes(writer):
    # Record a summary scalar to indicate the specific steps at which
    # training restarts occurred.
    writer.write_scalars(initial_step, {'start': 1})

    for step in range(initial_step, num_train_steps + 1):
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        with report_progress.timed('input', wait_jax_async_dispatch=False):
          with training.MeasureTime(timings, 'data_load'):
            batch = next(batch_iter)

          batch = postprocess_batch(batch)

          with training.MeasureTime(timings, 'train_step'):
            state, metrics_update, updated_seed, loss_scale = p_train_step(
                state, batch, loss_scale
            )

        logging.log_first_n(
            logging.INFO, 'Updated seed shape: %r', 1, updated_seed.shape
        )

        with training.MeasureTime(timings, 'metrics'):
          train_metrics = (
              metrics_update
              if train_metrics is None
              else train_metrics.merge(metrics_update)
          )

        with training.MeasureTime(timings, 'update_seed'):
          host_local_seeds = []  # [b, z, y, x, 1] * num_devices
          dev_to_slice = batch_sharding.addressable_devices_indices_map(
              updated_seed.shape
          )

          # Ensure device order is the same as that used to build the
          # global array in postprocess_batch().
          assert list(dev_to_slice.keys()) == list(mesh.local_devices)
          for slc in dev_to_slice.values():
            host_local_seeds.append(updated_seed[slc])

          batch_iter.update_seeds(host_local_seeds)

      with training.MeasureTime(timings, 'admin'):
        if checkpoint_manager.should_save(step) or is_last_step:
          logging.info('Saving checkpoint at %d.', step)
          train_state = jax.tree.map(np.array, state)
          checkpoint_manager.save(
              step, args=_make_ckpt_args(train_state, train_iter)
          )

        if checkpoint_manager.reached_preemption(step):
          logging.warn('Interrupting training loop due to shutdown request.')
          logging.flush()
          shutdown_request = True
          break

        for h in hooks:
          h(step)

        if step % config.log_loss_every_steps == 0 or is_last_step:
          scalars = train_metrics.compute()
          for name, values in timings.items():
            scalars[f'time_{name}'] = float(np.mean(values))
            scalars[f'time_{name}/min'] = float(np.min(values))
            scalars[f'time_{name}/max'] = float(np.max(values))

          timings = collections.defaultdict(list)
          raws = []
          for summ in eval_tracker.get_summaries():
            if summ.HasField('simple_value'):
              scalars[summ.tag] = summ.simple_value
            else:
              s = tf.compat.v1.summary.Summary()
              s.value.append(summ)
              raws.append(s.SerializeToString())

          writer.write_scalars(step, scalars)
          if jax.process_index() == 0:
            # pylint:disable=protected-access
            tfw = _get_tf_writer(writer._writers)
            assert tfw is not None
            # TODO(mjanusz): Find a cleaner and less brittle way of saving
            # raw summaries.
            with tfw._summary_writer.as_default():
              for s in raws:
                tf.summary.experimental.write_raw_pb(s, step=step)

            # pylint:enable=protected-access

          train_metrics = None
          eval_tracker.reset()

  checkpoint_manager.wait_until_finished()
  logging.info('Finished training at step %d.', step)

  if shutdown_request:
    # Allow time for other workers to finish checkpoint saving. Soon after
    # the first worker is terminated, it will be detected that the clique
    # is no longer complete, which will cause an immediate restart of the
    # current process via std::quick_exit(42).
    time.sleep(60)

    # This return code causes Borglet to restart the binary without changing
    # the state of the task as seen by the Borgmaster.
    os._exit(42)  # pylint:disable=protected-access
