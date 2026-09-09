# Copyright 2026 Google Inc.
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

"""Training script for SECGAN models.

See https://arxiv.org/abs/1703.10593 for more info on CycleGANs and
https://www.biorxiv.org/content/10.1101/548081v1 for Segmentation-Enhanced
CycleGANs (SECGANs).

Patches used for training are defined through bounding boxes (--bbox_*)
which are going to be randomly sampled.
"""

import collections
import dataclasses
import io
import os
import time
from typing import Any, Callable, cast, Sequence

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from connectomics.jax import training
from connectomics.jax.models import util as model_util
from etils import epath
from ffn.input import volume
from ffn.jax import train as ffn_train
from ffn.secgan import models
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
import PIL.Image
import tensorflow as tf


class TrainState(flax.struct.PyTreeNode):
  # dataclass_transform
  """Training state for SECGAN.

  Keeps track of parameters and optimizer states for both generators
  and both discriminators.
  """
  step: int

  gen_params: Any
  gen_opt_state: optax.OptState

  disc_a_params: flax.core.FrozenDict[str, Any]
  disc_a_opt_state: optax.OptState

  disc_b_params: flax.core.FrozenDict[str, Any]
  disc_b_opt_state: optax.OptState

  # Optional separate discriminator for raw image A
  # (used when ffn_mode='separate')
  disc_a_raw_params: flax.core.FrozenDict[str, Any] | None = None
  disc_a_raw_opt_state: optax.OptState | None = None

  # EMA parameters
  gen_ema_params: Any = None
  disc_a_ema_params: flax.core.FrozenDict[str, Any] | None = None
  disc_b_ema_params: flax.core.FrozenDict[str, Any] | None = None
  disc_a_raw_ema_params: flax.core.FrozenDict[str, Any] | None = None


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Metrics collected during SECGAN training."""
  gen_a_img_loss: metrics.Average.from_output('gen_a_img_loss')  # pyrefly: ignore[invalid-annotation]
  gen_b_img_loss: metrics.Average.from_output('gen_b_img_loss')  # pyrefly: ignore[invalid-annotation]
  gen_b_ffn_loss: metrics.Average.from_output('gen_b_ffn_loss')  # pyrefly: ignore[invalid-annotation]
  gen_b_both_loss: metrics.Average.from_output('gen_b_both_loss')  # pyrefly: ignore[invalid-annotation]
  cycle_loss_a: metrics.Average.from_output('cycle_loss_a')  # pyrefly: ignore[invalid-annotation]
  cycle_loss_b: metrics.Average.from_output('cycle_loss_b')  # pyrefly: ignore[invalid-annotation]
  # Discriminator A losses (one active slot per ffn_mode, see above).
  disc_a_real_img_loss: metrics.Average.from_output('disc_a_real_img_loss')  # pyrefly: ignore[invalid-annotation]
  disc_a_fake_img_loss: metrics.Average.from_output('disc_a_fake_img_loss')  # pyrefly: ignore[invalid-annotation]
  disc_a_real_ffn_loss: metrics.Average.from_output('disc_a_real_ffn_loss')  # pyrefly: ignore[invalid-annotation]
  disc_a_fake_ffn_loss: metrics.Average.from_output('disc_a_fake_ffn_loss')  # pyrefly: ignore[invalid-annotation]
  disc_a_real_both_loss: metrics.Average.from_output('disc_a_real_both_loss')  # pyrefly: ignore[invalid-annotation]
  disc_a_fake_both_loss: metrics.Average.from_output('disc_a_fake_both_loss')  # pyrefly: ignore[invalid-annotation]
  # Discriminator B always critiques the raw image.
  disc_b_real_img_loss: metrics.Average.from_output('disc_b_real_img_loss')  # pyrefly: ignore[invalid-annotation]
  disc_b_fake_img_loss: metrics.Average.from_output('disc_b_fake_img_loss')  # pyrefly: ignore[invalid-annotation]
  learning_rate: metrics.LastValue.from_output('learning_rate')  # pyrefly: ignore[invalid-annotation]


@dataclasses.dataclass(frozen=True)
class FfnContext:
  """Bundles the optional FFN model with its parameters and config.

  All fields default to None, meaning no FFN is used and FFN preprocessing is
  a no-op passthrough.
  """
  model: nn.Module | None = None
  params: flax.core.FrozenDict[str, Any] | None = None
  batch_stats: flax.core.FrozenDict[str, Any] | None = None
  config: ml_collections.ConfigDict | None = None


def _apply_module(
    module: nn.Module, params: Any, x: jnp.ndarray
) -> jnp.ndarray:
  """Applies `module` with `params` to `x`, returning the output array."""
  return cast(jnp.ndarray, module.apply({'params': params}, x))


def _apply_grads(
    optimizer: optax.GradientTransformation,
    grads: Any,
    opt_state: Any,
    params: Any,
) -> tuple[Any, Any]:
  """Applies `grads` to `params`, returning (new_params, new_opt_state)."""
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  return optax.apply_updates(params, updates), new_opt_state


def _disc_loss(
    disc_real: jnp.ndarray, disc_fake: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the least-squares discriminator loss."""
  d_loss_r = jnp.mean((disc_real - 1.0) ** 2)
  d_loss_f = jnp.mean(disc_fake ** 2)
  return d_loss_r, d_loss_f


def _gen_disc_loss(disc_gen: jnp.ndarray) -> jnp.ndarray:
  """Computes the generator's adversarial loss (fool the discriminator)."""
  return jnp.mean((disc_gen - 1.0) ** 2)


def _disc_value_and_grad(
    discriminator: nn.Module,
    disc_params: Any,
    real_crop: jnp.ndarray,
    fake_crop: jnp.ndarray,
    loss_scale: float = 1.0,
) -> tuple[Any, Any]:
  """Computes LSGAN discriminator loss and grads over (real, fake) crops.

  Args:
    discriminator: Discriminator module.
    disc_params: Discriminator parameters to differentiate.
    real_crop: Real (already preprocessed) image crop.
    fake_crop: Fake (already preprocessed) image crop.
    loss_scale: Scalar multiplier applied to the total loss (and hence grads).

  Returns:
    Tuple of ((loss, (d_loss_real, d_loss_fake)), grads).
  """
  def loss_fn(
      disc_params,
  ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    disc_real = _apply_module(discriminator, disc_params, real_crop)
    disc_fake = _apply_module(discriminator, disc_params, fake_crop)
    d_loss_r, d_loss_f = _disc_loss(disc_real, disc_fake)
    return loss_scale * (d_loss_r + d_loss_f), (d_loss_r, d_loss_f)

  return jax.value_and_grad(loss_fn, has_aux=True)(disc_params)


def _update_ema(ema_params, new_params, step_size) -> Any:
  """Returns the EMA-updated params, initializing to new_params if empty."""
  if ema_params is None:
    return new_params
  return optax.incremental_update(new_params, ema_params, step_size=step_size)


def _as_list(value: str | Sequence[Any]) -> list[Any]:
  """Wraps a bare string in a single-element list, passing others through."""
  if isinstance(value, str):
    return [value]
  return list(value)


def generator_input_size(
    config: ml_collections.ConfigDict,
) -> tuple[int, int, int]:
  """Computes the generator input size (zyx).

  This is the discriminator's input size plus enough context for two
  consecutive generator applications (needed for the cycle path).

  Args:
    config: SECGAN training configuration.

  Returns:
    (z, y, x) input size for the generator.
  """
  disc_input = np.array(config.discriminator_input_size)
  gen_config = models.ResNetGeneratorConfig(
      depth=config.generator_depth,
      features=config.generator_features,
  )
  gen_delta = models.ResNetGenerator(config=gen_config).out_delta
  return tuple((disc_input + 4 * gen_delta).tolist())


def crop_center(
    x: jnp.ndarray, target_shape_zyx: tuple[int, ...]
) -> jnp.ndarray:
  """Crops `x` to `target_shape_zyx` (a 3-element zyx shape) centered."""
  sz, sy, sx = x.shape[1], x.shape[2], x.shape[3]
  tz, ty, tx = target_shape_zyx
  dz, dy, dx = (sz - tz) // 2, (sy - ty) // 2, (sx - tx) // 2
  return x[:, dz:dz+tz, dy:dy+ty, dx:dx+tx, :]


def raw_to_norm(data: jnp.ndarray) -> jnp.ndarray:
  """Converts uint8 [0, 255] image data to normalized [-1, 1]."""
  return data.astype(jnp.float32) / 127.5 - 1.0


def norm_to_raw(data: jnp.ndarray) -> jnp.ndarray:
  """Converts normalized [-1, 1] image data to raw [0, 255] float."""
  return jnp.clip((data + 1.0) * 127.5, 0, 255)


def make_seed(
    shape: tuple[int, int, int],
    batch_size: int,
    pad: float = 0.05,
    seed: float = 0.95,
) -> jnp.ndarray:
  """Creates an FFN seed tensor."""
  seed_array = jnp.full((batch_size,) + shape + (1,), pad, dtype=jnp.float32)
  center_z, center_y, center_x = [s // 2 for s in shape]
  seed_array = seed_array.at[:, center_z, center_y, center_x, 0].set(seed)
  return seed_array


def _normalize_downsample(
    downsample: int | Sequence[int],
) -> tuple[int, int, int]:
  """Normalizes an FFN downsample spec into a (dz, dy, dx) tuple.

  Args:
    downsample: Either a single int factor applied to all spatial dims, or a
      sequence of three per-axis factors in (x, y, z) order.

  Returns:
    A (dz, dy, dx) tuple of downsample factors, matching the [batch, z, y, x,
    channels] tensor layout.
  """
  if isinstance(downsample, int):
    return (downsample, downsample, downsample)
  dx, dy, dz = downsample
  return (dz, dy, dx)


def apply_ffn(
    image: jnp.ndarray,
    ffn_model: nn.Module,
    ffn_params: flax.core.FrozenDict[str, Any],
    ffn_batch_stats: flax.core.FrozenDict[str, Any] | None,
    pad: float,
    downsample: int | Sequence[int] = 1,
) -> jnp.ndarray:
  """Runs a single step of the JAX FFN model.

  Args:
    image: Input image, shape [batch, z, y, x, channels].
    ffn_model: FFN Flax model.
    ffn_params: FFN model parameters.
    ffn_batch_stats: Optional FFN batch statistics.
    pad: Seed padding value.
    downsample: Average-pooling factor(s) applied before the FFN. Either a
        single int applied isotropically, or a 3-element (x, y, z) sequence for
        anisotropic pooling. Useful when the input is at higher resolution than
        the FFN was trained on (e.g. 8nm input with a 16nm FFN, downsample=2).

  Returns:
    FFN output (sigmoid - 0.5), same spatial dims as input.
  """
  batch_size, z, y, x, channels = image.shape
  assert channels == 1

  dz, dy, dx = _normalize_downsample(downsample)
  if dz > 1 or dy > 1 or dx > 1:
    # Average pool [batch, z, y, x, c] along spatial dims.
    image = image.reshape(
        batch_size, z // dz, dz, y // dy, dy, x // dx, dx, channels
    )
    image = image.mean(axis=(2, 4, 6))
    _, z, y, x, _ = image.shape

  # Normalize image for FFN: (raw - 128) / 33
  normalized_image = (image - 128.0) / 33.0

  # Make seed (in JAX)
  seed = make_seed((z, y, x), batch_size, pad=pad, seed=0.95)
  seed = jax.scipy.special.logit(seed)

  # Concatenate image and seed
  data = jnp.concatenate((normalized_image, seed), axis=-1)

  # FFN variables
  variables = {'params': ffn_params}
  if ffn_batch_stats is not None:
    variables['batch_stats'] = ffn_batch_stats

  logits = cast(jnp.ndarray, ffn_model.apply(variables, data))
  return jax.nn.sigmoid(logits) - 0.5


def preprocess(
    norm_image: jnp.ndarray,
    ffn_model: nn.Module | None,
    ffn_params: flax.core.FrozenDict[str, Any] | None,
    ffn_batch_stats: flax.core.FrozenDict[str, Any] | None,
    ffn_mode: str,
    ffn_pad: float,
    ffn_input_size: tuple[int, int, int] | None,
    ffn_downsample: int | Sequence[int] = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Preprocesses a domain-A image into the discriminator's input.

  When an FFN model is provided, the (raw) image is first run through the FFN
  and the result is combined according to `ffn_mode`. When `ffn_model` is None,
  this is a no-op passthrough that returns the input unchanged.

  Args:
    norm_image: Image normalized to [-1, 1].
    ffn_model: The FFN module, or None to disable FFN preprocessing.
    ffn_params: FFN parameters.
    ffn_batch_stats: FFN batch statistics.
    ffn_mode: Controls what the domain-A discriminator sees. Valid values:
      * 'mask': The discriminator sees only the FFN output (single channel).
        The raw image is used solely as FFN input and is not shown to the
        discriminator.
      * 'both': The discriminator sees the FFN output concatenated with the
        normalized raw image (two channels), so it can judge the pair jointly.
      * 'separate': Like 'mask' for this function (only the FFN output is
        returned here), but training additionally uses a second, dedicated
        discriminator (`disc_a_raw`) that critiques the raw image directly.
        That extra critic is wired up in the training step, not here.
      Any value other than 'both' is treated identically by this function
      (only the FFN output is returned); the 'mask' vs 'separate' distinction
      is realized elsewhere via the separate raw-image discriminator.
    ffn_pad: Padding value passed to the FFN.
    ffn_input_size: If set, the input is center-cropped to this (z, y, x) size
      before running the FFN.
    ffn_downsample: Downsampling factor(s) applied when running the FFN.

  Returns:
    A tuple of (discriminator_input, ffn_output_for_visualization). When
    `ffn_model` is None, this is (norm_image, raw_image).
  """
  if ffn_model is None:
    return norm_image, norm_to_raw(norm_image).astype(jnp.uint8)

  raw_image = norm_to_raw(norm_image)
  assert ffn_params is not None  # Always provided when ffn_model is set.

  if ffn_input_size is not None:
    raw_image = crop_center(raw_image, ffn_input_size)
    norm_image = crop_center(norm_image, ffn_input_size)

  ffn_out = apply_ffn(raw_image, ffn_model, ffn_params, ffn_batch_stats,
                      ffn_pad, downsample=ffn_downsample)
  ffn_out_vis = jnp.clip((ffn_out + 0.5) * 255.0, 0.0, 255.0)

  if ffn_mode == 'both':
    prep_image = jnp.concatenate((ffn_out, norm_image), axis=-1)
    return prep_image, ffn_out_vis
  else:
    return ffn_out, ffn_out_vis


def load_ffn_checkpoint(
    checkpoint_path: str,
    ffn_config: ml_collections.ConfigDict,
) -> ffn_train.TrainState:
  """Loads FFN TrainState from checkpoint."""
  ffn_model = model_util.model_from_config(ffn_config)
  ffn_input_shape = (1,) + tuple(ffn_config.fov_size)[::-1] + (2,)

  rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)}
  variables = ffn_model.init(rngs, jnp.ones(ffn_input_shape))
  params = variables['params']
  batch_stats = variables.get('batch_stats', None)

  tx, _ = training.get_optimizer(ffn_config)
  state_template = ffn_train.TrainState(
      step=0,
      opt_state=tx.init(params),
      batch_stats=batch_stats,
      params=params,
      ema_params=params if ffn_config.get('ema_decay', 0.0) > 0.0 else None,
  )

  restored_dict = None
  restored_state = None
  train_state_path = epath.Path(checkpoint_path) / 'train_state'
  try:
    handler = ocp.StandardCheckpointHandler()
    restored_state = handler.restore(
        train_state_path, args=ocp.args.StandardRestore(state_template)
    )
  except Exception as e:  # pylint:disable=broad-except
    logging.info(
        'FFN checkpoint failed StandardCheckpointHandler'
        ' (%s). Attempting PyTreeCheckpointHandler...',
        e,
    )
    try:
      raw_handler = ocp.PyTreeCheckpointHandler()
      restored_dict = raw_handler.restore(train_state_path)
    except Exception as e2:  # pylint:disable=broad-except
      logging.info(
          'FFN checkpoint is not in Orbax format (%s). '
          'Loading with CLU checkpoint reader.',
          e2,
      )
      from clu import checkpoint as clu_checkpoint  # pylint:disable=g-import-not-at-top
      ckpt = clu_checkpoint.Checkpoint(
          os.path.dirname(checkpoint_path)
      )
      restored_dict = ckpt.load_state(state=None, checkpoint=checkpoint_path)

  if restored_dict is None:
    if restored_state is None:
      raise ValueError(
          f'Failed to restore FFN checkpoint state from {checkpoint_path}.'
      )
    return restored_state

  if 'params' in restored_dict:
    params = restored_dict['params']
  elif 'optimizer' in restored_dict:
    opt_target = restored_dict['optimizer']['target']
    if isinstance(opt_target, dict) and 'params' in opt_target:
      params = opt_target['params']
    else:
      params = opt_target
  else:
    raise ValueError(
        'Could not find params in restored checkpoint '
        f'dict: {restored_dict.keys()}'
    )
  batch_stats = restored_dict.get('batch_stats', None)
  return ffn_train.TrainState(
      step=restored_dict.get('step', 0),
      opt_state=tx.init(params),
      batch_stats=batch_stats,
      params=params,
      ema_params=params if ffn_config.get('ema_decay', 0.0) > 0.0 else None,
  )


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: jax.Array,
) -> tuple[nn.Module, nn.Module, optax.Schedule, optax.GradientTransformation,
           optax.GradientTransformation, TrainState]:
  """Instantiates and initializes the SECGAN models.

  Args:
    config: SECGAN training configuration.
    rng: JAX PRNG key.

  Returns:
    Tuple of (generator, discriminator, lr_schedule, gen_optimizer,
              disc_optimizer, initial_train_state).
  """
  gen_config = models.ResNetGeneratorConfig(
      depth=config.generator_depth,
      features=config.generator_features,
  )
  generator = models.ResNetGenerator(config=gen_config)

  # Discriminator config classes follow the '<Class>Config' naming convention.
  disc_cls = getattr(models, config.discriminator_class)
  disc_config_cls = getattr(models, config.discriminator_class + 'Config')
  disc_config = disc_config_cls(**config.discriminator_args)
  discriminator = disc_cls(config=disc_config)

  # Compute input shapes.
  gen_in_size_zyx = generator_input_size(config)
  gen_in_shape = (1,) + gen_in_size_zyx + (config.input_channels,)

  # Discriminator inputs:
  # disc_b always takes raw image (1 channel)
  ffn_active = config.get('ffn_a_checkpoint') is not None
  ffn_mode = config.get('ffn_mode', 'mask')

  disc_a_channels = 2 if (ffn_active and ffn_mode == 'both') else 1
  disc_in_size = tuple(config.discriminator_input_size)
  disc_a_in_shape = 1, *disc_in_size, disc_a_channels
  disc_b_in_shape = 1, *disc_in_size, 1

  # Initialize parameters.
  rng, gen_rng, disc_a_rng, disc_b_rng = jax.random.split(rng, 4)

  # The generator is shared between A->B and B->A; using separate a2b and b2a
  # param collections prevents weight sharing after the first step.
  gen_variables = generator.init(gen_rng, jnp.ones(gen_in_shape))
  gen_params = gen_variables['params']
  all_gen_params = flax.core.freeze({'a2b': gen_params, 'b2a': gen_params})
  parameter_overview.log_parameter_overview(gen_params, msg='Generator A2B')

  # disc_a and disc_b use separate params due to possible size differences.
  disc_a_variables = discriminator.init(disc_a_rng, jnp.ones(disc_a_in_shape))
  disc_a_params = disc_a_variables['params']
  disc_b_variables = discriminator.init(disc_b_rng, jnp.ones(disc_b_in_shape))
  disc_b_params = disc_b_variables['params']
  parameter_overview.log_parameter_overview(
      disc_a_params, msg='Discriminator A'
  )
  parameter_overview.log_parameter_overview(
      disc_b_params, msg='Discriminator B'
  )

  disc_a_raw_params = None
  disc_a_raw_opt_state = None
  if ffn_active and ffn_mode == 'separate':
    rng, disc_raw_rng = jax.random.split(rng)
    disc_raw_variables = discriminator.init(
        disc_raw_rng, jnp.ones(disc_b_in_shape)
    )
    disc_a_raw_params = disc_raw_variables['params']
    parameter_overview.log_parameter_overview(
        disc_a_raw_params, msg='Discriminator A Raw'
    )

  gen_tx, lr = training.get_optimizer(config)
  disc_tx, _ = training.get_optimizer(config)

  if disc_a_raw_params is not None:
    disc_a_raw_opt_state = disc_tx.init(disc_a_raw_params)

  ema_decay = config.get('ema_decay', 0.0)
  state = TrainState(
      step=0,
      gen_params=all_gen_params,
      gen_opt_state=gen_tx.init(all_gen_params),
      gen_ema_params=all_gen_params if ema_decay > 0.0 else None,
      disc_a_params=disc_a_params,
      disc_a_opt_state=disc_tx.init(disc_a_params),
      disc_a_ema_params=disc_a_params if ema_decay > 0.0 else None,
      disc_b_params=disc_b_params,
      disc_b_opt_state=disc_tx.init(disc_b_params),
      disc_b_ema_params=disc_b_params if ema_decay > 0.0 else None,
      disc_a_raw_params=disc_a_raw_params,
      disc_a_raw_opt_state=disc_a_raw_opt_state,
      disc_a_raw_ema_params=disc_a_raw_params if ema_decay > 0.0 else None,
  )

  return generator, discriminator, lr, gen_tx, disc_tx, state


def train_step_gen(
    generator: nn.Module,
    discriminator: nn.Module,
    state: TrainState,
    real_a: jnp.ndarray,
    real_b: jnp.ndarray,
    config: ml_collections.ConfigDict,
    ffn: FfnContext = FfnContext(),
) -> tuple[Any, ...]:
  """Performs the generator training step.

  This updates both generators (A2B and B2A) simultaneously with:
  - Adversarial loss: fool the discriminator
  - Cycle-consistency loss: A->B->A ≈ A and B->A->B ≈ B

  Args:
    generator: Generator module.
    discriminator: Discriminator module.
    state: Current training state.
    real_a: Batch of images from domain A.
    real_b: Batch of images from domain B.
    config: Training configuration.
    ffn: Optional FFN context (model, params, batch stats, config).

  Returns:
    Tuple of (updated_state, metrics_dict, gen_a, gen_b, cyc_a, cyc_b).
  """
  disc_shape_zyx = tuple(config.discriminator_input_size)
  ffn_mode = config.get('ffn_mode', 'mask')
  ffn_pad = config.get('ffn_pad', 0.5)
  ffn_downsample = config.get('ffn_downsample', 1)
  input_channels = config.get('input_channels', 1)

  def loss_fn(gen_params) -> tuple[jnp.ndarray, dict[str, Any]]:
    # Generate translations.
    gen_b = _apply_module(generator, gen_params['a2b'], real_a)  # A->B
    gen_a = _apply_module(generator, gen_params['b2a'], real_b)  # B->A

    # Complete cycle. Generators only produce single-channel image data, so
    # any auxiliary input channels (e.g. a mask) must be re-attached before the
    # second generator application. The extra channels are passed through
    # unmodified from the original input, cropped to match the generator output.
    cyc_ba_in = gen_b
    cyc_ab_in = gen_a
    if input_channels > 1:
      zyx = gen_b.shape[1:4]
      cyc_ba_in = jnp.concatenate(
          [gen_b, crop_center(real_a[..., 1:], zyx)], axis=-1
      )
      cyc_ab_in = jnp.concatenate(
          [gen_a, crop_center(real_b[..., 1:], zyx)], axis=-1
      )

    cyc_a = _apply_module(generator, gen_params['b2a'], cyc_ba_in)  # A->B->A
    cyc_b = _apply_module(generator, gen_params['a2b'], cyc_ab_in)  # B->A->B

    # Crop generated images for discriminator.
    gen_crop_b = crop_center(gen_b, disc_shape_zyx)

    if ffn.model is not None:
      ffn_config = ffn.config
      if ffn_config is None:
        raise ValueError('ffn_config must be provided if ffn_model is active.')
      disc_shape_a = tuple(ffn_config.fov_size)[::-1]
    else:
      disc_shape_a = disc_shape_zyx
    gen_crop_a = crop_center(gen_a, disc_shape_a)

    # Preprocess domain A. preprocess() is a no-op passthrough when there is
    # no FFN model.
    prep_gen_a, _ = preprocess(
        gen_crop_a, ffn.model, ffn.params, ffn.batch_stats,
        ffn_mode, ffn_pad, ffn_input_size=None,
        ffn_downsample=ffn_downsample,
    )

    # Discriminator evaluations on generated images.
    disc_gen_a = _apply_module(discriminator, state.disc_a_params, prep_gen_a)
    disc_gen_b = _apply_module(discriminator, state.disc_b_params, gen_crop_b)

    # Adversarial loss (fool the discriminator). disc_b always critiques the
    # raw image B, so g_loss_a is always an image loss.
    g_loss_a = _gen_disc_loss(disc_gen_b)

    # disc_a critiques different content depending on ffn_mode; route the loss
    # into the matching slot so each metric always means the same thing.
    g_loss_b_primary = _gen_disc_loss(disc_gen_a)
    gen_b_img_loss = 0.0
    gen_b_ffn_loss = 0.0
    gen_b_both_loss = 0.0
    if ffn.model is None:
      gen_b_img_loss = g_loss_b_primary
    elif ffn_mode == 'both':
      gen_b_both_loss = g_loss_b_primary
    else:  # 'mask' or 'separate': disc_a critiques the FFN output.
      gen_b_ffn_loss = g_loss_b_primary
      if ffn_mode == 'separate':
        disc_img_ga = _apply_module(
            discriminator, state.disc_a_raw_params, gen_crop_a
        )
        gen_b_img_loss = _gen_disc_loss(disc_img_ga)
    g_loss_b = gen_b_img_loss + gen_b_ffn_loss + gen_b_both_loss

    # Cycle consistency loss.
    c_loss_a = jnp.mean(jnp.abs(
        crop_center(real_a[..., 0:1], cyc_a.shape[1:4]) - cyc_a
    ))
    c_loss_b = jnp.mean(jnp.abs(
        crop_center(real_b[..., 0:1], cyc_b.shape[1:4]) - cyc_b
    ))
    c_loss = c_loss_a + c_loss_b

    total_loss = (g_loss_a + g_loss_b +
                  config.cycle_loss_weight * c_loss)

    aux = {
        'gen_a_img_loss': g_loss_a,
        'gen_b_img_loss': gen_b_img_loss,
        'gen_b_ffn_loss': gen_b_ffn_loss,
        'gen_b_both_loss': gen_b_both_loss,
        'cycle_loss_a': c_loss_a,
        'cycle_loss_b': c_loss_b,
        'gen_b': gen_b,
        'gen_a': gen_a,
        'cyc_a': cyc_a,
        'cyc_b': cyc_b,
    }
    return total_loss, aux

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, aux), grads = grad_fn(state.gen_params)

  # Unpack non-differentiable outputs from aux.
  gen_a = aux.pop('gen_a')
  gen_b = aux.pop('gen_b')
  cyc_a = aux.pop('cyc_a')
  cyc_b = aux.pop('cyc_b')

  return state, aux, gen_a, gen_b, cyc_a, cyc_b, grads


def train_step_disc_a(
    discriminator: nn.Module,
    state: TrainState,
    real_a: jnp.ndarray,
    fake_a: jnp.ndarray,
    config: ml_collections.ConfigDict,
    ffn: FfnContext = FfnContext(),
) -> tuple[Any, ...]:
  """Trains discriminator A.

  Args:
    discriminator: Discriminator module.
    state: Current training state.
    real_a: Real images from domain A.
    fake_a: Generated images for domain A.
    config: Training configuration.
    ffn: Optional FFN context (model, params, batch stats, config).

  Returns:
    Tuple of (loss dict, gradients_a, gradients_a_raw).
  """
  disc_shape_zyx = tuple(config.discriminator_input_size)
  ffn_mode = config.get('ffn_mode', 'mask')
  ffn_pad = config.get('ffn_pad', 0.5)
  ffn_downsample = config.get('ffn_downsample', 1)

  if ffn.model is not None:
    ffn_config = ffn.config
    if ffn_config is None:
      raise ValueError('ffn_config must be provided if ffn_model is active.')
    disc_shape_a = tuple(ffn_config.fov_size)[::-1]
  else:
    disc_shape_a = disc_shape_zyx
  real_crop_a = crop_center(real_a[..., 0:1], disc_shape_a)
  fake_crop_a = crop_center(fake_a, disc_shape_a)

  # preprocess() is a no-op passthrough when there is no FFN model.
  prep_real_a, _ = preprocess(
      real_crop_a, ffn.model, ffn.params, ffn.batch_stats,
      ffn_mode, ffn_pad, ffn_input_size=None,
      ffn_downsample=ffn_downsample,
  )
  prep_fake_a, _ = preprocess(
      fake_crop_a, ffn.model, ffn.params, ffn.batch_stats,
      ffn_mode, ffn_pad, ffn_input_size=None,
      ffn_downsample=ffn_downsample,
  )

  (_, (d_loss_a_real, d_loss_a_fake)), grads_a = _disc_value_and_grad(
      discriminator, state.disc_a_params, prep_real_a, prep_fake_a
  )

  # Route the primary disc_a loss into the slot matching what it critiqued.
  real_img_loss = fake_img_loss = 0.0
  real_ffn_loss = fake_ffn_loss = 0.0
  real_both_loss = fake_both_loss = 0.0
  if ffn.model is None:
    real_img_loss, fake_img_loss = d_loss_a_real, d_loss_a_fake
  elif ffn_mode == 'both':
    real_both_loss, fake_both_loss = d_loss_a_real, d_loss_a_fake
  else:  # 'mask' or 'separate': disc_a critiques the FFN output.
    real_ffn_loss, fake_ffn_loss = d_loss_a_real, d_loss_a_fake

  grads_a_raw = None

  if ffn.model is not None and ffn_mode == 'separate':
    (_, (real_img_loss, fake_img_loss)), grads_a_raw = _disc_value_and_grad(
        discriminator, state.disc_a_raw_params, real_crop_a, fake_crop_a
    )

  aux = {
      'disc_a_real_img_loss': real_img_loss,
      'disc_a_fake_img_loss': fake_img_loss,
      'disc_a_real_ffn_loss': real_ffn_loss,
      'disc_a_fake_ffn_loss': fake_ffn_loss,
      'disc_a_real_both_loss': real_both_loss,
      'disc_a_fake_both_loss': fake_both_loss,
  }
  return aux, grads_a, grads_a_raw


def train_step_disc_b(
    discriminator: nn.Module,
    state: TrainState,
    real_b: jnp.ndarray,
    fake_b: jnp.ndarray,
    config: ml_collections.ConfigDict,
    ffn: FfnContext = FfnContext(),
) -> tuple[Any, ...]:
  """Trains discriminator B.

  Args:
    discriminator: Discriminator module.
    state: Current training state.
    real_b: Real images from domain B.
    fake_b: Generated images for domain B.
    config: Training configuration.
    ffn: Optional FFN context (model, params, batch stats, config).

  Returns:
    Tuple of (loss dict, gradients).
  """
  disc_shape_zyx = tuple(config.discriminator_input_size)
  real_crop = crop_center(real_b[..., 0:1], disc_shape_zyx)
  fake_crop = crop_center(fake_b, disc_shape_zyx)

  ffn_active = ffn.model is not None
  ffn_mode = config.get('ffn_mode', 'mask')

  loss_scale = 2.0 if (ffn_active and ffn_mode == 'separate') else 1.0
  (_, (d_loss_b_real, d_loss_b_fake)), grads = _disc_value_and_grad(
      discriminator, state.disc_b_params, real_crop, fake_crop,
      loss_scale=loss_scale,
  )

  aux = {
      'disc_b_real_img_loss': d_loss_b_real,
      'disc_b_fake_img_loss': d_loss_b_fake,
  }
  return aux, grads


def full_train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    state: TrainState,
    gen_optimizer: optax.GradientTransformation,
    disc_optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
    real_a: jnp.ndarray,
    real_b: jnp.ndarray,
    config: ml_collections.ConfigDict,
    ffn: FfnContext = FfnContext(),
) -> tuple[Any, ...]:
  """Performs a full SECGAN training step.

  This includes:
  1. Generator step (both A2B and B2A, with cycle loss)
  2. Discriminator A step
  3. Discriminator B step

  Args:
    generator: Generator module.
    discriminator: Discriminator module.
    state: Current training state.
    gen_optimizer: Generator optimizer.
    disc_optimizer: Discriminator optimizer (same for A and B).
    schedule: Learning rate schedule.
    real_a: Batch of images from domain A.
    real_b: Batch of images from domain B.
    config: Training configuration.
    ffn: Optional FFN context (model, params, batch stats, config).

  Returns:
    Tuple of (new_state, metrics, gen_a, gen_b, cyc_a, cyc_b).
  """
  step = state.step + 1

  # --- Generator step ---
  _, gen_metrics, gen_a, gen_b, cyc_a, cyc_b, gen_grads = train_step_gen(
      generator, discriminator, state, real_a, real_b, config, ffn)

  new_gen_params, new_gen_opt_state = _apply_grads(
      gen_optimizer, gen_grads, state.gen_opt_state, state.gen_params)

  # --- Discriminator A step (use generated images from the gen step) ---
  disc_a_metrics, disc_a_grads, disc_a_raw_grads = train_step_disc_a(
      discriminator, state, real_a, gen_a, config, ffn)

  new_disc_a_params, new_disc_a_opt_state = _apply_grads(
      disc_optimizer, disc_a_grads, state.disc_a_opt_state, state.disc_a_params)

  # Update disc_a_raw if needed
  new_disc_a_raw_params = state.disc_a_raw_params
  new_disc_a_raw_opt_state = state.disc_a_raw_opt_state
  if disc_a_raw_grads is not None:
    assert state.disc_a_raw_opt_state is not None
    assert state.disc_a_raw_params is not None
    new_disc_a_raw_params, new_disc_a_raw_opt_state = _apply_grads(
        disc_optimizer, disc_a_raw_grads, state.disc_a_raw_opt_state,
        state.disc_a_raw_params)

  # --- Discriminator B step ---
  disc_b_metrics, disc_b_grads = train_step_disc_b(
      discriminator, state, real_b, gen_b, config, ffn)

  new_disc_b_params, new_disc_b_opt_state = _apply_grads(
      disc_optimizer, disc_b_grads, state.disc_b_opt_state, state.disc_b_params)

  ema_decay = config.get('ema_decay', 0.0)
  new_gen_ema_params = state.gen_ema_params
  new_disc_a_ema_params = state.disc_a_ema_params
  new_disc_b_ema_params = state.disc_b_ema_params
  new_disc_a_raw_ema_params = state.disc_a_raw_ema_params

  if ema_decay > 0.0:
    decay = jnp.array(ema_decay, dtype=jnp.float32)
    decay = jnp.where(state.step == 0, 0.0, decay)
    step_size = 1.0 - decay

    new_gen_ema_params = _update_ema(
        new_gen_ema_params, new_gen_params, step_size)
    new_disc_a_ema_params = _update_ema(
        new_disc_a_ema_params, new_disc_a_params, step_size)
    new_disc_b_ema_params = _update_ema(
        new_disc_b_ema_params, new_disc_b_params, step_size)
    if state.disc_a_raw_params is not None:
      new_disc_a_raw_ema_params = _update_ema(
          new_disc_a_raw_ema_params, new_disc_a_raw_params, step_size)

  new_state = state.replace(
      step=step,
      gen_params=new_gen_params,
      gen_opt_state=new_gen_opt_state,
      gen_ema_params=new_gen_ema_params,
      disc_a_params=new_disc_a_params,
      disc_a_opt_state=new_disc_a_opt_state,
      disc_a_ema_params=new_disc_a_ema_params,
      disc_b_params=new_disc_b_params,
      disc_b_opt_state=new_disc_b_opt_state,
      disc_b_ema_params=new_disc_b_ema_params,
      disc_a_raw_params=new_disc_a_raw_params,
      disc_a_raw_opt_state=new_disc_a_raw_opt_state,
      disc_a_raw_ema_params=new_disc_a_raw_ema_params,
  )

  lr = schedule(state.step)
  all_metrics = {**gen_metrics, **disc_a_metrics, **disc_b_metrics,
                 'learning_rate': lr}
  metrics_update = TrainMetrics.single_from_model_output(**all_metrics)

  return new_state, metrics_update, gen_a, gen_b, cyc_a, cyc_b


def _add_image_summary(tag: str, parts: list[np.ndarray]) -> bytes:
  """Builds a serialized image-summary proto from concatenated `parts`."""
  concat = np.concatenate(parts, axis=1)
  im = PIL.Image.fromarray(concat, 'L')
  buf = io.BytesIO()
  im.save(buf, 'PNG')

  s = tf.compat.v1.summary.Summary(
      value=[
          tf.compat.v1.summary.Summary.Value(
              tag=tag,
              image=tf.compat.v1.summary.Summary.Image(
                  height=concat.shape[0],
                  width=concat.shape[1],
                  colorspace=1,
                  encoded_image_string=buf.getvalue(),
              ),
          )
      ]
  )
  return s.SerializeToString()


def visualize(
    writer: metric_writers.MetricWriter,
    step: int,
    real_a: np.ndarray,
    real_b: np.ndarray,
    gen_a: np.ndarray,
    gen_b: np.ndarray,
    cyc_a: np.ndarray,
    cyc_b: np.ndarray,
    ffn: FfnContext = FfnContext(),
    max_examples: int = 8,
) -> None:
  """Generates and writes image summaries for SECGAN training.

  For each direction (A->B, B->A), visualizes XY, XZ, and YZ mid-slices of:
  original, generated, and cycle-reconstructed image, for up to
  ``max_examples`` batch elements.

  When FFN is available, also visualizes FFN segmentation on real_a and gen_a
  with all three orthogonal views.

  Args:
    writer: Metric writer.
    step: Current training step.
    real_a: Original domain A image [B, Z, Y, X, C].
    real_b: Original domain B image [B, Z, Y, X, C].
    gen_a: Generated domain A image (from B).
    gen_b: Generated domain B image (from A).
    cyc_a: Cycle-reconstructed A (A->B->A).
    cyc_b: Cycle-reconstructed B (B->A->B).
    ffn: Optional FFN context (model, params, batch stats, config).
    max_examples: Maximum number of batch examples to visualize.
  """
  tfw = _get_tf_writer(writer)
  if tfw is None:
    return

  def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip((np.array(x) + 1.0) * 127.5, 0, 255).astype(np.uint8)

  def _to_uint8_raw(x: np.ndarray) -> np.ndarray:
    """Converts already-in-[0,255] float data to uint8."""
    return np.clip(np.array(x), 0, 255).astype(np.uint8)

  def _mid_slice(vol: np.ndarray, idx: int, axis: int) -> np.ndarray:
    """Returns the middle slice of batch element idx along a spatial axis.

    Args:
      vol: 5-D array [B, Z, Y, X, C].
      idx: Batch element index.
      axis: Spatial axis to slice through: 1=Z (XY view), 2=Y (XZ view),
        3=X (YZ view).

    Returns:
      A 2-D array (the mid-plane through `axis`, channel 0).
    """
    v = np.array(vol)
    mid = v.shape[axis] // 2
    return np.take(v[idx, ..., 0], mid, axis=axis - 1)

  def _mid_xy(vol: np.ndarray, idx: int) -> np.ndarray:
    return _mid_slice(vol, idx, axis=1)

  def _mid_xz(vol: np.ndarray, idx: int) -> np.ndarray:
    return _mid_slice(vol, idx, axis=2)

  def _mid_yz(vol: np.ndarray, idx: int) -> np.ndarray:
    return _mid_slice(vol, idx, axis=3)

  def _pad_to_match(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Pads images to the same spatial size for concatenation."""
    max_h = max(im.shape[0] for im in imgs)
    max_w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
      dh = max_h - im.shape[0]
      dw = max_w - im.shape[1]
      padded.append(np.pad(im, ((dh // 2, dh - dh // 2),
                                (dw // 2, dw - dw // 2))))
    return padded

  def _build_grid(
      volumes: list[np.ndarray],
      convert_fn: Callable[[np.ndarray], np.ndarray],
      n_examples: int,
  ) -> np.ndarray:
    """Builds a grid image: rows=examples, cols=XY|XZ|YZ for each volume.

    For each example, the columns are arranged as:
      vol0_XY | vol0_XZ | vol0_YZ | sep | vol1_XY | vol1_XZ | vol1_YZ | sep ...

    A 2-pixel black separator column is inserted between each volume group.

    Args:
      volumes: List of 5-D arrays [B, Z, Y, X, C].
      convert_fn: Function to convert a 2-D slice to uint8 (e.g. _to_uint8).
      n_examples: Number of batch examples to show.

    Returns:
      A single 2-D uint8 numpy array (the assembled grid).
    """
    sep_width = 2
    rows = []
    for i in range(n_examples):
      groups = []
      for vol in volumes:
        xy = convert_fn(_mid_xy(vol, i))
        xz = convert_fn(_mid_xz(vol, i))
        yz = convert_fn(_mid_yz(vol, i))
        # Pad the three slices to the same height for horizontal concat.
        padded = _pad_to_match([xy, xz, yz])
        groups.append(np.concatenate(padded, axis=1))
      # Pad all volume groups to the same height.
      groups = _pad_to_match(groups)
      # Insert separator columns between groups.
      parts = []
      for j, g in enumerate(groups):
        if j > 0:
          parts.append(np.zeros((g.shape[0], sep_width), dtype=np.uint8))
        parts.append(g)
      row = np.concatenate(parts, axis=1)
      rows.append(row)

    # Insert a 2-pixel black separator row between examples.
    sep_row_width = rows[0].shape[1] if rows else 0
    assembled = []
    for k, row in enumerate(rows):
      if k > 0:
        assembled.append(np.zeros((sep_width, sep_row_width), dtype=np.uint8))
      assembled.append(row)
    return np.concatenate(assembled, axis=0) if assembled else np.zeros(
        (1, 1), dtype=np.uint8)

  n_examples = min(max_examples, np.array(real_a).shape[0],
                   np.array(gen_b).shape[0], np.array(cyc_a).shape[0])

  # A -> B direction: real_a | gen_b | cyc_a
  grid_ab = _build_grid([real_a, gen_b, cyc_a], _to_uint8, n_examples)
  raw_ab = _add_image_summary('a_to_b', [grid_ab])

  # B -> A direction: real_b | gen_a | cyc_b
  grid_ba = _build_grid([real_b, gen_a, cyc_b], _to_uint8, n_examples)
  raw_ba = _add_image_summary('b_to_a', [grid_ba])

  raw_summaries = [raw_ab, raw_ba]

  # FFN visualizations.
  if ffn.model is not None and ffn.params is not None:
    ffn_pad = 0.5
    try:
      ffn_input_size = None
      ffn_config = ffn.config
      if ffn_config is not None and ffn_config.get('fov_size'):
        ffn_input_size = tuple(ffn_config.fov_size)[::-1]  # xyz -> zyx

      # preprocess() returns (prep_image, ffn_vis); we only need the
      # visualization (FFN output scaled to [0, 255]). Reusing it keeps the
      # visualization in sync with what the discriminator actually sees.
      _, ffn_real_a_vis = preprocess(
          jnp.array(real_a), ffn.model, ffn.params, ffn.batch_stats,
          'mask', ffn_pad, ffn_input_size)
      ffn_real_a_vis = np.array(ffn_real_a_vis)

      _, ffn_gen_a_vis = preprocess(
          jnp.array(gen_a), ffn.model, ffn.params, ffn.batch_stats,
          'mask', ffn_pad, ffn_input_size)
      ffn_gen_a_vis = np.array(ffn_gen_a_vis)

      n_ffn = min(max_examples, ffn_real_a_vis.shape[0],
                  ffn_gen_a_vis.shape[0])
      grid_ffn = _build_grid(
          [ffn_real_a_vis, ffn_gen_a_vis], _to_uint8_raw, n_ffn)
      raw_ffn = _add_image_summary('ffn_real_a_vs_gen_a', [grid_ffn])
      raw_summaries.append(raw_ffn)
    except Exception as e:  # pylint:disable=broad-except
      logging.warning('FFN visualization failed at step %d: %s', step, e)

  with tfw._summary_writer.as_default():  # pylint:disable=protected-access
    for s in raw_summaries:
      tf.summary.experimental.write_raw_pb(s, step=step)


def _get_tf_writer(writers) -> Any | None:
  """Extracts the TF SummaryWriter from a CLU writer."""
  # pylint:disable=protected-access
  if not hasattr(writers, '_writers'):
    return None
  for w in writers._writers:
    assert isinstance(w, metric_writers.AsyncWriter)
    if isinstance(w._writer, metric_writers.SummaryWriter):
      return w._writer
  # pylint:enable=protected-access
  return None


def _check_intensity_inversion(
    gen_b: np.ndarray,
    cyc_a: np.ndarray,
    step: int,
) -> bool:
  """Detects intensity inversion in the SECGAN.

  Inversion occurs when the model learns A->B: x -> -x and B->A: x -> -x.
  This satisfies the cycle-consistency constraint perfectly but produces
  useless results. Detected early in training; model should be restarted.

  Args:
    gen_b: Generated B image.
    cyc_a: Cycle-reconstructed A image.
    step: Current training step.

  Returns:
    True if intensity inversion is detected.
  """
  if step < 300 or step > 2000:
    return False

  gen_b_flat = gen_b.ravel()
  cyc_a_flat = cyc_a.ravel()

  # Use indices within each array's own bounds.
  # Check if the min/max relationship is inverted between the two images.
  min_size = min(len(gen_b_flat), len(cyc_a_flat))
  gen_b_flat = gen_b_flat[:min_size]
  cyc_a_flat = cyc_a_flat[:min_size]

  a_min = np.argmin(gen_b_flat)
  a_max = np.argmax(gen_b_flat)

  return bool(cyc_a_flat[a_min] > cyc_a_flat[a_max])


def _normalize_volume(ex: volume.Example) -> volume.Example:
  """Normalizes a uint8 'em' volume to [-1, 1] and stores it under 'image'."""
  em = tf.cast(ex['em'], tf.float32)
  em = em / 127.5 - 1.0
  return dict(ex, image=em)


def _build_volume_input_config(
    paths: dict[str, str],
    sampling: volume.SamplingConfig,
    patch_size_xyz: tuple[int, int, int],
    config: ml_collections.ConfigDict,
) -> volume.InputConfig:
  """Builds a volume.InputConfig shared by the dataset loaders."""
  return volume.InputConfig(
      sampling=sampling,
      volumes={
          'em': volume.VolumeConfig(
              paths=paths,
              load_shape=patch_size_xyz,
              filter_shape=patch_size_xyz,
          )
      },
      augmentation=volume.AugmentationConfig(
          permutable_axes=config.get('permutable_axes'),
          reflectable_axes=config.get('reflectable_axes'),
          contrast_factor_range=config.get('contrast_factor_range'),
          brightness_factor_range=config.get('brightness_factor_range'),
          apply_adjustment_to=config.get('apply_adjustment_to'),
      ),
  )


def _create_dataset_from_volinfo(
    volinfo_paths: Sequence[str],
    bboxes_txt: Sequence[str | None] | None,
    patch_size_xyz: tuple[int, int, int],
    rng: jax.Array,
    config: ml_collections.ConfigDict,
) -> tf.data.Dataset:
  """Loads volumetric data patches and normalizes them."""
  if bboxes_txt is None:
    bboxes_txt = [None] * len(volinfo_paths)
  if len(volinfo_paths) != len(bboxes_txt):
    raise ValueError(
        'volinfo_paths and bboxes_txt must be same length. (Multiple bboxes '
        'on a single volinfo should be specified as a single textproto with '
        'multiple box entries.)')

  paths = {}
  bounding_boxes = {}
  for i, (path, bbox_txt) in enumerate(zip(volinfo_paths, bboxes_txt)):
    volname = f'vol_{i}'
    paths[volname] = path
    from connectomics.common import bounding_box
    if bbox_txt is not None:
      boxes = [bounding_box.from_json(bbox_txt)]
    else:
      boxes = []
    bounding_boxes[volname] = boxes

  sampling = volume.SamplingConfig(bounding_boxes=bounding_boxes)
  cfg = _build_volume_input_config(paths, sampling, patch_size_xyz, config)

  per_host_batch_size = config.per_device_batch_size * jax.local_device_count()
  loader, _ = volume.grain_load_and_augment_subvolumes(
      cfg, np.array(rng), _normalize_volume, batch_size=per_host_batch_size
  )
  return loader.as_dataset(start_index=jax.process_index())


def create_input_pipeline(
    config: ml_collections.ConfigDict,
    gen_input_size_zyx: tuple[int, int, int],
    rng: jax.Array,
) -> tf.data.Dataset:
  """Creates the input pipeline for SECGAN training.

  Loads volumetric image patches from two datasets (A and B) and normalizes
  them to [-1, 1].

  Args:
    config: Training config with volume paths and bounding boxes.
    gen_input_size_zyx: Size of patches to extract (z, y, x).
    rng: PRNG key for dataset shuffling and augmentations.

  Returns:
    A tf.data.Dataset yielding dicts with 'image_a' and 'image_b' keys.
  """
  size_z, size_y, size_x = gen_input_size_zyx
  patch_size_xyz = (size_x, size_y, size_z)
  logging.info('Using patch_size_xyz = %r', patch_size_xyz)

  rng_a, rng_b = jax.random.split(rng)

  image_a_volinfo = _as_list(config.image_a_volinfo)
  image_b_volinfo = _as_list(config.image_b_volinfo)

  bbox_a = config.get('bbox_a', None)
  if bbox_a is not None:
    bbox_a = _as_list(bbox_a)
  bbox_b = config.get('bbox_b', None)
  if bbox_b is not None:
    bbox_b = _as_list(bbox_b)

  ds_a = _create_dataset_from_volinfo(
      image_a_volinfo, bbox_a, patch_size_xyz, rng_a, config
  )
  ds_b = _create_dataset_from_volinfo(
      image_b_volinfo, bbox_b, patch_size_xyz, rng_b, config
  )

  ds = tf.data.Dataset.zip((ds_a, ds_b))  # pyrefly: ignore[bad-argument-type]
  ds = ds.prefetch(128)
  return ds


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: epath.PathLike,
) -> None:
  """Main SECGAN training loop.

  Args:
    config: Training configuration.
    workdir: Directory for checkpoints and summaries.
  """
  workdir_path = epath.Path(workdir)
  if jax.process_index() == 0:
    workdir_path.mkdir(parents=True, exist_ok=True)

  rng = training.get_rng(config.seed)

  # Create models and optimizer.
  rng, model_rng = jax.random.split(rng)
  generator, discriminator, schedule, gen_tx, disc_tx, state = (
      create_train_state(config, model_rng)
  )

  # Load FFN if configured.
  ffn_model = None
  ffn_params = None
  ffn_batch_stats = None
  ffn_config = None

  if config.get('ffn_a_checkpoint'):
    if not config.get('ffn_config'):
      raise ValueError('ffn_config must be provided if ffn_a_checkpoint is set')
    ffn_config = config.ffn_config

    logging.info('Loading FFN checkpoint from %s', config.ffn_a_checkpoint)
    ffn_state = load_ffn_checkpoint(config.ffn_a_checkpoint, ffn_config)
    ffn_model = model_util.model_from_config(ffn_config)
    ffn_params = ffn_state.params
    ffn_batch_stats = ffn_state.batch_stats

  ffn_ctx = FfnContext(
      model=ffn_model,
      params=ffn_params,
      batch_stats=ffn_batch_stats,
      config=ffn_config,
  )

  gen_in_size_zyx = generator_input_size(config)
  logging.info('Generator input size (zyx): %r', gen_in_size_zyx)
  logging.info('Discriminator input size (zyx): %r',
               config.discriminator_input_size)

  # Checkpointing.
  checkpoint_dir = workdir_path / 'checkpoints'
  options_kwargs = {}
  if config.get('checkpoint_every_minutes'):
    options_kwargs['save_decision_policy'] = (
        ocp.checkpoint_managers.AnySavePolicy([
            ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
                minimum_interval_secs=int(config.checkpoint_every_minutes * 60)
            ),
            ocp.checkpoint_managers.PreemptionCheckpointingPolicy(),
        ])
    )
  else:
    options_kwargs['save_interval_steps'] = config.checkpoint_every_steps

  policies = []
  if config.get('keep_checkpoint_every_minutes'):
    policies.append(
        ocp.checkpoint_managers.EveryNSeconds(
            interval_secs=int(config.keep_checkpoint_every_minutes * 60)
        )
    )
  if config.get('max_checkpoints_to_keep') is not None:
    policies.append(
        ocp.checkpoint_managers.LatestN(n=config.max_checkpoints_to_keep)
    )
  if policies:
    options_kwargs['preservation_policy'] = (
        ocp.checkpoint_managers.AnyPreservationPolicy(policies)
    )

  checkpoint_options = ocp.CheckpointManagerOptions(**options_kwargs)
  checkpoint_manager = ocp.CheckpointManager(
      checkpoint_dir,
      item_names=('train_state',),
      options=checkpoint_options,
  )

  # Restore checkpoint if available.
  latest_step = checkpoint_manager.latest_step()
  if config.get('init_from_cpoint') and latest_step is None:
    handler = ocp.StandardCheckpointHandler()
    train_state_path = epath.Path(config.init_from_cpoint) / 'train_state'
    state = handler.restore(
        train_state_path, args=ocp.args.StandardRestore(state)
    )
    logging.info('Initializing from %r', config.init_from_cpoint)
  elif latest_step is not None:
    restore_args = {'train_state': ocp.args.StandardRestore(state)}
    checkpointed = checkpoint_manager.restore(
        latest_step, args=ocp.args.Composite(**restore_args))
    state = checkpointed['train_state']
    logging.info('Restored checkpoint for step %d', latest_step)
  else:
    logging.info('Starting training from scratch.')
    if jax.process_index() == 0:
      with tf.io.gfile.GFile(
          tf.io.gfile.join(str(workdir_path), 'config.json'), 'w'
      ) as f:
        f.write(config.to_json_best_effort() + '\n')

  state = jax.tree.map(np.array, state)
  initial_step = int(state.step) + 1

  # Build input pipeline.
  rng, input_rng = jax.random.split(rng)
  train_ds = create_input_pipeline(config, gen_in_size_zyx, input_rng)
  train_iter = iter(train_ds)

  # Setup JIT-compiled train step.
  mesh = Mesh(np.array(jax.devices()), ('batch',))
  replicate_sharding = NamedSharding(mesh, P())
  data_sharding = NamedSharding(mesh, P('batch'))

  def train_fn(state, real_a, real_b) -> tuple[Any, ...]:
    return full_train_step(
        generator=generator,
        discriminator=discriminator,
        state=state,
        gen_optimizer=gen_tx,
        disc_optimizer=disc_tx,
        schedule=schedule,
        real_a=real_a,
        real_b=real_b,
        config=config,
        ffn=ffn_ctx,
    )

  shard_in = (replicate_sharding, data_sharding, data_sharding)
  shard_out = (
      replicate_sharding,  # state
      replicate_sharding,  # metrics
      replicate_sharding,  # gen_a
      replicate_sharding,  # gen_b
      replicate_sharding,  # cyc_a
      replicate_sharding,  # cyc_b
  )
  p_train_step = jax.jit(
      train_fn, in_shardings=shard_in, out_shardings=shard_out
  )

  # Summary writer.
  writer = metric_writers.create_default_writer(
      workdir_path, just_logging=jax.process_index() > 0
  )
  if initial_step == 1:
    writer.write_hparams({
        k: v for k, v in config.items()
        if isinstance(v, (bool, float, int, str))
    })

  logging.info('Starting training loop at step %d.', initial_step)
  num_train_steps = config.max_steps
  hooks = []
  report_progress = training.ReportProgress(
      batch_size=1, num_train_steps=num_train_steps, writer=writer)
  if jax.process_index() == 0:
    hooks.append(report_progress)

  train_metrics = None
  shutdown_request = False
  timings = collections.defaultdict(list)

  with metric_writers.ensure_flushes(writer):
    writer.write_scalars(initial_step, {'start': 1})

    for step in range(initial_step, num_train_steps + 1):
      is_last_step = step == num_train_steps

      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        with report_progress.timed('input', wait_jax_async_dispatch=False):
          with training.MeasureTime(timings, 'data_load'):
            # Load data from the iterator.
            while True:
              ds_a, ds_b = next(train_iter)
              image_a = np.array(ds_a['image'])
              image_b = np.array(ds_b['image'])
              # Skip empty batches.
              if not (np.all(image_a == 0.0) or np.all(image_b == 0.0)):
                break

          # Move data to devices.
          if image_a.ndim == 6 and image_a.shape[1] == 1:
            image_a = np.squeeze(image_a, axis=1)
            image_b = np.squeeze(image_b, axis=1)

          # Each host has loaded per_device_batch_size * local_device_count
          # examples.  Split across local devices and assemble a globally-
          # sharded array so the mesh (which spans *all* devices) is satisfied.
          local_devices = jax.local_devices()
          local_a = np.split(image_a, len(local_devices))
          local_b = np.split(image_b, len(local_devices))
          per_device_a = [jax.device_put(x, d)
                          for x, d in zip(local_a, local_devices)]
          per_device_b = [jax.device_put(x, d)
                          for x, d in zip(local_b, local_devices)]
          global_batch = config.per_device_batch_size * jax.device_count()
          global_shape_a = (global_batch,) + image_a.shape[1:]
          global_shape_b = (global_batch,) + image_b.shape[1:]
          real_a = jax.make_array_from_single_device_arrays(
              global_shape_a, data_sharding, per_device_a)
          real_b = jax.make_array_from_single_device_arrays(
              global_shape_b, data_sharding, per_device_b)

          with training.MeasureTime(timings, 'train_step'):
            state, metrics_update, gen_a, gen_b, cyc_a, cyc_b = (
                p_train_step(state, real_a, real_b))

        with training.MeasureTime(timings, 'metrics'):
          train_metrics = (
              metrics_update if train_metrics is None
              else train_metrics.merge(metrics_update))

      # Intensity inversion check.
      if step >= 300 and step <= 2000 and step % 100 == 0:
        gen_b_np = np.array(gen_b)
        cyc_a_np = np.array(cyc_a)
        if _check_intensity_inversion(gen_b_np, cyc_a_np, step):
          logging.error(
              'Detected intensity inversion at step %d. Restarting training.',
              step,
          )
          rng, model_rng = jax.random.split(rng)
          _, _, _, _, _, fresh_state = create_train_state(config, model_rng)
          state = state.replace(
              gen_params=fresh_state.gen_params,
              gen_opt_state=fresh_state.gen_opt_state,
              disc_a_params=fresh_state.disc_a_params,
              disc_a_opt_state=fresh_state.disc_a_opt_state,
              disc_b_params=fresh_state.disc_b_params,
              disc_b_opt_state=fresh_state.disc_b_opt_state,
              disc_a_raw_params=fresh_state.disc_a_raw_params,
              disc_a_raw_opt_state=fresh_state.disc_a_raw_opt_state,
          )

      with training.MeasureTime(timings, 'admin'):
        if checkpoint_manager.should_save(step) or is_last_step:
          logging.info('Saving checkpoint at %d.', step)
          train_state = jax.tree.map(np.array, state)
          checkpoint_manager.save(
              step,
              args=ocp.args.Composite(
                  train_state=ocp.args.StandardSave(train_state)))

        if checkpoint_manager.reached_preemption(step):
          logging.warning(
              'Interrupting training loop due to shutdown request.')
          logging.flush()
          shutdown_request = True
          break

        for h in hooks:
          h(step)

        if step % config.log_loss_every_steps == 0 or is_last_step:
          scalars = train_metrics.compute()
          for name, values in timings.items():
            scalars[f'time_{name}'] = float(np.mean(values))
          timings = collections.defaultdict(list)

          # Domain-A adversarial losses live in mode-specific slots (img/ffn/
          # both); the slots that are inactive for this ffn_mode are constant
          # zero, so drop them to avoid cluttering TensorBoard with flat curves.
          ffn_active = config.get('ffn_a_checkpoint') is not None
          ffn_mode = config.get('ffn_mode', 'mask')
          if not ffn_active:
            active_slots = {'img'}
          elif ffn_mode == 'both':
            active_slots = {'both'}
          elif ffn_mode == 'separate':
            active_slots = {'ffn', 'img'}
          else:  # 'mask'
            active_slots = {'ffn'}
          for kind in ('img', 'ffn', 'both'):
            if kind not in active_slots:
              for name in (f'gen_b_{kind}_loss',
                           f'disc_a_real_{kind}_loss',
                           f'disc_a_fake_{kind}_loss'):
                scalars.pop(name, None)

          writer.write_scalars(step, scalars)

          # Write image summaries.
          if jax.process_index() == 0:
            visualize(writer, step, image_a, image_b,
                      np.array(gen_a), np.array(gen_b),
                      np.array(cyc_a), np.array(cyc_b),
                      ffn=ffn_ctx)

          train_metrics = None

  checkpoint_manager.wait_until_finished()
  logging.info('Finished training at step %d.', step)

  if shutdown_request:
    time.sleep(60)
    os._exit(42)  # pylint:disable=protected-access
