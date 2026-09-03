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

"""JAX/Flax implementations of SECGAN generator and discriminator networks.

Ported from the TensorFlow implementations in:
  research/neuromancer/tensorflow/transfer/cyclegan/models/

Generators translate image content between two domains and are fully
convolutional networks producing outputs in [-1, 1].

Discriminators classify image patches as real or fake, outputting logits.

All models expect 5D input tensors of shape [batch, z, y, x, channels].
"""

from flax import struct
import flax.linen as nn


# =============================================================================
# Generator
# =============================================================================


@struct.dataclass
class ResNetGeneratorConfig:
  """Config for the ResNet-style fully convolutional generator.

  Attributes:
    depth: number of residual modules.
    features: number of feature maps in convolutional layers.
    out_channels: number of channels in the output image.
  """

  depth: int = 4
  features: int = 32
  out_channels: int = 1


class ResNetGenerator(nn.Module):
  """ResNet-style, fully convolutional generator.

  Uses VALID padding so the output is spatially smaller than the input.
  The single-sided size reduction per axis is ``2 * depth``.  Output values
  are in [-1, 1] (tanh activation).

  Ported from ``cyclegan.models.generators.ResNet``.
  """

  config: ResNetGeneratorConfig

  @property
  def out_delta(self) -> int:
    """Single-sided size reduction per spatial axis (same for z, y, x).

    Each residual block contains two 3x3 VALID convolutions, each removing
    1 voxel per side, for a total of ``2 * depth`` voxels per side.
    """
    return 2 * self.config.depth

  @nn.compact
  def __call__(self, x):
    """Forward pass.

    Args:
      x: input tensor, shape ``[batch, z, y, x, channels]``.

    Returns:
      Translated image, shape ``[batch, z', y', x', out_channels]`` with
      values in [-1, 1].  The spatial dimensions are reduced by
      ``2 * depth`` on each side relative to the input.
    """
    cfg = self.config

    # Initial conv pair (no residual connection).
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv0_a',
    )(x)
    x = nn.relu(x)
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv0_b',
    )(x)

    # Residual blocks (pre-activation).
    for i in range(1, cfg.depth):
      in_net = x
      x = nn.relu(x)
      x = nn.Conv(
          features=cfg.features,
          kernel_size=(3, 3, 3),
          padding='VALID',
          name=f'conv{i}_a',
      )(x)
      x = nn.relu(x)
      x = nn.Conv(
          features=cfg.features,
          kernel_size=(3, 3, 3),
          padding='VALID',
          name=f'conv{i}_b',
      )(x)
      # Crop shortcut to match VALID-padded output.
      x = x + in_net[:, 2:-2, 2:-2, 2:-2, :]

    # Final pointwise conv with tanh.
    x = nn.relu(x)
    x = nn.Conv(
        features=cfg.out_channels,
        kernel_size=(1, 1, 1),
        padding='VALID',
        name='conv_point',
    )(x)
    x = nn.tanh(x)
    return x


# =============================================================================
# Discriminators
# =============================================================================


@struct.dataclass
class ConvPoolConfig:
  """Config for the anisotropic conv-pooling discriminator.

  Attributes:
    min_input_size: minimum zyx spatial dimensions so that the output is a
      single value.
    features: number of feature maps in convolutional layers.
  """

  min_input_size: tuple[int, int, int] = (17, 33, 33)
  features: int = 32


class ConvPool(nn.Module):
  """Anisotropic conv-pooling discriminator.

  Assumes approximately 1:2:2 ZYX resolution anisotropy. All operations use
  VALID padding.

  Ported from ``cyclegan.models.discriminators.ConvPool``.
  """

  config: ConvPoolConfig

  @nn.compact
  def __call__(self, x):
    """Forward pass.

    Args:
      x: input tensor, shape ``[batch, z, y, x, channels]``.

    Returns:
      Discriminator logits.
    """
    cfg = self.config

    # [z, y, x] -> conv -> [z-2, y-2, x-2]
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv0',
    )(x)
    x = nn.relu(x)

    # Anisotropic pooling: downsample only y, x.
    x = nn.max_pool(
        x, window_shape=(1, 2, 2), strides=(1, 2, 2), padding='VALID'
    )

    # conv -> pool(2,2,2)
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv1',
    )(x)
    x = nn.relu(x)
    x = nn.max_pool(
        x, window_shape=(2, 2, 2), strides=(2, 2, 2), padding='VALID'
    )

    # conv
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv2',
    )(x)
    x = nn.relu(x)

    # Final conv -> logits (no activation).
    x = nn.Conv(
        features=1,
        kernel_size=(4, 4, 4),
        padding='VALID',
        name='conv_out',
    )(x)
    return x


@struct.dataclass
class IsoConvPoolConfig(ConvPoolConfig):
  """ConvPoolConfig variant with an isotropic default ``min_input_size``."""

  min_input_size: tuple[int, int, int] = (33, 33, 33)


class IsoConvPool(nn.Module):
  """Isotropic conv-pooling discriminator.

  All operations use VALID padding with isotropic kernels and strides.

  Ported from ``cyclegan.models.discriminators.IsoConvPool``.
  """

  config: IsoConvPoolConfig

  @nn.compact
  def __call__(self, x):
    """Forward pass.

    Args:
      x: input tensor, shape ``[batch, z, y, x, channels]``.

    Returns:
      Discriminator logits.
    """
    cfg = self.config

    # conv -> pool(2,2,2)
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv0',
    )(x)
    x = nn.relu(x)
    x = nn.max_pool(
        x, window_shape=(2, 2, 2), strides=(2, 2, 2), padding='VALID'
    )

    # conv -> pool(2,2,2)
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv1',
    )(x)
    x = nn.relu(x)
    x = nn.max_pool(
        x, window_shape=(2, 2, 2), strides=(2, 2, 2), padding='VALID'
    )

    # conv
    x = nn.Conv(
        features=cfg.features,
        kernel_size=(3, 3, 3),
        padding='VALID',
        name='conv2',
    )(x)
    x = nn.relu(x)

    # Final conv -> logits (no activation).
    x = nn.Conv(
        features=1,
        kernel_size=(4, 4, 4),
        padding='VALID',
        name='conv_out',
    )(x)
    return x


@struct.dataclass
class ResNet18Config:
  """Config for the ResNet-18-like discriminator.

  Uses the pre-activation variant of residual modules.

  Attributes:
    min_input_size: minimum zyx spatial dimensions so that the output is a
      single value.
    input_stride: zyx strides applied in the second convolution of the input
      module; can be used to downsample anisotropic data toward isotropy.
  """

  min_input_size: tuple[int, int, int] = (33, 33, 33)
  input_stride: tuple[int, int, int] = (1, 1, 1)


def _conv_module(
    data_in,
    features,
    layer_id,
    stride=(1, 1, 1),
    stride2=(1, 1, 1),
    preactivate=True,
):
  """Two-conv (pre-activation) sub-block used by ``ResNet18``.

  Does not close over any module state, so it lives at module scope; the
  ``nn.Conv`` submodules it creates are still attached to the calling module.

  Args:
    data_in: input tensor, shape ``[batch, z, y, x, channels]``.
    features: number of feature maps in both convolutions.
    layer_id: identifier used to name the two convolution submodules.
    stride: zyx strides for the first convolution.
    stride2: zyx strides for the second convolution.
    preactivate: if True, apply a ReLU to ``data_in`` before the first conv.

  Returns:
    Output tensor after the two convolutions.
  """
  if preactivate:
    data_in = nn.relu(data_in)

  net = nn.Conv(
      features=features,
      kernel_size=(3, 3, 3),
      strides=stride,
      padding='SAME',
      name=f'conv{layer_id}a',
  )(data_in)
  net = nn.relu(net)
  net = nn.Conv(
      features=features,
      kernel_size=(3, 3, 3),
      strides=stride2,
      padding='SAME',
      name=f'conv{layer_id}b',
  )(net)
  return net


class ResNet18(nn.Module):
  """ResNet-18-like classification discriminator (pre-activation variant).

  Uses SAME padding throughout.  The network has four stages with
  progressively increasing feature counts (32 -> 64 -> 128 -> 256) and
  stride-2 transitions between stages.

  More info about the original model: https://arxiv.org/abs/1512.03385
  Pre-activation variant: https://arxiv.org/abs/1603.05027

  Ported from ``cyclegan.models.discriminators.ResNet18``.
  """

  config: ResNet18Config

  @nn.compact
  def __call__(self, x):
    """Forward pass.

    Args:
      x: input tensor, shape ``[batch, z, y, x, channels]``.

    Returns:
      Discriminator logits.
    """
    cfg = self.config

    # --- Input module (32 features) ---
    in_net = _conv_module(
        x, 32, 0, stride2=cfg.input_stride, preactivate=False
    )

    # --- Stage 1: 32 features, residual ---
    net = _conv_module(in_net, 32, 1) + in_net

    # --- Stage 2: 64 features, stride-2 transition ---
    in_net = nn.Conv(
        features=64,
        kernel_size=(1, 1, 1),
        strides=(2, 2, 2),
        padding='SAME',
        name='skip1',
    )(net)
    net = _conv_module(net, 64, 2, stride=(2, 2, 2), preactivate=True)
    net = net + in_net
    net = _conv_module(net, 64, 3) + net

    # --- Stage 3: 128 features, stride-2 transition ---
    in_net = nn.Conv(
        features=128,
        kernel_size=(1, 1, 1),
        strides=(2, 2, 2),
        padding='SAME',
        name='skip2',
    )(net)
    net = _conv_module(net, 128, 4, stride=(2, 2, 2), preactivate=True)
    net = net + in_net
    net = _conv_module(net, 128, 5) + net

    # --- Stage 4: 256 features, stride-2 transition ---
    in_net = nn.Conv(
        features=256,
        kernel_size=(1, 1, 1),
        strides=(2, 2, 2),
        padding='SAME',
        name='skip3',
    )(net)
    net = _conv_module(net, 256, 6, stride=(2, 2, 2), preactivate=True)
    net = net + in_net
    net = _conv_module(net, 256, 7) + net

    # --- Output: pointwise conv -> logits ---
    net = nn.Conv(
        features=1,
        kernel_size=(1, 1, 1),
        padding='SAME',
        name='conv_out',
    )(net)
    return net

