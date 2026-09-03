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

from absl.testing import absltest
from ffn.secgan import models
import jax
import jax.numpy as jnp


class ModelsTest(absltest.TestCase):

  def test_resnet_generator(self):
    config = models.ResNetGeneratorConfig(depth=4, features=16)
    model = models.ResNetGenerator(config=config)
    rng = jax.random.PRNGKey(0)
    # Input: [batch, z, y, x, channels]
    # With depth=4, out_delta = 8, so input of 33 -> output of 17
    x = jnp.ones((1, 33, 33, 33, 1))
    variables = model.init(rng, x)
    y = model.apply(variables, x)
    self.assertEqual(y.shape, (1, 17, 17, 17, 1))
    # Output should be in [-1, 1] due to tanh.
    self.assertLessEqual(float(jnp.max(y)), 1.0)
    self.assertGreaterEqual(float(jnp.min(y)), -1.0)

  def test_resnet_generator_out_delta(self):
    gen4 = models.ResNetGenerator(
        config=models.ResNetGeneratorConfig(depth=4))
    self.assertEqual(gen4.out_delta, 8)
    gen8 = models.ResNetGenerator(
        config=models.ResNetGeneratorConfig(depth=8))
    self.assertEqual(gen8.out_delta, 16)

  def test_convpool_discriminator(self):
    config = models.ConvPoolConfig(min_input_size=(17, 33, 33), features=16)
    model = models.ConvPool(config=config)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 17, 33, 33, 1))
    variables = model.init(rng, x)
    y = model.apply(variables, x)
    # Output should be a small spatial volume.
    self.assertEqual(y.shape[0], 1)  # batch
    self.assertEqual(y.shape[-1], 1)  # single output channel

  def test_iso_convpool_discriminator(self):
    config = models.IsoConvPoolConfig(min_input_size=(33, 33, 33), features=16)
    model = models.IsoConvPool(config=config)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 33, 33, 33, 1))
    variables = model.init(rng, x)
    y = model.apply(variables, x)
    self.assertEqual(y.shape[0], 1)
    self.assertEqual(y.shape[-1], 1)

  def test_resnet18_discriminator(self):
    config = models.ResNet18Config(min_input_size=(33, 33, 33))
    model = models.ResNet18(config=config)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 33, 33, 33, 1))
    variables = model.init(rng, x)
    y = model.apply(variables, x)
    self.assertEqual(y.shape[0], 1)
    self.assertEqual(y.shape[-1], 1)


if __name__ == '__main__':
  absltest.main()
