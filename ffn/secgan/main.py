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

"""Main entry point for SECGAN training."""

from collections.abc import Sequence

from absl import app
from absl import flags
from connectomics.jax import training
from ffn.secgan import train
import jax

FLAGS = flags.FLAGS

training.define_training_flags()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Disable space-to-batch XLA transform which crashes on small spatial dims.
  if 'xla_tpu_run_space_to_batch' in flags.FLAGS:
    flags.FLAGS.xla_tpu_run_space_to_batch = False
  if 'xla_tpu_run_space_to_batch_on_new_platforms' in flags.FLAGS:
    flags.FLAGS.xla_tpu_run_space_to_batch_on_new_platforms = False

  training.prep_training()
  train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
