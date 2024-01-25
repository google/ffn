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
"""Utilities to configure TF optimizers."""

from absl import flags
import tensorflow.compat.v1 as tf

_OPTIMIZER = flags.DEFINE_enum(
    'optimizer',
    'sgd',
    ['momentum', 'sgd', 'adagrad', 'adam', 'rmsprop'],
    'Which optimizer to use.',
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.001, 'Initial learning rate.'
)
_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum.')
_LEARNING_RATE_DECAY_FACTOR = flags.DEFINE_float(
    'learning_rate_decay_factor', None, 'Learning rate decay factor.'
)
_DECAY_STEPS = flags.DEFINE_integer(
    'decay_steps',
    None,
    (
        'How many steps the model needs to train for in order for '
        'the decay factor to be applied to the learning rate.'
    ),
)
_NUM_EPOCHS_PER_DECAY = flags.DEFINE_float(
    'num_epochs_per_decay',
    2.0,
    'Number of epochs after which learning rate decays.',
)
_RMSPROP_DECAY = flags.DEFINE_float(
    'rmsprop_decay', 0.9, 'Decay term for RMSProp.'
)
_ADAM_BETA1 = flags.DEFINE_float(
    'adam_beta1', 0.9, 'Gradient decay term for Adam.'
)
_ADAM_BETA2 = flags.DEFINE_float(
    'adam_beta2', 0.999, 'Gradient^2 decay term for Adam.'
)
_EPSILON = flags.DEFINE_float(
    'epsilon', 1e-8, 'Epsilon term for RMSProp and Adam.'
)
_SYNC_SGD = flags.DEFINE_boolean(
    'sync_sgd', False, 'Whether to use synchronous SGD.'
)
_REPLICAS_TO_AGGREGATE = flags.DEFINE_integer(
    'replicas_to_aggregate',
    None,
    'When using sync SGD, over how many replicas to aggregate the gradients.',
)
_TOTAL_REPLICAS = flags.DEFINE_integer(
    'total_replicas',
    None,
    'When using sync SGD, total number of replicas in the training pool.',
)


def _optimizer_from_flags():
  """Defines a TF optimizer based on flag settings."""
  lr = _LEARNING_RATE.value
  if (
      _LEARNING_RATE_DECAY_FACTOR.value is not None
      and _DECAY_STEPS.value is not None
  ):
    lr = tf.train.exponential_decay(
        _LEARNING_RATE.value,
        tf.train.get_or_create_global_step(),
        _DECAY_STEPS.value,
        _LEARNING_RATE_DECAY_FACTOR.value,
        staircase=True,
    )

  tf.summary.scalar('learning_rate', lr)

  if _OPTIMIZER.value == 'momentum':
    return tf.train.MomentumOptimizer(lr, _MOMENTUM.value)
  elif _OPTIMIZER.value == 'sgd':
    return tf.train.GradientDescentOptimizer(lr)
  elif _OPTIMIZER.value == 'adagrad':
    return tf.train.AdagradOptimizer(lr)
  elif _OPTIMIZER.value == 'adam':
    return tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=_ADAM_BETA1.value,
        beta2=_ADAM_BETA2.value,
        epsilon=_EPSILON.value,
    )
  elif _OPTIMIZER.value == 'rmsprop':
    return tf.train.RMSPropOptimizer(
        lr,
        _RMSPROP_DECAY.value,
        momentum=_MOMENTUM.value,
        epsilon=_EPSILON.value,
    )
  else:
    raise ValueError('Unknown optimizer: %s' % _OPTIMIZER.value)


def optimizer_from_flags():
  """Defines a TF optimizer based on command-line flags."""
  opt = _optimizer_from_flags()
  if _SYNC_SGD.value:
    assert _REPLICAS_TO_AGGREGATE.value is not None
    if _TOTAL_REPLICAS.value is not None:
      assert _TOTAL_REPLICAS.value >= _REPLICAS_TO_AGGREGATE.value

    return tf.train.SyncReplicasOptimizer(
        opt,
        replicas_to_aggregate=_REPLICAS_TO_AGGREGATE.value,
        total_num_replicas=_TOTAL_REPLICAS.value,
    )
  else:
    return opt
