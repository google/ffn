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
"""Simplest FFN model, as described in https://arxiv.org/abs/1611.00421."""

import functools
import itertools
import tensorflow.compat.v1 as tf
import tf_slim
from .. import model


# Note: this model was originally trained with conv3d layers initialized with
# TruncatedNormalInitializedVariable with stddev = 0.01.
def _predict_object_mask(net, depth=9, features=32):
  """Computes single-object mask prediction."""
  conv = tf_slim.convolution3d
  conv = functools.partial(
      tf_slim.convolution3d, kernel_size=(3, 3, 3), padding='SAME'
  )

  if isinstance(features, int):
    feats = itertools.repeat(features)
  else:
    feats = iter(features)

  net = conv(net, scope='conv0_a', num_outputs=next(feats))
  net = conv(net, scope='conv0_b', activation_fn=None, num_outputs=next(feats))

  for i in range(1, depth):
    with tf.name_scope('residual%d' % i):
      in_net = net
      net = tf.nn.relu(net)
      net = conv(net, scope='conv%d_a' % i, num_outputs=next(feats))
      net = conv(
          net, scope='conv%d_b' % i, activation_fn=None, num_outputs=next(feats)
      )
      net += in_net

  net = tf.nn.relu(net)
  logits = tf_slim.convolution3d(
      net, 1, (1, 1, 1), activation_fn=None, scope='conv_lom'
  )

  return logits


class ConvStack3DFFNModel(model.FFNModel):
  """A simple conv-stack FFN model.

  The model is composed of `depth` residual modules, operating at a
  constant spatial resolution.
  """

  dim = 3

  def __init__(
      self,
      fov_size=None,
      deltas=None,
      batch_size=None,
      depth: int = 9,
      features: int = 32,
      **kwargs
  ):
    info = model.ModelInfo(deltas, fov_size, fov_size, fov_size)
    super().__init__(info, batch_size, **kwargs)
    self.set_input_shapes()
    self.depth = depth
    self.features = features

  def define_tf_graph(self):
    self.show_center_slice(self.input_seed)

    net = tf.concat([self.input_patches, self.input_seed], 4)

    with tf.variable_scope('seed_update', reuse=False):
      logit_update = _predict_object_mask(net, self.depth, self.features)

    logit_seed = self.update_seed(self.input_seed, logit_update)

    # Make predictions available, both as probabilities and logits.
    self.logits = logit_seed
    self.logistic = tf.sigmoid(logit_seed)

    if self.labels is not None:
      self.set_up_sigmoid_pixelwise_loss(logit_seed)
      self.set_up_optimizer()
      self.show_center_slice(logit_seed)
      self.show_center_slice(self.labels, sigmoid=False)
      self.add_summaries()
