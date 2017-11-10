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
"""Utilites for dealing with 2d and 3d object masks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# TODO(mjanusz): Consider integrating this with the numpy-only crop_and_pad,
# with dynamic dispatch to helper functions based on type of the data argument.
def crop(tensor, offset, crop_shape):
  """Extracts 'crop_shape' around 'offset' from 'tensor'.

  Args:
    tensor: tensor to extract data from (b, [z], y, x, c)
    offset: (x, y, [z]) offset from the center point of 'tensor'; center is
        taken to be 'shape // 2'
    crop_shape: (x, y, [z]) shape to extract

  Returns:
    cropped tensor
  """
  with tf.name_scope('offset_crop'):
    shape = tensor.shape_as_list()

    # Nothing to do?
    if shape[1:-1] == crop_shape[::-1]:
      return tensor

    off_y = shape[-3] // 2 - crop_shape[1] // 2 + offset[1]
    off_x = shape[-2] // 2 - crop_shape[0] // 2 + offset[0]

    if len(offset) == 2:
      cropped = tensor[:,
                       off_y:(off_y + crop_shape[1]),
                       off_x:(off_x + crop_shape[0]),
                       :]
    else:
      off_z = shape[-4] // 2 - crop_shape[2] // 2 + offset[2]
      cropped = tensor[:,
                       off_z:(off_z + crop_shape[2]),
                       off_y:(off_y + crop_shape[1]),
                       off_x:(off_x + crop_shape[0]),
                       :]
    return cropped


# Functions operating on Numpy arrays below this line.
##############################################################################


def update_at(to_update, offset, new_value, valid=None):
  """Pastes 'new_value' into 'to_update'.

  Args:
    to_update: numpy array to update (b, [z], y, x, c)
    offset: (x, y, [z]) offset from the center point of 'to_update',
        at which to locate the center of 'new_value'. Center is taken to
        be 'shape // 2'.
    new_value: numpy array with values to paste (b, [z], y, x, c)
    valid: (optional) mask selecting values to be updated; typically a 1d bool
        mask over the batch dimension

  Returns:
    None. 'to_update' is modified in place.
  """
  # Spatial dimensions only. All vars in zyx.
  shape = np.array(to_update.shape[1:-1])
  crop_shape = np.array(new_value.shape[1:-1])
  offset = np.array(offset[::-1])

  start = shape // 2 - crop_shape // 2 + offset
  end = start + crop_shape

  assert np.all(start >= 0)

  selector = [slice(s, e) for s, e in zip(start, end)]
  selector = tuple([slice(None)] + selector + [slice(None)])

  if valid is not None:
    to_update[selector][valid] = new_value[valid]
  else:
    to_update[selector] = new_value


def crop_and_pad(data, offset, crop_shape, target_shape=None):
  """Extracts 'crop_shape' around 'offset' from 'data'.

  Optionally pads with zeros to 'target_shape'.

  Args:
    data: 4d/5d array (b, [z], y, x, c)
    offset: (x, y, [z]) offset from the center of 'data'. Center is taken to
        be 'shape // 2'
    crop_shape: ([cz], cy, cx) shape to extract
    target_shape: optional ([tz], ty, tx) shape to return; if specified,
        the cropped data is padded with zeros symmetrically on all sides
        in order to expand it to the target shape. If padding is odd,
        the padding on the left is 1 shorter than the one on the right.

  Returns:
    Extracted array as (b, tz, ty, tx, c) if target_shape is specified
        or (b, cz, cy, cx, c) if only crop_shape is given.
  """
  # Spatial dimensions only. All vars in zyx.
  shape = np.array(data.shape[1:-1])
  crop_shape = np.array(crop_shape)
  offset = np.array(offset[::-1])

  start = shape // 2 - crop_shape // 2 + offset
  end = start + crop_shape

  assert np.all(start >= 0)

  selector = [slice(s, e) for s, e in zip(start, end)]
  selector = tuple([slice(None)] + selector + [slice(None)])
  cropped = data[selector]

  if target_shape is not None:
    target_shape = np.array(target_shape)
    delta = target_shape - crop_shape
    pre = delta // 2
    post = delta - delta // 2

    paddings = [(0, 0)]  # no padding for batch
    paddings.extend(zip(pre, post))
    paddings.append((0, 0))  # no padding for channels

    cropped = np.pad(cropped, paddings, mode='constant')

  return cropped


def make_seed(shape, batch_size, pad=0.05, seed=0.95):
  """Builds a numpy array with a single voxel seed in the center.

  Center is taken to be 'shape // 2'.

  Args:
    shape: spatial size of the seed array (z, y, x)
    batch_size: batch dimension size
    pad: value used where the seed is inactive
    seed: value used where the seed is active

  Returns:
    float32 ndarray of shape [b, z, y, x] with the seed
  """
  seed_array = np.full([batch_size] + list(shape) + [1], pad, dtype=np.float32)
  idx = tuple([slice(None)] + list(np.array(shape) // 2))
  seed_array[idx] = seed
  return seed_array

