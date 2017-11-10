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
"""Utility functions to display axis orthogonal slices from 3d volumes.

* Cutting through specified center location
* Assembling slices to a single image diplay
* NaN-aware image color normalization
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import expit as sigmoid


def cut_ortho_planes(vol, center=None, cross_hair=False):
  """Cuts 3 axis orthogonal planes from a 3d volume.

  Args:
    vol: zyx(c) 3d volume array
    center: coordinate triple where the planes intersect, if none the volume
      center is used (vol.shape//2)
    cross_hair: boolean, inserts transparent cross hair lines through
      center point

  Returns:
    planes: list of 3 2d (+channel optional) images. Can be assembled to a
    single image display using ``concat_ortho_planes``. The order of planes is
    yx, zx, zy.
  """
  if center is None:
    center = np.array(vol.shape[:3]) // 2

  planes = []
  full_slice = [slice(None)] * 3
  for axis, ix in enumerate(center):
    cut_slice = list(full_slice)
    cut_slice[axis] = ix
    planes.append(vol[cut_slice])
    if cross_hair:
      # Copy because cross hair is written into array data.
      plane = planes[-1].copy()
      i = 0
      for ax, c in enumerate(center):
        if ax != axis:
          # Make axis i the 0-axis an work in-place.
          view = np.rollaxis(plane, i)
          view[c] *= 0.5
          i += 1

      planes[-1] = plane

  return planes


def concat_ortho_planes(planes):
  """Concatenates 3 axis orthogonal planes to a single image display.

  Args:
    planes: list of 3 2d (+channel optional) planes as obtained
      from ``cut_ortho_planes``. The order of planes must be
      yx, zx, zy.

  Returns:
    image: 2d (+channel optional) array
  """
  assert len(planes) == 3

  h_yx, w_yx = planes[0].shape[0], planes[0].shape[1]
  h_zx, w_zx = planes[1].shape[0], planes[1].shape[1]
  h_zy, w_zy = planes[2].shape[1], planes[2].shape[0]

  assert h_yx == h_zy
  assert w_yx == w_zx
  assert h_zx == w_zy

  height = h_yx + 1 + h_zx
  width = w_yx + 1 + w_zy
  channel = planes[0].shape[2:]
  ret = np.zeros((height, width) + channel, dtype=planes[0].dtype)

  # Insert yx plane in top left.
  ret[:h_yx, :w_yx] = planes[0]
  # Insert zx plane in bottom left.
  ret[-h_zx:, :w_zx] = planes[1]
  # Insert zy plane in top right, swap to align y-axis with main yx panel.
  ret[:h_zy, -w_zy:] = np.swapaxes(planes[2], 0, 1)

  return ret


def normalize_image(img2d, act=None):
  """Map unbounded grey image to [0,1]-RGB, r:negative, b:positive, g:nan.

  Args:
    img2d: (x,y) image array, channels are not supported.
    act: ([None]|'tanh'|'sig') optional activation function to scale grey
      values. None means normalized between min and 0 for negative values and
      between 0 and max for positive values.

  Returns:
    img_rgb: (x,y,3) image array
  """
  nan_mask = np.isnan(img2d)
  img2d[nan_mask] = 0
  m, mm = img2d.min(), img2d.max()
  img_rgb = np.zeros(img2d.shape + (3,), dtype=np.float32)
  if act == 'tanh':
    img_rgb[~nan_mask, 0] = np.tanh(np.clip(img2d, m, 0))[~nan_mask]
    img_rgb[~nan_mask, 2] = np.tanh(np.clip(img2d, 0, mm))[~nan_mask]
  elif act == 'sig':
    img_rgb[~nan_mask, 0] = sigmoid(img2d[~nan_mask])
    img_rgb[~nan_mask, 2] = img_rgb[~nan_mask, 0]
  else:
    img_rgb[~nan_mask, 0] = (np.clip(img2d, m, 0) / m)[~nan_mask]
    img_rgb[~nan_mask, 2] = (np.clip(img2d, 0, mm) / mm)[~nan_mask]

  img_rgb[nan_mask, 1] = 1.0
  return img_rgb
