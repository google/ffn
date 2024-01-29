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
"""Simple augmentation operations for volumetric EM data."""

from typing import Optional

from connectomics.common import object_utils
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import AffineTransform
from skimage.transform import warp
import tensorflow.compat.v1 as tf
import tensorflow.google.compat.v1 as tf
from tf import transformations

from multidim_image_augmentation import augmentation_ops


def standard_rotation_matrix(
    mask: tf.Tensor, voxel_size: tuple[float, float, float]) -> tf.Operation:
  """Computes a rotation matrix to put an object into a standard orientation.

  In the standard orientation, the axis of highest variance is 'z', and the
  axis of 2nd highest variance is 'y'.

  Args:
    mask: shape-[1, z, y, x, 1] bool tensor representing a single object mask
    voxel_size: xyz-tuple defining the physical voxel size

  Returns:
    3x3 tensor representing the rotation matrix
  """

  def _compute_rot_mtx(mask, voxel_size=voxel_size):
    points = object_utils.mask_to_points(mask, voxel_size)
    eigvecs, _ = object_utils.compute_orientation(points)
    mtx = object_utils.compute_rotation_matrix(eigvecs)
    return mtx.astype(np.float32)

  ret = tf.py_func(
      func=_compute_rot_mtx, inp=[mask], Tout=tf.float32, name='std_rot_mtx')
  ret.set_shape([3, 3])
  return ret


def random_3d_rotation_matrix(
    uniform_variate: Optional[tf.Tensor] = None) -> tf.Operation:
  """Computes a random 3d rotation matrix.

  Args:
    uniform_variate: optional float tensor with variates from U[0, 1]; shape [3]

  Returns:
    3x3 tensor representing the random rotation matrix
  """

  def _random_3d_rot_mtx(var):
    return transformations.random_rotation_matrix(var).astype(
        np.float32)[:3, :3]

  if uniform_variate is None:
    uniform_variate = tf.random.uniform([3], 0, 1)

  ret = tf.py_func(
      func=_random_3d_rot_mtx,
      inp=[uniform_variate],
      Tout=tf.float32,
      name='rand_3d_rot_mtx')
  ret.set_shape([3, 3])
  return ret


def random_2d_rotation_matrix(
    uniform_variate: Optional[tf.Tensor] = None) -> tf.Operation:
  """Computes a matrix for a random rotation around the 'z' axis.

  Args:
    uniform_variate: optional float tensor with a variate from U[0, 1]; shape
      [1]

  Returns:
    3x3 tensor representing the rotation matrix
  """

  def _random_2d_rot_mtx(var):
    angle = var * 2 * np.pi
    return transformations.rotation_matrix(angle,
                                           [0, 0, 1]).astype(np.float32)[:3, :3]

  if uniform_variate is None:
    uniform_variate = tf.random.uniform([1], 0, 1)

  ret = tf.py_func(
      func=_random_2d_rot_mtx,
      inp=[uniform_variate],
      Tout=tf.float32,
      name='rand_2d_rot_mtx')
  ret.set_shape([3, 3])
  return ret


def input_size_for_rotated_output(
    desired_size: tuple[int, int, int],
    in_voxel_size: tuple[float, float, float],
    out_voxel_size: Optional[tuple[float, float, float]] = None) -> list[int]:
  """Computes the input size necessary for a given output size.

  The input size is computed to be large enough so that if an arbitrary
  rotation is applied, the output will only contain valid data.

  Args:
    desired_size: xyz size of the output in voxels
    in_voxel_size: xyz voxel size of the input
    out_voxel_size: xyz voxel size of the output; if not specified, assumed same
      as in_voxel_size

  Returns:
    minimum xyz size of the input in voxels
  """
  if out_voxel_size is None:
    out_voxel_size = in_voxel_size
  out_phys_size = np.array(desired_size) * out_voxel_size
  phys_r = np.max(out_phys_size) / 2.0 * np.sqrt(2.0)
  return np.ceil(2.0 * phys_r / in_voxel_size).astype(int).tolist()


def apply_rotation(data: tf.Tensor,
                   rotation_matrix: tf.Tensor,
                   in_voxel_size: tuple[float, float, float],
                   out_voxel_size: Optional[tuple[float, float, float]] = None,
                   interpolation='nearest') -> tf.Operation:
  """Applies a rotation to a tensor of volumetric data.

  Args:
    data: [1, z, y, x, 1] data to apply rotation to
    rotation_matrix: [3, 3] rotation matrix
    in_voxel_size: xyz voxel size of the input
    out_voxel_size: xyz voxel size of the output; if not specified, assumed same
      as in_voxel_size
    interpolation: interpolation method to use; passed to apply_deformation3d

  Returns:
    [1, z', y', x', 1] tensor with rotated `data`; note that the output
    size is smaller than the input size (see `input_size_for_rotated_output`)
  """
  in_voxel_size = np.asarray(in_voxel_size)
  if out_voxel_size is None:
    out_voxel_size = in_voxel_size

  shape = data.shape.as_list()
  assert len(shape) == 5
  in_phys_diam = np.array(shape[1:4][::-1]) * in_voxel_size

  # Minimum extent of physical space sampled by the voxels of the original data
  # defines the diameter of a sphere inscribed into the bounding box of the
  # input. The output subvolume is inscribed into this sphere.
  out_phys_diam = np.min(in_phys_diam) / np.sqrt(2.0)

  # Size of the output tensor in voxels.
  out_diam_vx = out_phys_diam // out_voxel_size  # xyz

  # Coordinates within the output tensor for which interpolated values will
  # be computed.
  hz, hy, hx = tf.meshgrid(
      tf.range(0, out_diam_vx[2]),
      tf.range(0, out_diam_vx[1]),
      tf.range(0, out_diam_vx[0]),
      indexing='ij')

  # Convert back to physical coordinates. Shift by half a voxel since the grid
  # coordinates are assumed to correspond to the center of the voxel.
  hz = (tf.cast(hz, tf.float32) + 0.5) * out_voxel_size[2]
  hy = (tf.cast(hy, tf.float32) + 0.5) * out_voxel_size[1]
  hx = (tf.cast(hx, tf.float32) + 0.5) * out_voxel_size[0]

  # Coordinate system change so that the origin of the rotated and source
  # data match.
  out_phys_r = (out_diam_vx * out_voxel_size) / 2
  hz -= out_phys_r[2]
  hy -= out_phys_r[1]
  hx -= out_phys_r[0]

  # Query vector of points to be rotated, in physical space. Note that the
  # rotation matrix transforms old coordinates to new coordinates. Here, we
  # are applying the inverse transformation (new coordinates in the canonical
  # system of the output tensor are known):
  #
  #  new = M.old
  #  M^T.new = old    // left multiply and use orthogonality of M
  #  new^T.M = old^T  // transpose
  points = tf.stack(
      [tf.reshape(hx, [-1]),
       tf.reshape(hy, [-1]),
       tf.reshape(hz, [-1])],
      axis=1)
  phys_coords = tf.matmul(points, rotation_matrix)

  # -0.5 because the origin of the physical coordinate system is a half a voxel
  # before the grid coordinate.
  in_phys_r = in_phys_diam / 2
  orig_coords = (phys_coords + in_phys_r) / in_voxel_size - 0.5

  # apply_deformation expects the coordinates to be zyx, which is the opposite
  # of what we used for the matrix transformation above.
  orig_coords = orig_coords[:, ::-1]
  orig_coords = tf.reshape(orig_coords, hx.shape.as_list() + [3])

  rotated = augmentation_ops.apply_deformation3d(
      data[0, ...],
      orig_coords,
      padding_constant=[],
      interpolation=interpolation)
  rotated = tf.cast(rotated, data.dtype)
  return tf.reshape(rotated, [1] + rotated.shape.as_list())


def reflection(data, decision):
  """Conditionally reflects the data in XYZ.

  Args:
    data: input tensor, shape: [..], z, y, x, c
    decision: boolean tensor, shape 3, indicating on which spatial dimensions
       to apply the reflection (x, y, z)

  Returns:
    TF op to conditionally apply reflection.
  """
  with tf.name_scope('augment_reflection'):
    rank = data.get_shape().ndims
    spatial_dims = tf.constant([rank - 2, rank - 3, rank - 4])
    selected_dims = tf.boolean_mask(spatial_dims, decision)
    return tf.reverse(data, selected_dims)


def xy_transpose(data, decision):
  """Conditionally transposes the X and Y axes of a tensor.

  Args:
    data: input tensor, shape: [..], y, x, c.
    decision: boolean scalar indicating whether to apply the transposition

  Returns:
    TF op to conditionally apply XY transposition.
  """
  with tf.name_scope('augment_xy_transpose'):
    rank = data.get_shape().ndims
    perm = list(range(rank))
    perm[rank - 3], perm[rank - 2] = perm[rank - 2], perm[rank - 3]
    return tf.cond(decision,
                   lambda: tf.transpose(data, perm),
                   lambda: data)


def permute_axes(x, permutation, permutable_axes):
  """Permutes the axes of `x` using the specified permutation.

  All axes not in `permutable_axes` must be identity mapped by `permutation`:
    `permutation[i] = i   if i not in permutable_axes`

  Args:
    x: Tensor of rank N to permute.
    permutation: Rank 1 tensor of shape `[N]`.
    permutable_axes: List (not Tensor) of axes that may be permuted.

  Returns:
    Permuted Tensor obtained by calling:
      `tf.transpose(x, permutation)`
    but with additional static shape information due to the restriction imposed
    by `permutable_axes`.
  """
  x = tf.convert_to_tensor(x)
  shape = x.shape.as_list()
  result = tf.transpose(x, permutation)
  # Transpose loses shape information because it does not know that the
  # permutation is restricted to the permutable_axes, but we can restore
  # some or all of it.

  # If the static dimensions of all of the permutable axes are not the
  # same, then set them all to None.  Otherwise, they can stay as is.
  if len({shape[i] for i in permutable_axes}) > 1:
    for i in permutable_axes:
      shape[i] = None

  result.set_shape(shape)
  return result


def random_contrast_brightness_adjustment(
    input_tensor: tf.Tensor,
    seg_tensor: tf.Tensor,
    contrast_factor_range: tuple[float, float] | None,
    brightness_factor_range: tuple[float, float] | None,
    apply_adjustment_to: str | None,
) -> tf.Tensor:
  """Applies brightness/contrast adjustment for the entire volume.

  Args:
    input_tensor: Tensor of raw input.
    seg_tensor: Tensor of label mask.
    contrast_factor_range: (min_contrast_factor, max_contrast_factor)
    brightness_factor_range: (min_brightness_factor, max_brightness_factor).
    apply_adjustment_to: Str, could be 'foreground', or 'background', or None.

  Returns:
    tf.Tensor: Adjusted Tensor.
  """
  adjust_tensor = tf.identity(input_tensor)
  if contrast_factor_range:
    min_contrast, max_contrast = contrast_factor_range
    contrast_factor = tf.random.uniform([], min_contrast, max_contrast)
    adjust_tensor = tf.image.adjust_contrast(adjust_tensor, contrast_factor)
  if brightness_factor_range:
    min_delta_factor, max_delta_factor = (
        brightness_factor_range
    )
    delta_factor = tf.random.uniform([], min_delta_factor, max_delta_factor)
    adjust_tensor = tf.image.adjust_brightness(
        adjust_tensor, delta=delta_factor
    )
  if apply_adjustment_to == 'foreground':
    adjust_tensor = tf.where(seg_tensor > 0, adjust_tensor, input_tensor)
  elif apply_adjustment_to == 'background':
    adjust_tensor = tf.where(seg_tensor <= 0, input_tensor, adjust_tensor)
  return adjust_tensor


class PermuteAndReflect:
  """Class for performing random permutation and reflection of axes.

  Consructing an instance of this class adds tensors to the default Tensorflow
  graph representing a randomly sampled permutation of `permutable_axes` and
  randomly sampled reflection decisions for `reflectable_axes`.

  Calling an instance of this class as a function (i.e. invoking the `__call__`
  method) applies the sampled transformations to a specified `Tensor`.

  Attributes:
    rank: The rank of Tensor that can be transformed.
    permutable_axes: 1-D int32 numpy array specifying the axes that may be
      permuted.
    reflectable_axes: 1-D int32 numpy array specifying the axes that may be
      reflected.
    reflect_decisions: bool Tensor of shape `[len(reflectable_axes)]` containing
      the sampled reflection decision for each axis in `reflectable_axes`.
    reflected_axes: Rank 1 int32 Tensor specifying the axes to be reflected
      (i.e. corresponding to a True value in `reflect_decisions`).
    permutation: int32 Tensor of shape `[len(permutable_axes)]` containing the
      sampled permutation of `permutable_axes`.
    full_permutation: int32 Tensor of shape `[rank]` that extends `permutation`
      to be a permutation of all `rank` axes, where axes not in
      `permutable_axes` are identity mapped.
  """

  def __init__(self, rank, permutable_axes, reflectable_axes,
               permutation_seed=None, reflection_seed=None):
    """Initializes the transformation nodes.

    Args:
      rank: The rank of the Tensor to be transformed.
      permutable_axes: The list (not a Tensor) of axes to be permuted.
      reflectable_axes: The list (not a Tensor) of axes to be reflected.
      permutation_seed: Optional integer.  Seed value to use for sampling axes
        permutation.
      reflection_seed: Optional integer.  Seed value to use for sampling
        reflection decisions.
    Raises:
      ValueError: if arguments are invalid.
    """
    self.rank = rank
    if len(set(permutable_axes)) != len(permutable_axes):
      raise ValueError('permutable_axes must not contain duplicates')
    if len(set(reflectable_axes)) != len(reflectable_axes):
      raise ValueError('reflectable_axes must not contain duplicates')
    if not all(0 <= x < rank for x in permutable_axes):
      raise ValueError('permutable_axes must be a subset of [0, rank-1].')
    if not all(0 <= x < rank for x in reflectable_axes):
      raise ValueError('reflectable_axes must be a subset of [0, rank-1].')
    # Cast to int32 to ensure the proper tensorflow types are used.
    self.permutable_axes = np.array(permutable_axes, dtype=np.int32)
    self.reflectable_axes = np.array(reflectable_axes, dtype=np.int32)

    if self.reflectable_axes.size > 0:
      self.reflect_decisions = tf.random_uniform([len(self.reflectable_axes)],
                                                 seed=reflection_seed) > 0.5
      self.reflected_axes = tf.boolean_mask(self.reflectable_axes,
                                            self.reflect_decisions)

    if self.permutable_axes.size > 0:
      self.permutation = tf.random_shuffle(self.permutable_axes,
                                           seed=permutation_seed)
      # full_permutation must be a list rather than an np.array of int32 because
      # some elements are set to be tensors below.
      full_permutation = [np.int32(x) for x in range(rank)]
      for i, d in enumerate(self.permutable_axes):
        full_permutation[d] = self.permutation[i]
      self.full_permutation = tf.stack(full_permutation)

  def __call__(self, x):
    """Applies the sampled permutation and reflection to `x`.

    Args:
      x: A Tensor of rank `self.rank`.

    Returns:
      The transformed Tensor, retaining as much static shape information as
      possible.
    """
    x = tf.convert_to_tensor(x)
    with tf.name_scope('permute_and_reflect'):
      if self.permutable_axes.size > 0:
        x = permute_axes(x, self.full_permutation, self.permutable_axes)
      if self.reflectable_axes.size > 0:
        x = tf.reverse(x, self.reflected_axes)
      return x


def warp_transform_size_factor(deformation_stdev_ratio, rotation_max, scale_max,
                               shear_max):
  """Estimates max patch size factor for affine transform and warping.

  Uses linear estimations of extra data needed for rotation and
  shear transformations. These estimations are not strict upper bounds,
  but the amount of extra data indicated by this function will only be
  exceeded in the unlikely case that all random transform parameters are
  chosen to be their maximum values. In this case, the remaining extra
  data will be supplied by mirror padding (mode argument to _elastic_warp_2d
  and _affine_transform_2d.

  Args:
    deformation_stdev_ratio: ratio of the standard deviation of normally
      distributed deltas added to control points for elastic deformation to the
      size of the deformed patch
    rotation_max: rotation angle (radians) chosen randomly between +-
      rotation_max
    scale_max: scale factors chosen randomly between 1 +- scale_max
    shear_max: shear angle (counter-clockwise radians) chosen randomly between
      +- shear_max

  Returns:
    estimated maximum patch size factor
  """
  rotation_factor = min(np.pi / 4, rotation_max) / (np.pi / 4)
  shear_factor = min(np.pi, shear_max) / np.pi
  return 1 + (
      deformation_stdev_ratio + rotation_factor + scale_max + shear_factor)


def _elastic_warp_2d(patch,
                     num_control_points_ratio,
                     deformation_stdev_ratio,
                     mode='reflect'):
  """Applies 2D elastic deformation to all y,x slices of patch.

  The same deformation is applied separately at each pair of
  y,x coordinates in patch

  Args:
    patch: input 4D numpy array, [b, y, x, c]
    num_control_points_ratio: ratio of number of control points in elastic
      deformation control grid to patch shape
    deformation_stdev_ratio: ratio of the standard deviation of normally
      distributed deltas added to control points for elastic deformation to the
      size of the deformed patch
    mode: method to handle points outside of patch; see documentation for
      skimage.transform.warp

  Returns:
    warped patch
  """
  num_control_points_y = max(int(num_control_points_ratio * patch.shape[1]), 1)
  num_control_points_x = max(int(num_control_points_ratio * patch.shape[2]), 1)
  y = np.linspace(0, patch.shape[1], num_control_points_y)
  x = np.linspace(0, patch.shape[2], num_control_points_x)
  coords = np.array([(y0, x0) for y0 in y for x0 in x])  # pylint: disable=g-complex-comprehension
  deformation_stdev = deformation_stdev_ratio * np.min(patch.shape)
  deformations = np.random.normal(0, deformation_stdev, coords.shape)
  deformed_coords = coords + deformations
  grid_y, grid_x = np.mgrid[0:patch.shape[1], 0:patch.shape[2]]
  grid = griddata(
      coords, deformed_coords, (grid_y, grid_x), method='cubic', fill_value=0)
  warped_patch = np.zeros(patch.shape, dtype=patch.dtype)
  for b in range(patch.shape[0]):
    for c in range(patch.shape[3]):
      warped_patch[b, :, :, c] = warp(
          patch[b, :, :, c],
          np.array((grid[:, :, 0], grid[:, :, 1])),
          mode=mode)
  return warped_patch


def _affine_transform_2d(patch,
                         rotation_max,
                         scale_max,
                         shear_max,
                         mode='reflect'):
  """Applies 2D affine transformation to all y,x slices of patch.

  The same transform is applied separately at each pair of
  y,x coordinates in patch

  Args:
    patch: input 4D numpy array, [b, y, x, c]
    rotation_max: rotation angle (radians) chosen randomly between +-
      rotation_max
    scale_max: scale factors chosen randomly between 1 +- scale_max
    shear_max: shear angle (counter-clockwise radians) chosen randomly between
      +- shear_max
    mode: method to handle points outside of patch; see documentation for
      skimage.transform.warp

  Returns:
    transformed patch
  """
  rotation = (np.random.rand() * 2 - 1) * rotation_max
  scale = 1 - (np.random.rand(2) * 2 - 1) * scale_max
  shear = (np.random.rand() * 2 - 1) * shear_max
  # Preserves skimage < 0.22.0 behavior.
  scale[1] *= np.cos(shear)
  at = AffineTransform(scale=scale, rotation=rotation, shear=shear)
  transformed_patch = np.zeros(patch.shape, dtype=patch.dtype)
  for b in range(patch.shape[0]):
    for c in range(patch.shape[3]):
      transformed_patch[b, :, :, c] = warp(patch[b, :, :, c], at, mode=mode)
  return transformed_patch


def _apply_at_random_z_indices(patch, fn, max_indices_ratio):
  """Applies argument function to randomly selected z indices of patch.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    fn: function to apply, should take a 4D numpy array, [b, y, x, c]
    max_indices_ratio: maximum fraction of z indices in patch at which transform
      will be applied

  Returns:
    [array with function applied, z_indices for tensorboard summary]
  """
  max_indices = max(int(max_indices_ratio * patch.shape[1]), 1)
  num_indices = np.random.randint(1, max_indices + 1)
  z_indices = np.random.choice(patch.shape[1], num_indices, replace=False)
  for z in z_indices:
    transformed = fn(patch[:, z, :, :, :].astype(np.float64))
    patch[:, z, :, :, :] = transformed.astype(patch.dtype)
  return patch, z_indices


def elastic_warp(patch,
                 max_indices_ratio,
                 num_control_points_ratio,
                 deformation_stdev_ratio,
                 skip_ratio=0,
                 mode='reflect'):
  """Applies elastic deformation to selected z indices of patch.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    max_indices_ratio: maximum fraction of z indices in patch at which warping
      will be applied
    num_control_points_ratio: ratio of number of control points in elastic
      deformation control grid to patch shape
    deformation_stdev_ratio: ratio of the standard deviation of normally
      distributed deltas added to control points for elastic deformation to the
      size of the deformed patch
    skip_ratio: probability of skipping augmentation and returning unmodified
      patch
    mode: method to handle points outside of patch; see documentation for
      skimage.transform.warp

  Returns:
    [transformed array with warping applied,
    z_indices (or -1 if skipped) for tensorboard summary]
  """
  patch = patch.copy()
  if np.random.rand() < skip_ratio:
    return [patch, -1]

  def warp_function(p):
    return _elastic_warp_2d(
        p, num_control_points_ratio, deformation_stdev_ratio, mode=mode)

  return _apply_at_random_z_indices(patch, warp_function, max_indices_ratio)


def affine_transform(patch,
                     max_indices_ratio,
                     rotation_max,
                     scale_max,
                     shear_max,
                     skip_ratio=0,
                     mode='reflect'):
  """Applies affine transform to selected z indices of patch.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    max_indices_ratio: maximum fraction of z indices in patch at which transform
      will be applied
    rotation_max: rotation angle (radians) chosen randomly between +-
      rotation_max
    scale_max: scale factors chosen randomly between 1 +- scale_max
    shear_max: shear angle (counter-clockwise radians) chosen randomly between
      +- shear_max
    skip_ratio: probability of skipping augmentation and returning unmodified
      patch
    mode: method to handle points outside of patch; see documentation for
      skimage.transform.warp

  Returns:
    [transformed array with affine transform applied,
    z_indices (or -1 if skipped) for tensorboard summary]
  """
  patch = patch.copy()
  if np.random.rand() < skip_ratio:
    return [patch, -1]

  def transform_function(p):
    return _affine_transform_2d(
        p, rotation_max, scale_max, shear_max, mode=mode)

  return _apply_at_random_z_indices(patch, transform_function,
                                    max_indices_ratio)


def _center_crop(patch, zyx_cropped_shape):
  """Crops center z,y,x dimensions of patch.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    zyx_cropped_shape: dimensions of cropped result, [z, y, x]

  Returns:
    center (along z,y,x dimension) of patch cropped to final_shape
  """
  diff = np.array(patch.shape[1:-1]) - np.array(zyx_cropped_shape)
  assert np.all(diff >= 0)
  start = diff // 2
  end = patch.shape[1:-1] - np.ceil(diff / 2.0).astype(int)
  return patch[:, start[0]:end[0], start[1]:end[1], start[2]:end[2], :]


def _edge_pad(patch, zyx_padded_shape, mode='edge'):
  """Pads z,y,x dimensions of patch.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    zyx_padded_shape: dimensions of padded result, [z, y, x]
    mode: optional padding mode, see numpy.pad() for details

  Returns:
    padded array with patch in center of z,y,x dimensions
  """
  diff = np.array(zyx_padded_shape) - np.array(patch.shape[1:-1])
  assert np.all(diff >= 0)
  pad = [[d // 2, np.ceil(d / 2.0).astype(int)] for d in diff]
  pad = [[0, 0]] + pad + [[0, 0]]
  return np.pad(patch, pad, mode)


def misalignment(patch,
                 labels,
                 mask,
                 patch_final_zyx,
                 labels_final_zyx,
                 mask_final_zyx,
                 max_offset,
                 slip_ratio,
                 skip_ratio=0):
  """Performs slip and translation misalignment augmentations.

  Patch, labels, and mask inputs are first edge padded to the same size
  on the z,y,x axes. A random z-index is selected.  With slip_ratio
  probability, all values at this index are translated in x,y.
  Otherwise all values at z indices >= the selected index are translated.
  The same z index and offset values are used for patch, label, and mask
  inputs. Patch, labels, and mask are then cropped to the center of the
  z,y,x axes to match *_final_zyx arguments.

  Args:
    patch: 5D numpy array of data, [b, z, y, x, c]
    labels: 5D numpy array of labels, [b, z, y, x, c]
    mask: 5D numpy mask array, [b, z, y, x, c]
    patch_final_zyx: final shape to crop patch, [z, y, x]
    labels_final_zyx: final shape to crop labels, [z, y, x]
    mask_final_zyx: final shape to crop mask, [z, y, x]
    max_offset: max pixel offset for x,y misalignment
    slip_ratio: probability of slip (vs translation)
    skip_ratio: probability of skipping augmentation and returning
      un-translated, cropped input

  Returns:
    list including transformed arrays
    and starting z index for tensorboard summary (-1 if skipped)
  """
  patch, labels, mask = patch.copy(), labels.copy(), mask.copy()
  if np.random.rand() < skip_ratio:
    return (_center_crop(patch, patch_final_zyx),
            _center_crop(labels, labels_final_zyx),
            _center_crop(mask, mask_final_zyx), -1)

  zyx_max_shape = np.array([patch.shape, labels.shape,
                            mask.shape]).max(axis=0)[1:-1]
  padded_data = [
      _edge_pad(patch, zyx_max_shape),
      _edge_pad(labels, zyx_max_shape),
      _edge_pad(mask, zyx_max_shape)
  ]

  offset_y, offset_x = np.random.randint(-max_offset, max_offset + 1, 2)
  z_start = np.random.randint(0, zyx_max_shape[0])
  is_slip = np.random.rand() < slip_ratio

  results = []
  for d in padded_data:
    if is_slip:
      d[:, z_start, :, :, :] = np.roll(d[:, z_start, :, :, :], offset_y, 1)
      d[:, z_start, :, :, :] = np.roll(d[:, z_start, :, :, :], -offset_x, 2)
    else:
      d[:, z_start:, :, :, :] = np.roll(d[:, z_start:, :, :, :], offset_y, 2)
      d[:, z_start:, :, :, :] = np.roll(d[:, z_start:, :, :, :], -offset_x, 3)
    results.append(d)
  results[0] = _center_crop(results[0], patch_final_zyx)
  results[1] = _center_crop(results[1], labels_final_zyx)
  results[2] = _center_crop(results[2], mask_final_zyx)
  results.append(z_start)
  return results


def _quadrant_replace(patch, z, replacement, quadrant_prob):
  """Replaces randomly selected x,y quadrants of patch at specified z index.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c] patch is modified in place
    z: z index on which to replace x,y quadrants
    replacement: 4D numpy array containing replacement values, [b, y, x, c]
                 same shape as patch[:, z, :, :, :]
   quadrant_prob: probability that values in each quadrant are replaced
  """
  apply_quadrants = np.random.rand(4) < quadrant_prob
  y = np.random.randint(0, patch.shape[2])
  x = np.random.randint(0, patch.shape[3])
  if apply_quadrants[0]:
    patch[:, z, 0:y, 0:x, :] = replacement[:, 0:y, 0:x, :]
  if apply_quadrants[1]:
    patch[:, z, y:, 0:x, :] = replacement[:, y:, 0:x, :]
  if apply_quadrants[2]:
    patch[:, z, 0:y, x:, :] = replacement[:, 0:y, x:, :]
  if apply_quadrants[3]:
    patch[:, z, y:, x:, :] = replacement[:, y:, x:, :]


def missing_section(patch,
                    max_indices_ratio,
                    skip_ratio=0,
                    fill_value=None,
                    max_fill_val=256,
                    full_prob=0.5,
                    quadrant_prob=0.5):
  """Performs missing section augmentation.

  All values in randomly selected x,y quadrants of randomly
  selected z indices are replaced with a randomly selected value.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    max_indices_ratio: maximum fraction of z indices in patch at which sections
      will be replaced
    skip_ratio: probability of skipping augmentation and returning unmodified
      patch
    fill_value: replacement value for blank slides. If not set, will default to
      uniform random from [0, max_fill_val]
    max_fill_val: replacement value selected uniformly at random from [0,
      max_fill_val)
    full_prob: probability that all values at selected z indices are replaced
    quadrant_prob: probability that values in each randomly selected x,y
      quadrant are replaced

  Returns:
    [transformed array with random value replacing selected sections,
     z_indices (or -1 if skipped) for tensorboard summary]
  """
  patch = patch.copy()
  if np.random.rand() < skip_ratio:
    return [patch, -1]
  max_indices = max(int(max_indices_ratio * patch.shape[1]), 1)
  num_indices = np.random.randint(1, max_indices + 1)
  z_indices = np.random.choice(patch.shape[1], num_indices, replace=False)
  fill_val = (
      fill_value if fill_value is not None else np.random.rand() * max_fill_val)
  fill_array = np.full(patch[:, 0, :, :, :].shape, fill_val, patch.dtype)
  for z in z_indices:
    if np.random.rand() < full_prob:
      patch[:, z, :, :, :] = fill_val
    else:
      _quadrant_replace(patch, z, fill_array, quadrant_prob)
  return patch, z_indices


def out_of_focus_section(patch,
                         max_indices_ratio,
                         max_filter_stdev,
                         skip_ratio=0,
                         full_prob=0.5,
                         quadrant_prob=0.5):
  """Applies out-of-focus-section augmentation.

  A Gaussian blur is applied to all values in randomly selected x,y
  quadrants at randomly selected z indices.

  Args:
    patch: input 5D numpy array, [b, z, y, x, c]
    max_indices_ratio: maximum fraction of z indices in patch at which sections
      will be blurred
    max_filter_stdev: standard deviation of Gaussian filter chosen from [0,
      max_filter_stdev), in pixels
    skip_ratio: probability of skipping augmentation and returning unmodified
      patch
    full_prob: probability that all x,y values at selected z indices are blurred
    quadrant_prob: probability that each randomly selected x,y quadrant is
      blurred

  Returns:
    [transformed array with blurred section,
     z_indices (or -1 if skipped) for tensorboard summary]
  """
  patch = patch.copy()
  if np.random.rand() < skip_ratio:
    return [patch, -1]
  max_indices = max(int(max_indices_ratio * patch.shape[1]), 1)
  num_indices = np.random.randint(1, max_indices + 1)
  z_indices = np.random.choice(patch.shape[1], num_indices, replace=False)
  filter_stdev = np.random.rand() * max_filter_stdev
  for z in z_indices:
    blurred = gaussian_filter(patch[:, z, :, :, :], filter_stdev)
    if np.random.rand() < full_prob:
      patch[:, z, :, :, :] = blurred
    else:
      _quadrant_replace(patch, z, blurred, quadrant_prob)
  return patch, z_indices


def grayscale_perturb(patch,
                      max_contrast_factor,
                      max_brightness_factor,
                      skip_ratio=0,
                      max_val=255,
                      full_prob=0.5):
  """Applies brightness/contrast adjustment and gamma correction.

  Grayscale perturbation factors are chosen once for the
  entire input tensor (0.5 prob) or independently for all
  individual z indices (0.5 prob).

  Perturbed result = (((patch / max_val) * cf + bf)**g) * max_val
  where
    cf: contrast factor drawn from 1 +- (max_contrast_factor / 2)
    bf: brightness factor drawn from +-(max_brightness_factor / 2)
    g:  gamma power = selected between 0.5 and 2
  note: (p / max_val) * cf + bf is clipped to [0, 1]
        before being raised to g power

  Args:
    patch: input numpy array, rank 5, [b, z, y, x, c]
    max_contrast_factor: contrast factor distribution parameter (see above)
    max_brightness_factor: brightness factor distribution parameter (see above)
    skip_ratio: probability of skipping augmentation and returning unmodified
      patch
    max_val: maximum value of patch dtype
    full_prob: probability that perturbation factors are chosen for entire
      volume, as opposed to independently for each z index

  Returns:
    [array with brightness/contrast and gamma adjustment,
     1/0 apply/skip for tensorboard summary]
  """
  patch = patch.copy()
  if np.random.rand() < skip_ratio:
    return patch, 0

  def perturb_fn(patch):
    contrast_factor = 1 + (np.random.rand() - 0.5) * max_contrast_factor
    brightness_factor = (np.random.rand() - 0.5) * max_brightness_factor
    power = 2.0**(np.random.rand() * 2 - 1)
    normalized = patch.astype(np.float32) / max_val
    adjusted = normalized * contrast_factor + brightness_factor
    gamma = np.clip(adjusted, 0, 1)**power
    rescaled = (gamma * max_val).astype(patch.dtype)
    return rescaled

  if np.random.rand() < full_prob:
    return [perturb_fn(patch), np.array(1)]
  else:
    for z in range(patch.shape[1]):
      patch[:, z, :, :, :] = perturb_fn(patch[:, z, :, :, :])
    return patch, 1


def apply_section_augmentations(
    patch, labels, mask, patch_final_zyx, labels_final_zyx, mask_final_zyx,
    elastic_warp_skip_ratio, affine_transform_skip_ratio,
    misalignment_skip_ratio, missing_section_skip_ratio, outoffocus_skip_ratio,
    grayscale_skip_ratio, max_warp_indices_ratio, num_control_points_ratio,
    deformation_stdev_ratio, max_affine_transform_indices_ratio, rotation_max,
    scale_max, shear_max, max_xy_offset, slip_vs_translate_ratio,
    max_missing_section_indices_ratio, max_outoffocus_indices_ratio,
    max_filter_stdev, max_contrast_factor, max_brightness_factor):
  """Performs ssEM training set augmentations.

  Augmentations performed by this function were designed
  based on those in the paper "Superhuman Accuracy on the
  SNEMI3D Connectomics Challenge" by Kisuk Lee, et al.
  (https://arxiv.org/abs/1706.00120).

  Args:
    patch: input 5D tensor affected by all augmentations, [b, z, y, x, c]
    labels: input 5D tensor affected only by misalignment, [b, z, y, x, c]
    mask: input 5D tensor affected only by misalignment, [b, z, y, x, c]
    patch_final_zyx: shape to crop patch after misalignment, [z, y, x]
    labels_final_zyx: shape to crop labels after misalignment, [z, y, x]
    mask_final_zyx: shape to crop mask after misalignment, [z, y, x]
    elastic_warp_skip_ratio: elastic deformation augmentation skip prob
    affine_transform_skip_ratio: affine transfom augmentation skip prob
    misalignment_skip_ratio: misalignment augmentation skip prob
    missing_section_skip_ratio: missing section augmentation skip prob
    outoffocus_skip_ratio: out-of-focus section augmentation skip prob
    grayscale_skip_ratio: grayscale perturbation skip prob
    max_warp_indices_ratio: maximum fraction of z indices at which elastic
      warping is to be applied
    num_control_points_ratio: number of control points to patch shape in elastic
      deformation control grid
    deformation_stdev_ratio: ratio of the standard deviation of normally
      distributed deltas added to control points for elastic deformation to the
      size of the deformed patch
    max_affine_transform_indices_ratio: maximum fraction of z indices at which
      affine transform applied
    rotation_max: rotation angle (radians) for affine transform chosen randomly
      between +- rotation_max
    scale_max: scale factors chosen randomly between 1 +- scale_max
    shear_max: shear angle (counter-clockwise radians) chosen randomly between
      +- shear_max
    max_xy_offset: maximum translation in y,x axes for misalignment
    slip_vs_translate_ratio: probability of single index slip vs multi-index
      translation misalignment
    max_missing_section_indices_ratio: maximum fraction of "missing" z indices
    max_outoffocus_indices_ratio: maximum fraction of out-of-focus z indices
    max_filter_stdev: maximum standard deviation of Gaussian filter used for
      out-of-focus blur
    max_contrast_factor: contrast factor drawn from 1 +- (max_contrast_factor /
      2)
    max_brightness_factor: brightness factor drawn from +-
      (max_brightness_factor / 2)

  Returns:
    tensorflow ops for patch, labels, and mask with augmentations
    applied and cropped to final shapes
  """

  def elastic_warp_fn(patch):
    return elastic_warp(patch, max_warp_indices_ratio, num_control_points_ratio,
                        deformation_stdev_ratio, elastic_warp_skip_ratio)

  def affine_transform_fn(patch):
    return affine_transform(patch, max_affine_transform_indices_ratio,
                            rotation_max, scale_max, shear_max,
                            affine_transform_skip_ratio)

  def misalignment_fn(patch, labels, mask):
    return misalignment(patch, labels, mask, patch_final_zyx, labels_final_zyx,
                        mask_final_zyx, max_xy_offset, slip_vs_translate_ratio,
                        misalignment_skip_ratio)

  def missing_section_fn(patch):
    return missing_section(patch, max_missing_section_indices_ratio,
                           missing_section_skip_ratio)

  def outoffocus_section_fn(patch):
    return out_of_focus_section(patch, max_outoffocus_indices_ratio,
                                max_filter_stdev, outoffocus_skip_ratio)

  def grayscale_perturb_fn(patch):
    return grayscale_perturb(patch, max_contrast_factor, max_brightness_factor,
                             grayscale_skip_ratio)

  patch_shape = [patch.shape[0]] + list(patch_final_zyx) + [patch.shape[-1]]
  labels_shape = [labels.shape[0]] + list(labels_final_zyx) + [labels.shape[-1]]
  mask_shape = [mask.shape[0]] + list(mask_final_zyx) + [mask.shape[-1]]

  with tf.name_scope('section_augmentations'):
    patch, elastic_warp_summary = tf.py_func(elastic_warp_fn, [patch],
                                             [patch.dtype, tf.int64])
    tf.summary.histogram('elastic_warp_z_indices', elastic_warp_summary)
    patch, affine_transform_summary = tf.py_func(affine_transform_fn, [patch],
                                                 [patch.dtype, tf.int64])
    tf.summary.histogram('affine_transform_z_indices', affine_transform_summary)
    patch, labels, mask, misalignment_summary = tf.py_func(
        misalignment_fn, [patch, labels, mask],
        [patch.dtype, labels.dtype, mask.dtype, tf.int64])
    tf.summary.scalar('misalignment_summary', misalignment_summary)
    patch, missing_section_summary = tf.py_func(missing_section_fn, [patch],
                                                [patch.dtype, tf.int64])
    tf.summary.histogram('missing_section_z_indices', missing_section_summary)
    patch, outoffocus_summary = tf.py_func(outoffocus_section_fn, [patch],
                                           [patch.dtype, tf.int64])
    tf.summary.histogram('out-of-focus_z_indices', outoffocus_summary)
    patch, grayscale_summary = tf.py_func(grayscale_perturb_fn, [patch],
                                          [patch.dtype, tf.int64])
    tf.summary.scalar('grayscale_applied', grayscale_summary)

  patch.set_shape(patch_shape)
  labels.set_shape(labels_shape)
  mask.set_shape(mask_shape)
  return patch, labels, mask
