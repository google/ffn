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
"""Utilities for working with geometry objects."""

import numpy
from . import vector_pb2


def ToVector3j(*args):
  """Converts from *args to Vector3j.

  Args:
    *args: Can either be three separate ints, or a single sequence arg, or a
           single Vector3j arg.  No-op if already Vector3j.

  Returns:
    New Vector3j.

  Raises:
    Exception: Bad input.
  """
  if len(args) == 3:
    seq = args
  elif len(args) == 1:
    seq = args[0]
  else:
    raise ValueError('Expected three ints, a 3-sequence of ints, or a Vector3j')

  if isinstance(seq, vector_pb2.Vector3j): return seq
  if isinstance(seq, numpy.ndarray) and seq.dtype.kind in 'iu':
    seq = [long(s) for s in seq]
  if len(seq) != 3:
    raise ValueError('Expected three ints, a 3-sequence of ints, or a Vector3j')

  p = vector_pb2.Vector3j()
  p.x = seq[0]
  p.y = seq[1]
  p.z = seq[2]
  return p


def To3Tuple(vector):
  """Converts from Vector3j/tuple/numpy array to tuple.

  Args:
    vector: Vector3j/Vector3f proto, 3-element tuple, or 3-element numpy array.
  Returns:
    (x, y, z) tuple.
  Raises:
    ValueError: Unsupported argument type.
  """
  if isinstance(vector, (vector_pb2.Vector3j, vector_pb2.Vector3f)):
    return (vector.x, vector.y, vector.z)
  if isinstance(vector, numpy.ndarray):
    if vector.shape != (3,):
      raise ValueError('Expected a Vector3j or 3-element sequence/numpy array')
  else:
    if len(vector) != 3:
      raise ValueError('Expected a Vector3j or 3-element sequence/numpy array')
    if (vector[0] != int(vector[0]) or
        vector[1] != int(vector[1]) or
        vector[2] != int(vector[2])):
      raise ValueError('All elements must be integers')
  return (int(vector[0]), int(vector[1]), int(vector[2]))


def ToNumpy3Vector(vector, dtype=None):
  """Converts from Vector3j or 3-element sequence to numpy array."""
  return numpy.array(To3Tuple(vector), dtype=dtype)
