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

"""Functions for processing segmentation tensors."""

import tensorflow.google as tf


def dict_to_hashtable(relabel_map: dict[int, int]) -> tf.lookup.StaticHashTable:
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=list(relabel_map.keys()),
          key_dtype=tf.int64,
          values=list(relabel_map.values()),
          value_dtype=tf.int64),
      default_value=-1)


def relabel_with_table(seg: tf.Tensor,
                       table: tf.lookup.StaticHashTable) -> tf.Tensor:
  relabeled = table.lookup(seg)
  return tf.where(relabeled < 0, seg, relabeled)


def relabel(seg: tf.Tensor, relabel_map: dict[int, int]) -> tf.Tensor:
  table = dict_to_hashtable(relabel_map)
  return relabel_with_table(seg, table)


def match_and_relabel(seg: tf.Tensor, volname: tf.Tensor,
                      relabel_maps: dict[str, dict[int, int]]) -> tf.Tensor:
  """Finds and applies a relabel map to the segmentation.

  Args:
    seg: segmentation tensor to relabel (any shape)
    volname: string tensor with volume name (shape: [1])
    relabel_maps: map from volume names to old_id->new_id maps

  Returns:
    relabeled segmentation if a matching relabel map is found in
    `relabel_maps`, original segmentation otherwise
  """

  # The alternative here would be to pass relabel_maps to a py_function and
  # select the right entry there (based on volname). As of Jul 2022, this causes
  # memory leaks, so instead we precompute the hash tables, define a separate
  # function for every label, and dispatch via switch_case().
  keys, fns = [tf.convert_to_tensor('')], [lambda: seg]
  for key, id_map in relabel_maps.items():
    ht = dict_to_hashtable(id_map)
    fns.append(lambda ht=ht: relabel_with_table(seg, ht))
    keys.append(key)

  sel = tf.equal(volname, keys)
  idx = tf.argmax(tf.cast(sel, tf.int32), output_type=tf.int32)
  ret = tf.switch_case(idx, fns)
  ret.set_shape(seg.shape)
  return ret
