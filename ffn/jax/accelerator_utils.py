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
"""Accelerator topology and batch size utilities."""

import dataclasses
import jax


@dataclasses.dataclass(frozen=True)
class AcceleratorTopologyInfo:
  global_batch_size: int
  host_batch_size: int
  cores_per_chip: int
  num_chips: int
  local_chips: int


def get_accelerator_topology_info(
    per_device_batch_size: int,
) -> AcceleratorTopologyInfo:
  """Computes topology info and batch sizes.

  Args:
    per_device_batch_size: Desired batch size per device (core).

  Returns:
    AcceleratorTopologyInfo with topology and batch size info.
  """
  devices = jax.local_devices()
  if devices and hasattr(devices[0], 'core_on_chip'):
    cores_per_chip = max(d.core_on_chip for d in devices) + 1
  else:
    cores_per_chip = 1

  num_chips = jax.device_count() // cores_per_chip
  local_chips = jax.local_device_count() // cores_per_chip

  global_batch_size = per_device_batch_size * num_chips
  host_batch_size = per_device_batch_size * local_chips

  return AcceleratorTopologyInfo(
      global_batch_size=global_batch_size,
      host_batch_size=host_batch_size,
      cores_per_chip=cores_per_chip,
      num_chips=num_chips,
      local_chips=local_chips,
  )
