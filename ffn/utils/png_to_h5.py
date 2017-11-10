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
"""Converts PNG files from the working directory into a HDF5 volume.

Usage:
  ./png_to_h5.py output_filename.h5
"""

import glob
import sys

import h5py
import numpy as np
from scipy import misc

assert len(sys.argv) >= 2

png_files = glob.glob('*.png')
png_files.sort()
images = [misc.imread(i) for i in png_files]
images = np.array(images)
with h5py.File(sys.argv[1], 'w') as f:
  f.create_dataset('raw', data=images, compression='gzip')

