#!/usr/bin/env python
# Copyright 2017-2023 Google Inc.
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

from distutils.core import setup

setup(
    name='ffn',
    version='0.1.0',
    author='Michal Januszewski',
    author_email='mjanusz@google.com',
    packages=[
        'ffn',
        'ffn.inference',
        'ffn.training',
        'ffn.training.models',
        'ffn.utils',
    ],
    scripts=[
        'build_coordinates.py',
        'compute_partitions.py',
        'run_inference.py',
        'train.py',
    ],
    url='https://github.com/google/ffn',
    license='LICENSE',
    description='Flood-Filling Networks for volumetric instance segmentation',
    long_description=open('README.md').read(),
    install_requires=[
        'connectomics',
        'edt>=2.3.0',
        'pandas',
        'dataclasses-json>=0.5.6',
        'scikit-image>=0.11.0',
        'scipy>=0.15.1',
        'numpy>=1.11.1',
        'tensorflow>=1.4.0',
        'tensorstore>=0.1.40',
        'h5py>=2.7.0',
        'Pillow>=5.3.0',
        'Pillow-PIL',
        'absl-py>=0.1.4',
        'tf-slim>=1.1.0',
        'jax>=0.2.25',
    ],
)
