#!/usr/bin/env python

from distutils.core import setup

setup(
	name		= 'ffn',
	version		= '0.1.0',
	author		= 'Michal Januszewski',
	author_email	= 'mjanusz@google.com',
	packages	= ['ffn', 'ffn.inference', 'ffn.training', 'ffn.utils'],
	scripts		= ['build_coordinates.py', 'compute_partitions.py', 'run_inference.py', 'train.py'],
	url		= 'https://github.com/google/ffn',
	license		= 'LICENSE',
	description	= 'Flood-Filling Networks for volumetric instance segmentation',
	long_description= open('README.md').read(),
	install_requires= ['scikit-image', 'scipy', 'numpy', 'tensorflow', 'h5py', 'PIL', 'absl-py'],
)
