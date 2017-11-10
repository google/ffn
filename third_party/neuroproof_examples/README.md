This directory contains FIB-SEM data and segmentation ground truth from FlyEM's
neuroproof GitHub repository:

  https://github.com/janelia-flyem/neuroproof_examples

The raw .png images were converted into h5 files using the png_to_h5.py script.

Due to size, the files are stored in Google Cloud Storage. To create a local
copy, run:

  gsutil rsync -r -x ".*.gz" gs://ffn-flyem-fib25/ .
