In this document we present a sample workflow for use of FFN.

# Data
* The lab has electron microscopy data for a biological organism, represented as an image stack (set of pngs or tifs) or a dense data format (e.g. hdf5 cube). The data may or may not be isotropic, and there is some agreed-upon set of (global) coordinates for the span of the full dataset. Smaller boxes may be extracted from the larger volume for various operations. We refer to this dataset as "raw" data. Note that even if alternate views of the data have been generated for human visualisation, you will usually want to use the original unmodified values for this pipeline. You will use a subset of the raw data for training (for good results, aim for a 600-edge cube)
* The lab has produced segmentations - dense datasets that are meant to correspond to particular boxes in the larger dataset, but rather than having image data, each voxel in the segmentation dataset has a value corresponding to a believed segment that voxel belongs to. A segment is understood as the set of coordinates in the segmentation dataset that share the same value (the actual value that corresponds to a segment is not particularly meaningful, but keeping them consistent between parts of the workflow makes talking about particular segments easier).

# Goal in using FFN
The lab would like to produce new segmentation datasets using machine learning and inference. The lab has some GPU-accelerated hardware suitable for this use.

# Software prerequisites
* It is not necessary, but you can install FFN with "pip install ." so other software can use it as a library
* Conda is a reasonable way to get a python environment with the components you need

# Setup and Process
* The area of the global coordinate system that a segmentation dataset covers should be prepared into two hdf5 volumes, one with the raw data and one with the segmentation data. The segmentation data should be in an int64 collection in the volume. The raw data is often in a uint8 collection.
* Run compute_partitions.py and build_coordinates as described in the readme; these transform the ground truth into a tensorflow record file (used to launch the network for training)
* Run train.py, giving it the record file under train_coords, as well as the raw image data and the label volume. You should not need to adjust other parameters. This code continually drops new numbered network snapshots; if interrupted it can be restarted. It is best to leave it training for a very long time (even with high-end GPUs, it can be reasonable to leave it running for weeks or more depending on the desired quality). Multi-GPU training is not built-in, but can be done using the distributed Tensorflow API: https://www.tensorflow.org/deploy/distributed using an asynchronous SGD and a custom training script.
* After train.py is finished, you will have a directory full of model.ckpt-NUMBER.EXTENSION files. Likely you are only interested in the set with the greatest number; this is your trained network.
* Extract a new, non-overlapping set of data from your image stack (300-edge cubes are reasonable, 600-edge or larger cubes will take awhile but are usable by patient biologists) and save it into an hdf5 file with the same format as the raw data set used for training. This will be your inference target.
* Write a pbtxt as a configuration for inference. See the bottom of this doc for specifics
* Run run_inference.py, passing it the protobuf-text config you just generated (as in the readme), as well as a bounding box spec (leave the start bit at zero, set the size to be the size of your inference target along each dimension). This will produce results in the directory you specified. It can take awhile
* When the results are done, you will want to convert them into a format suitable for modification with other software (hdf5, or an image stack). A code snippet to do this is provided below.

# Exporting FFN results
``` python
sys.path.append('/path/to/ffn')
from ffn.inference import storage
import h5py

seg, _ = storage.load_segmentation('/path/to/ffn/resultsdir', (0, 0, 0))
dest = h5py.File('/target/h5pyfile.h5', 'w')
dest.create_dataset('seg', data=seg, dtype='int64')
```

# Writing an inference config
Start with configs/inference_training_sample2.pbtxt, making a copy of it and modifying that to suit
your needs.

The protobuf file here (in the sources) documents most of the parameters in that file:

ffn/inference/inference.proto

You will need to modify at least these fields:
* image - to specify a volume of image data on which you wish to run inference
* segmentation_output_dir - to specify a (possibly unique) directory for snapshots and results of your run

You likely will want to experiment with the pad_value and move_threshold in the inference options to fine-tune your results.
