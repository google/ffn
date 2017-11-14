# Flood-Filling Networks

Flood-Filling Networks (FFNs) are a class of neural networks designed for
instance segmentation of complex and large shapes, particularly in volume
EM datasets of brain tissue.

For more details, see the related publications:

 * https://arxiv.org/abs/1611.00421
 * https://doi.org/10.1101/200675

This is not an official Google product.

# Installation

No installation is required, but please ensure that the following dependencies
are available on your system:

- skimage >= 0.11.0
- scipy >= 0.15.1
- numpy >= 1.11.1
- tensorflow >= 1.4.0
- h5py >= 2.7.0
- PIL >= 1.1.7
- absl-py >= 0.1.4

The code has been tested on an Ubuntu 16.04.3 LTS system with the library
versions listed above, and equipped with a Tesla P100 GPU.

# Training

FFN networks can be trained with the `train.py` script, which expects a
TFRecord file of coordinates at which to sample data from input volumes.
We provide a sample coordinate file for the FIB-25 `validation1` volume
included in `third_party`. Due to its size, that file is hosted in
Google Cloud Storage. If you haven't used it before, you will need to
install the Google Cloud SDK and set it up with:

```shell
  gcloud auth application-default login
```

You will also need to create a local copy of the labels and image with:

```shell
  gsutil rsync -r -x ".*.gz" gs://ffn-flyem-fib25/ third_party/neuroproof_examples
```

You can then start training the FFN with:

```shell
  python train.py \
    --train_coords gs://ffn-flyem-fib25/validation_sample/fib_flyem_validation1_label_lom24_24_24_part14_wbbox_coords-*-of-00025.gz \
    --data_volumes validation1:third_party/neuroproof_examples/validation_sample/grayscale_maps.h5:raw \
    --label_volumes validation1:third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" \
    --image_mean 128 \
    --image_stddev 33
```

Note that both training and inference with the provided model are
computationally expensive processes. We recommend a GPU-equipped machine
for best results, particularly when using the FFN interactively in a Jupyter
notebook. Training the FFN as configured above requires a GPU with 12 GB of RAM.
You can reduce the model depth, `fov_size`, or number of features in the convolutional
layers to reduce the memory usage.

# Inference

We provide two examples of how to run inference with a trained FFN model.
For a non-interactive setting, you can use the `run_inference.py` script:

```shell
  python run_inference.py \
    --inference_request="$(cat configs/inference_training_sample2.pbtxt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }'
```

which will segment the `training_sample2` volume and save the results in
the `results/fib25/training2` directory. Two files will be produced:
`seg-0_0_0.npz` and `seg-0_0_0.prob`. Both are in the `npz` format and
contain a segmentation map and quantized probability maps, respectively.
In Python, you can load the segmentation as follows:

```python
  from ffn.inference import storage
  seg, _ = storage.load_segmentation('results/fib25/training2', (0, 0, 0))
```

We provide sample segmentation results in `results/fib25/sample-training2.npz`.
For the training2 volume, segmentation takes ~7 min with a P100 GPU.

For an interactive setting, check out `ffn_inference_demo.ipynb`. This Jupyter
notebook shows how to segment a single object with an explicitly defined
seed and visualize the results while inference is running.

Both examples are configured to use a 3d convstack FFN model trained on the
`validation1` volume of the FIB-25 dataset from the FlyEM project at Janelia.
