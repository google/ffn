This document describes how to use the code from the FFN repository to build a
neuron reconstruction pipeline similar to the one described in
[the FFN paper](https://doi.org/10.1038/s41592-018-0049-4).
The recommended settings have been updated to reflect the current
best practices.

# Model training

Vanilla asynchronous SGD with `learning_rate = 0.001` is recommended for best
model quality. Batch sizes are generally determined by the amount of available
RAM, and are often quite small (2-4). Training with up to 32 GPUs normally provides
speed-ups compared to smalleer configurations. Beyond that, model convergence
suffers, despite an apparent faster processing speed as measured by optimization
steps/s.

## Data preprocessing

It is important to ensure any data preprocessing is identical between the
training and inference data sets. We recommend preprocessing all images
with CLAHE to reduce section-to-section and region-to-region contrast
variance. We also stress the importance of precise data alignment. Elastic
alignment is normally required for sufficient precision, with the possible
exception of some datasets acquired with FIB-SEM. The data can be considered
well-aligned when scrolling through the 'z' dimension appears visually smooth,
with no jitter, jumps, or drift.

## Annotation requirements

For best results, it is recommended to use at least 150 Mvx of dataset-specific
ground truth annotations. If this is not available, or cannot be easily
collected, a bootstrapping approach is usually effective and can reduce the
total effort needed to obtain the necessary annotations. To do so, an initial
FFN model is trained with the available labels, and used to generate a draft
segmentation. Neurons from this segmentation are then manually proofread, and
added to the ground truth set. Once enough data is collected, the FFN model
is retrained. This process can be done iteratively as needed, and through
the use of oversegmentation consesus, the proofreading can typically be
restricted to manual agglomeration of neuron fragments. Bootstrapping
is also recommended if initial manual annotations are not pixel-precise. For
best results, the segment mask should cover the entire process, i.e. cytoplasm,
organelles, and the membrane.

## Convergence and checkpoint selection

A good rule of thumb is to train the model long enough so that the number of training
FOVs seen matches the number of annotated voxels in the ground truth set.
The number of FOVs seen can be computed as `<batch_size> * <step_number>`.
For instance, with `batch_size=4` and 150 Mvx of training data, the model
should train for at least 37.5M steps.

During training, a model snapshot (checkpoint) is saved at a predefined frequency.
Once the model has finished training, checkpoint selection should be performed
using a separate validation dataset. For this we recommend running inference
over the dataset with as many saved checkpoints as possible, starting with the
latest ones available. The resulting segmentations should then be evaluated
using metrics of interest, and the best checkpoint used for large-scale inference.
The segmentation evaluation code is currently not part of the FFN repository.
We recommend evaluation with skeleton metrics to ensure that the selected
checkpoint is optimized for topological correctness.

# Segmentation inference

Segmentation inference is configured through an `InferenceRequest` protocol
buffer message. The following options are currently recommneded for FFN models:

```
 image_stddev: 33.0
 image_mean: 128.0
 seed_policy: "PolicyPeaks"
 inference_options {
   init_activation: 0.95
   pad_value: 0.5
   move_threshold: 0.6
   min_boundary_dist {x:2 y:2 z:1}
   segment_threshold: 0.6
   min_segment_size: 1000
 }
```

`pad_value` and `move_threshold` should match `--seed_pad` and `--threshold`
used during training. `image_stddev` and `image_mean` do not matter much
in practice, and can be kept at the recommended default values provided
the EM images were normalized with CLAHE and matching settings (`--image_mean`,
`--image_stddev`) were used during taining. `min_boundary_dist` can be
adjusted to `{x:2 y:2 z:2}` for datasets with isotropic voxels, or the
values can be reduced to 1 for a higher fill rate, at increased computational
cost.

## Performance considerations

When performing inference using accelerators such as GPUs, processing
multiple subvolumes on a single worker might be required to achieve full device utilization.
The number of concurrently executed inference calls is controlled by the
`batch_size` field of the `InferenceRequest`, starting a single `inference.Runner`,
and calling `runnner.run()` from separate threads (typically, in a 1:1 thread-to-subvolume
configuration). To further hide the host-side overhead, more than `batch_size`
subvolumes can be processed at the same time by the same worker.

Further gains in inference speed can be obtained by using a `tf.Session` created
with `graph_options.rewrite_options.auto_mixed_precision=1` set in `tf.ConfigProto`.
This automatically converts selected operations in the graph to operate on
float16 data. This conversion can provide 2x+ speedup, at the cost of
a slightly increased merge error rate. The magnitude of the accuracy drop might
be dataset specific, so we advise experiments on a validation dataset before
any large-scale deployment.

## Distributed processing and assembly

The code contained in this repository contains functions necessary to generate
segmentations on a per-subvolume basis ('subvolume' understood as a region
of the dataset typically a couple hundred voxels on the side), stored as
npz files. These operations are embarassingly parallel with no dependencies
between subvolume from the same processing step. This repository does not
contain support for workload distribution, as we expect it
to be highly specific to the computing environment utilized by the user. We
recommend a simple task queue system with distributed workers as the processing
model.

Once the subvolume results are available, a global segmentation still needs to be assembled
out of them, with the local subvolume-specific ID spaces reconciled into a single
global ID space, and the result saved in some format such as HDF5. This
reconcilation process involves maintaining a union-find data structure,
potentially in a distributed system. This functionality is currently *not
implemented* in this repository. The FFN paper used a custom implementation of
the algorithm described in [Rastogi et al](https://arxiv.org/abs/1203.5387) in order to
establish the global ID space, and a predecessor of [TensorStore](
https://github.com/google/tensorstore) to store the reconciled data.

## Consensus

Oversegmentation consensus can be used to reduce the false merger rate at the
supervoxel level and is run using `ffn/inference/consensus.py:compute_consensus()`, and
configured through the `ConsensusRequest` protocol buffer message. We
recommend setting `split_min_size` to 1000.

Consensus can in principle be computed between any two FFN segmentations (e.g.
generated using different model checkpoints), but the most typical application
is the forward-reverse seed order consensus, in which the first segmentation
uses `PolicyPeaks`, and the second segmentation uses `PolicyInverseOrigins` with
`segmentation_dir` pointing to the location of the first segmentation. Note that
the segmentations used as consesnsus inputs do not need to be assembled (i.e.
the consensus code processes data from the per-subvolume .npz files directly).

# Agglomeration through resegmentation

FFN models can be used to agglomerate segments through a process called
resegmentation, which involves recomputing a subset of the segmentation from scratch twice,
starting from two different seeds. The segmentation is restricted to a small
subvolume centered at a decision point, which is typically chosen as a point
of maximum proximity of the two original segments. The results of resegmentation
are then compared to the original segmentation to compute compatibility scores.

Similarly to segmentation inference, this repository provides the library
functions necessary to select decision points, as well as to run and score
resegmentation, but does not provide a way to distribute and manage the work. For the latter
we recommend a task-based queue system, similarly to segmentation inference.

## Decision points

Decision points are locations of maximum proximity of two objects in the
segmentation, and can be computed using the `decision_point.find_decision_points()`
function. This is usually done by running the function over overlapping subvolumes
of the assembled and reconciled base segmentation.

## Resegmentation

Resegmentation can be executed by calling `resegmentation.py:process_point()`
and is configured through a `ResegmentationRequest` protocol buffer message.
Within the request, `inference` specifies the FFN inference configuration,
and should generally be set to the settings used for base segmentation
inference. Importantly, `inference.init_segmentation`, should point to the
segmentation volume for which decision points were computed.

A single resegmentation request can be used for many decision points, with
the location of these points together with the corresponding segment ID pairs
stored within the repeated `points` field.

Typical resegmentation-specific settings (shown below for a volume with
2x Z anisotropy are):
```
radius { x: 50 y: 50 z: 25 }
analysis_radius { x: 34 y: 34: z: 17 }
max_retry_iters: 8
exclusion_radius: { x: 8 y: 8 z: 4 }
segment_recovery_fraction: 0.6
```

Increasing `max_retry_iters` or `segment_recovery_fraction`, or decreasing
`exclusion_radius` will cause the resegmentation code to spend more effort
searching for a potential connection for any given segment pair, at the
expense of increased compute cost.


## Scoring

Resegmentation results can be postprocessed with 
`resegmentation_analysis.py:evaluate_pair_resegmentation()`, which produces
a `PairResegmentationResult` proto with the scores.

The map between the score vector and a pair merge probability in principle needs
to be calibrated separately on a per-dataset basis. We have observed the following
rule to be a good conversative (i.e., minimizing false mergers) starting point:

 * `eval.iou > 0.8`
 * `eval.from_(a|b)_segmment_(a|b)_consistency > 0.6`
 * `eval.from_a.deleted_voxels / eval.from_a.num_voxels < 0.02` or
   `eval.from_b.deleted_voxels / eval.from_b.num_voxels < 0.02`

If all of the above conditions are fulfilled, the corresponding segment pair
can be accepted as merged.

Intuitively, `iou` measures segmentation consistency between the two different
seeds, `from_X.segment_Y_consistency` measures how well resegmentation reproduced
segment `Y` when seeded from `X`, and the 'deleted voxel fraction' `deleted_voxels / num_voxels`
is a measure of inference-time model confusion (i.e. inconsistent voxel labeling at different
time steps).
