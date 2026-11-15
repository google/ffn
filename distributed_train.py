#!/usr/bin/env python3
"""
Launch FFN training using TensorFlow 1-style asynchronous parameter servers.

One process is started as a CPU-only parameter server.
One worker process is started for each GPU.
"""

from __future__ import annotations

import json
import random
import time

import tensorflow.compat.v1 as tf

# The current FFN training code uses placeholders, sessions, and TF1 queues.
tf.disable_v2_behavior()

from absl import app
from absl import flags

import train_mod as ffn_train
from ffn.training.import_util import import_symbol


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "job_name",
    None,
    'Distributed job name: "ps" or "worker".',
)
flags.DEFINE_integer(
    "task_index",
    0,
    "Task index within the selected job.",
)
flags.DEFINE_list(
    "ps_hosts",
    None,
    "Comma-separated parameter-server addresses.",
)
flags.DEFINE_list(
    "worker_hosts",
    None,
    "Comma-separated worker addresses.",
)


def _validate_worker_flags() -> None:
    required = (
        "train_coords",
        "data_volumes",
        "label_volumes",
        "model_name",
        "model_args",
    )
    missing = [
        name
        for name in required
        if getattr(FLAGS, name, None) in (None, "")
    ]
    if missing:
        raise app.UsageError(
            "Missing worker arguments: " + ", ".join(missing)
        )


def main(argv: list[str]) -> None:
    del argv

    if FLAGS.job_name not in ("ps", "worker"):
        raise app.UsageError('--job_name must be "ps" or "worker".')

    if not FLAGS.ps_hosts:
        raise app.UsageError("--ps_hosts is required.")

    if not FLAGS.worker_hosts:
        raise app.UsageError("--worker_hosts is required.")

    cluster = tf.train.ClusterSpec(
        {
            "ps": list(FLAGS.ps_hosts),
            "worker": list(FLAGS.worker_hosts),
        }
    )

    server_config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    server_config.gpu_options.allow_growth = True

    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        protocol="grpc",
        config=server_config,
    )

    if FLAGS.job_name == "ps":
        # The parameter server owns the shared variables and remains active.
        server.join()
        return

    _validate_worker_flags()

    # Pass the distributed configuration into the existing FFN train.py.
    FLAGS.master = server.target
    FLAGS.task = FLAGS.task_index
    FLAGS.ps_tasks = len(FLAGS.ps_hosts)

    # Match the task-dependent seeding behavior of the original train.py.
    seed = int(time.time() + FLAGS.task * 24 * 60 * 60)
    random.seed(seed)

    model_class = import_symbol(FLAGS.model_name)

    ffn_train.train_ffn(
        model_class,
        batch_size=FLAGS.batch_size,
        **json.loads(FLAGS.model_args),
    )


if __name__ == "__main__":
    app.run(main)
    
