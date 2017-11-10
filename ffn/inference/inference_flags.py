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
"""Helpers to initialize protos from flag values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format
from absl import flags
from . import inference_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string('inference_request', None,
                    'InferenceRequest proto in text format.')
flags.DEFINE_string('inference_options', None,
                    'InferenceOptions proto in text format.')


def options_from_flags():
  options = inference_pb2.InferenceOptions()
  if FLAGS.inference_options:
    text_format.Parse(FLAGS.inference_options, options)

  return options


def request_from_flags():
  request = inference_pb2.InferenceRequest()
  if FLAGS.inference_request:
    text_format.Parse(FLAGS.inference_request, request)

  return request
