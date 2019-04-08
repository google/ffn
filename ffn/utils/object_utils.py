# Copyright 2018 Google Inc.
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

"""Utilities for object processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import pandas as pd
import logging


def load_equivalences(paths):
  """Loads equivalences from a text file.

  Args:
    paths: sequence of paths to the text files of equivalences; id0,id1 per
      line, or id0,id1,x,y,z.

  Returns:
    NX graph object representing the equivalences
  """
  equiv_graph = nx.Graph()

  for path in paths:
    with open(path, "r") as f:
      reader = pd.read_csv(
          f, sep=",", engine="c", comment="#", chunksize=4096, header=None)
      for chunk in reader:
        if len(chunk.columns) not in (2, 5):
          logging.critical("Unexpected # of columns (%d), want 2 or 5",
                           len(chunk.columns))

        edges = chunk.values[:, :2]
        equiv_graph.add_edges_from(edges)

  return equiv_graph
