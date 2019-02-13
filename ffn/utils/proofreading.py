"""Utilities for small-scale proofreading in Neuroglancer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import defaultdict
import copy
import itertools
import networkx as nx
import neuroglancer


class Base(object):
  """Base class for proofreading workflows.

  To use, define a subclass overriding the `set_init_state` method to
  provide initial Neuroglancer settings.

  The segmentation volume needs to be called `seg`.
  """

  def __init__(self, num_to_prefetch=10):
    self.viewer = neuroglancer.Viewer()
    self.num_to_prefetch = num_to_prefetch

    self.todo = []  # items are lists of segment IDs
    self.index = 0
    self.batch = 1
    self.apply_equivs = False
    self.locations = None

    self.set_init_state()

  def set_init_state(self):
    raise NotImplementedError()

  def update_msg(self, msg):
    with self.viewer.config_state.txn() as s:
      s.status_messages['status'] = msg

  def update_segments(self, segments, loc=None, layer='seg'):
    s = copy.deepcopy(self.viewer.state)
    l = s.layers[layer]
    l.segments = segments

    if not self.apply_equivs:
      l.equivalences.clear()
    else:
      l.equivalences.clear()
      for a in self.todo[self.index:self.index + self.batch]:
        l.equivalences.union(*a)

    if loc is not None:
      s.navigation.pose.position.voxel_coordinates = loc

    self.viewer.set_state(s)

  def toggle_equiv(self):
    self.apply_equivs = not self.apply_equivs
    self.update_batch()

  def batch_dec(self):
    self.batch //= 2
    self.batch = max(self.batch, 1)
    self.update_batch()

  def batch_inc(self):
    self.batch *= 2
    self.update_batch()

  def next_batch(self):
    self.index += self.batch
    self.index = min(self.index, len(self.todo) - 1)
    self.prefetch()
    self.update_batch()

  def prev_batch(self):
    self.index -= self.batch
    self.index = max(0, self.index)
    self.update_batch()

  def list_segments(self, index=None):
    if index is None:
      index = self.index
    return list(set(itertools.chain(*self.todo[index:index + self.batch])))

  def custom_msg(self):
    return ''

  def update_batch(self, update=True):
    if self.batch == 1 and self.locations is not None:
      loc = self.locations[self.index]
    else:
      loc = None
    self.update_segments(self.list_segments(), loc)
    self.update_msg('index:%d/%d  batch:%d  %s' % (self.index, len(
        self.todo), self.batch, self.custom_msg()))

  def prefetch(self):
    prefetch_states = []
    for i in range(self.num_to_prefetch):
      idx = self.index + (i + 1) * self.batch
      if idx >= len(self.todo):
        break
      prefetch_state = copy.deepcopy(self.viewer.state)
      prefetch_state.layers['seg'].segments = self.list_segments(idx)
      prefetch_state.layout = '3d'
      prefetch_states.append(prefetch_state)

    with self.viewer.config_state.txn() as s:
      s.prefetch = [
          neuroglancer.PrefetchState(state=prefetch_state, priority=-i)
          for i, prefetch_state in enumerate(prefetch_states)
      ]


class ObjectReview(Base):
  """Base class for rapid (agglomerated) object review.

  To achieve good throughput, smaller objects are usually reviewed in
  batches.
  """

  def __init__(self, objects, bad, num_to_prefetch=10, locations=None):
    """Constructor.

    Args:
      objects: iterable of object IDs or iterables of object IDs. In the latter
        case it is assumed that every iterable forms a group of objects to be
        agglomerated together.
      bad: set in which to store objects or groups of objects flagged as bad.
      num_to_prefetch: number of items from `objects` to prefetch
      locations: iterable of xyz tuples of length len(objects). If specified,
        the cursor will be automaticaly moved to the location corresponding to
        the current object if batch == 1.
    """
    super(ObjectReview, self).__init__(num_to_prefetch=num_to_prefetch)

    self.todo = []
    for o in objects:
      if isinstance(o, collections.Iterable):
        self.todo.append(o)
      else:
        self.todo.append([o])

    if locations is not None:
      assert len(self.todo) == len(locations)
      self.locations = list(locations)

    self.bad = bad

    self.viewer.actions.add('next-batch', lambda s: self.next_batch())
    self.viewer.actions.add('prev-batch', lambda s: self.prev_batch())
    self.viewer.actions.add('dec-batch', lambda s: self.batch_dec())
    self.viewer.actions.add('inc-batch', lambda s: self.batch_inc())
    self.viewer.actions.add('mark-bad', lambda s: self.mark_bad())
    self.viewer.actions.add('mark-removed-bad', lambda s: self.mark_removed_bad())
    self.viewer.actions.add('toggle-equiv', lambda s: self.toggle_equiv())

    with self.viewer.config_state.txn() as s:
      s.input_event_bindings.viewer['keyj'] = 'next-batch'
      s.input_event_bindings.viewer['keyk'] = 'prev-batch'
      s.input_event_bindings.viewer['keym'] = 'dec-batch'
      s.input_event_bindings.viewer['keyp'] = 'inc-batch'
      s.input_event_bindings.viewer['keyv'] = 'mark-bad'
      s.input_event_bindings.viewer['keyt'] = 'toggle-equiv'
      s.input_event_bindings.viewer['keya'] = 'mark-removed-bad'

    self.update_batch()

  def custom_msg(self):
    return 'num_bad: %d' % len(self.bad)

  def mark_bad(self):
    if self.batch > 1:
      self.update_msg('decrease batch to 1 to mark objects bad')
      return

    sids = self.todo[self.index]
    if len(sids) == 1:
      self.bad.add(list(sids)[0])
    else:
      self.bad.add(frozenset(sids))

    self.update_msg('marked bad: %r' % (sids, ))
    self.next_batch()

  def mark_removed_bad(self):
    original = set(self.list_segments())
    new_bad = original - set(self.viewer.state.layers['seg'].segments)
    if new_bad:
      self.bad |= new_bad
      self.update_msg('marked bad: %r' % (new_bad, ))


class ObjectClassification(Base):
  """Base class for object classification."""

  def __init__(self, objects, key_to_class, num_to_prefetch=10, locations=None):
    """Constructor.

    Args:
      objects: iterable of object IDs
      key_to_class: dict mapping keys to class labels
      num_to_prefetch: number of `objects` to prefetch
    """
    super(ObjectClassification, self).__init__(num_to_prefetch=num_to_prefetch)
    self.todo = [[o] for o in objects]
    self.results = defaultdict(set)  # class -> ids

    if locations is not None:
      assert len(self.todo) == len(locations)
      self.locations = list(locations)

    self.viewer.actions.add('mr-next-batch', lambda s: self.next_batch())
    self.viewer.actions.add('mr-prev-batch', lambda s: self.prev_batch())
    self.viewer.actions.add('unclassify', lambda s: self.classify(None))

    for key, cls in key_to_class.items():
      self.viewer.actions.add(
          'classify-%s' % cls, lambda s, cls=cls: self.classify(cls))

    with self.viewer.config_state.txn() as s:
      for key, cls in key_to_class.items():
        s.input_event_bindings.viewer['key%s' % key] = 'classify-%s' % cls

      # Navigation without classification.
      s.input_event_bindings.viewer['keyj'] = 'mr-next-batch'
      s.input_event_bindings.viewer['keyk'] = 'mr-prev-batch'
      s.input_event_bindings.viewer['keyv'] = 'unclassify'

    self.update_batch()

  def custom_msg(self):
    return ' '.join('%s:%d' % (k, len(v)) for k, v in self.results.items())

  def classify(self, cls):
    sid = list(self.todo[self.index])[0]
    for v in self.results.values():
      v -= set([sid])

    if cls is not None:
      self.results[cls].add(sid)

    self.next_batch()


class GraphUpdater(Base):
  """Base class for agglomeration graph modification.

  Usage:
    * splitting
      1) select merged objects (start with a supervoxel, then press 'c')
      2) shift-click on two supervoxels that should be separated; a new layer
         will be displayed showing the supervoxels along the shortest path
         between selected objects
      3) use '[' and ']' to restrict the path so that the displayed supervoxels
         are not wrongly merged
      4) press 's' to remove the edge next to the last shown one from the
         agglomeration graph

    * merging
      1) select segments to be merged
      2) press 'm'

  Press 'c' to add any supervoxels connected to the ones currently displayed
  (according to the current state of the agglomeraton graph).
  """

  def __init__(self, graph, objects, bad):
    """Constructor.

    Args:
      graph: networkx Graph object to modify
      objects: iterable of object IDs or iterables of object IDs
      bad: set in which to store objects flagged as bad
    """
    super(GraphUpdater, self).__init__()
    self.graph = graph
    self.split_objects = []
    self.split_path = []
    self.split_index = 1

    self.todo = []
    for o in objects:
      if isinstance(o, collections.Iterable):
        self.todo.append(o)
      else:
        self.todo.append([o])

    self.bad = bad
    self.viewer.actions.add('add-ccs', lambda s: self.add_ccs())
    self.viewer.actions.add('clear-splits', lambda s: self.clear_splits())
    self.viewer.actions.add('add-split', self.add_split)
    self.viewer.actions.add('accept-split', lambda s: self.accept_split())
    self.viewer.actions.add('split-inc', lambda s: self.inc_split())
    self.viewer.actions.add('split-dec', lambda s: self.dec_split())
    self.viewer.actions.add('merge-segments', lambda s: self.merge_segments())
    self.viewer.actions.add('mark-bad', lambda s: self.mark_bad())
    self.viewer.actions.add('next-batch', lambda s: self.next_batch())
    self.viewer.actions.add('prev-batch', lambda s: self.prev_batch())

    with self.viewer.config_state.txn() as s:
      s.input_event_bindings.viewer['keyj'] = 'next-batch'
      s.input_event_bindings.viewer['keyk'] = 'prev-batch'
      s.input_event_bindings.viewer['keyc'] = 'add-ccs'
      s.input_event_bindings.viewer['keya'] = 'clear-splits'
      s.input_event_bindings.viewer['keym'] = 'merge-segments'
      s.input_event_bindings.viewer['bracketleft'] = 'split-dec'
      s.input_event_bindings.viewer['bracketright'] = 'split-inc'
      s.input_event_bindings.viewer['keys'] = 'accept-split'
      s.input_event_bindings.data_view['shift+mousedown0'] = 'add-split'
      s.input_event_bindings.viewer['keyv'] = 'mark-bad'

    with self.viewer.txn() as s:
      s.layers['split'] = neuroglancer.SegmentationLayer(
          source=s.layers['seg'].source)
      s.layers['split'].visible = False

  def merge_segments(self):
    sids = [sid for sid in self.viewer.state.layers['seg'].segments if sid > 0]
    self.graph.add_edges_from(zip(sids, sids[1:]))

  def update_split(self):
    s = copy.deepcopy(self.viewer.state)
    s.layers['split'].segments = list(self.split_path)[:self.split_index]
    self.viewer.set_state(s)

  def inc_split(self):
    self.split_index = min(len(self.split_path), self.split_index + 1)
    self.update_split()

  def dec_split(self):
    self.split_index = max(1, self.split_index - 1)
    self.update_split()

  def add_ccs(self):
    curr = set(self.viewer.state.layers['seg'].segments)
    for sid in self.viewer.state.layers['seg'].segments:
      if sid in self.graph:
        curr |= set(nx.node_connected_component(self.graph, sid))

    self.update_segments(curr)

  def accept_split(self):
    edge = self.split_path[self.split_index - 1:self.split_index + 1]
    if len(edge) < 2:
      return

    self.graph.remove_edge(edge[0], edge[1])
    self.clear_splits()

  def clear_splits(self):
    self.split_objects = []
    self.update_msg('splits cleared')

    s = copy.deepcopy(self.viewer.state)
    s.layers['split'].visible = False
    s.layers['seg'].visible = True
    self.viewer.set_state(s)

  def start_split(self):
    self.split_path = nx.shortest_path(self.graph, self.split_objects[0],
                                       self.split_objects[1])
    self.split_index = 1
    self.update_msg(
        'splitting: %s' % ('-'.join(str(x) for x in self.split_path)))

    s = copy.deepcopy(self.viewer.state)
    s.layers['seg'].visible = False
    s.layers['split'].visible = True
    self.viewer.set_state(s)
    self.update_split()

  def add_split(self, s):
    if len(self.split_objects) < 2:
      self.split_objects.append(s.selected_values['seg'])
    self.update_msg(
        'split: %s' % (':'.join(str(x) for x in self.split_objects)))

    if len(self.split_objects) == 2:
      self.start_split()

  def mark_bad(self):
    if self.batch > 1:
      self.update_msg('decrease batch to 1 to mark objects bad')
      return

    sids = self.todo[self.index]
    if len(sids) == 1:
      self.bad.add(list(sids)[0])
    else:
      self.bad.add(frozenset(sids))

    self.update_msg('marked bad: %r' % (sids, ))
    self.next_batch()
