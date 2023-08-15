# Copyright 2018-2023 Google Inc.
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

"""Utilities for small-scale proofreading in Neuroglancer."""

import collections
from collections import defaultdict
import copy
import itertools
import threading

import networkx as nx
import neuroglancer

from typing import Union, Optional, Iterable, Any


class Base:
  """Base class for proofreading workflows.

  To use, define a subclass overriding the `set_init_state` method to provide
  initial Neuroglancer settings. The segmentation volume needs to be called
  `seg`.
  """

  def __init__(self,
               num_to_prefetch: int = 10,
               locations: Optional[Iterable[tuple[int, int, int]]] = None,
               objects: Optional[
                 Union[dict[str, Any], Iterable[int]]] = None):
    """Initializes the Base class for proofreading.

    Args:
        num_to_prefetch: Number of items to prefetch.
        locations: List of xyz coordinates corresponding to object locations.
        objects: Object IDs or a dictionary mapping layer names to object IDs.
    """
    self.viewer = neuroglancer.Viewer()
    self.num_to_prefetch = num_to_prefetch

    self.managed_layers = set(['seg'])
    self.todo = []  # items are maps from layer name to lists of segment IDs
    if objects is not None:
      self._set_todo(objects)

    self.index = 0
    self.batch = 1
    self.apply_equivs = False

    if locations is not None:
      self.locations = list(locations)
      assert len(self.todo) == len(locations)
    else:
      self.locations = None

    self.set_init_state()

  def _set_todo(self, objects: Union[list[str, Any], Iterable[int]]) -> None:
    """Private method to set the todo list."""
    for o in objects:
      if isinstance(o, collections.abc.Mapping):
        self.todo.append(o)
        self.managed_layers |= set(o.keys())
      elif isinstance(o, collections.abc.Iterable):
        self.todo.append({'seg': o})
      else:
        self.todo.append({'seg': [o]})

  def set_init_state(self) -> None:
    """Sets the initial state for Neuroglancer.
    Subclasses should override this method.
    """
    raise NotImplementedError()

  def update_msg(self, msg: str) -> None:
    """Updates the status message in Neuroglancer viewer."""
    with self.viewer.config_state.txn() as s:
      s.status_messages['status'] = msg

  def update_segments(self,
                      segments: list[int],
                      loc: Optional[tuple[int, int, int]] = None,
                      layer: str = 'seg') -> None:
    """Updates segments in Neuroglancer viewer.

    Args:
        segments: List of segment IDs to update.
        loc: 3D coordinates to set the viewer to.
        layer: Layer name in Neuroglancer to be updated.
    """
    s = copy.deepcopy(self.viewer.state)
    l = s.layers[layer]
    l.segments = segments

    if not self.apply_equivs:
      l.equivalences.clear()
    else:
      l.equivalences.clear()
      for a in self.todo[self.index:self.index + self.batch]:
        a = [aa[layer] for aa in a]
        l.equivalences.union(*a)

    if loc is not None:
      s.position = loc

    self.viewer.set_state(s)

  def toggle_equiv(self) -> None:
    """Toggle the apply equivalence flag and update the batch."""
    self.apply_equivs = not self.apply_equivs
    self.update_batch()

  def batch_dec(self) -> None:
    """Decrease the batch size by half and update the batch."""
    self.batch //= 2
    self.batch = max(self.batch, 1)
    self.update_batch()

  def batch_inc(self) -> None:
    """Increase the batch size by double and update the batch."""
    self.batch *= 2
    self.update_batch()

  def next_batch(self) -> None:
    """Move to the next batch of segments and update the viewer."""
    self.index += self.batch
    self.index = min(self.index, len(self.todo) - 1)
    self.prefetch()
    self.update_batch()

  def prev_batch(self) -> None:
    """Move to the previous batch of segments and update the viewer."""
    self.index -= self.batch
    self.index = max(0, self.index)
    self.update_batch()

  def list_segments(self,
                    index: Optional[int] = None,
                    layer: str = 'seg') -> list[int]:
    """Get a list of segment IDs for a given index and layer.

    Args:
        index: Index of segments to list.
        layer: Layer name to list the segments from.

    Returns:
        List of segment IDs.
    """
    if index is None:
      index = self.index
    return list(
      set(
        itertools.chain(
          *[x[layer] for x in self.todo[index:index + self.batch]])))

  def custom_msg(self) -> str:
    """Generate a custom message for the current state.

    Returns:
        A custom message string.
    """
    return ''

  def update_batch(self) -> None:
    """Update the segments displayed in the viewer based on batch settings."""
    if self.batch == 1 and self.locations is not None:
      loc = self.locations[self.index]
    else:
      loc = None

    for layer in self.managed_layers:
      self.update_segments(self.list_segments(layer=layer), loc,
                           layer=layer)
    self.update_msg('index:%d/%d  batch:%d  %s' %
                    (self.index, len(self.todo), self.batch,
                     self.custom_msg()))

  def prefetch(self) -> None:
    """Pre-fetch the segments for smoother navigation in the viewer."""
    prefetch_states = []
    for i in range(self.num_to_prefetch):
      idx = self.index + (i + 1) * self.batch
      if idx >= len(self.todo):
        break
      prefetch_state = copy.deepcopy(self.viewer.state)
      for layer in self.managed_layers:
        prefetch_state.layers[layer].segments = self.list_segments(
          idx, layer=layer)
      prefetch_state.layout = '3d'
      if self.locations is not None:
        prefetch_state.position = self.locations[idx]

      prefetch_states.append(prefetch_state)

    with self.viewer.config_state.txn() as s:
      s.prefetch = [
        neuroglancer.PrefetchState(state=prefetch_state, priority=-i)
        for i, prefetch_state in enumerate(prefetch_states)
      ]

  def get_cursor_position(self,
                          action_state: neuroglancer.viewer_config_state.ActionState):
    """Return coordinates of the cursor position from a neuroglancer action state

    Args:
        action_state : Neuroglancer action state

    Returns:
        (x, y, z) cursor position
    """
    try:
      cursor_position = [int(x) for x in
                         action_state.mouse_voxel_coordinates]
    except Exception:
      self.update_msg('cursor misplaced')
      return

    return cursor_position


class ObjectReview(Base):
  """Base class for rapid (agglomerated) object review.

  To achieve good throughput, smaller objects are usually reviewed in
  batches.
  """

  def __init__(self,
               objects: Iterable,
               bad: list,
               num_to_prefetch: int = 10,
               locations: Optional[Iterable[tuple[int, int, int]]] = None):
    """Constructor.

    Args:
      objects: iterable of object IDs or iterables of object IDs. In the latter
        case it is assumed that every iterable forms a group of objects to be
        agglomerated together.
      bad: set in which to store objects or groups of objects flagged as bad.
      num_to_prefetch: number of items from `objects` to prefetch
      locations: iterable of xyz tuples of length len(objects). If specified,
        the cursor will be automatically moved to the location corresponding to
        the current object if batch == 1.
    """
    super().__init__(
      num_to_prefetch=num_to_prefetch, locations=locations,
      objects=objects)
    self.bad = bad

    self.set_keybindings()

    self.update_batch()

  def set_keybindings(self) -> None:
    """Set key bindings for the viewer."""
    self.viewer.actions.add('next-batch', lambda s: self.next_batch())
    self.viewer.actions.add('prev-batch', lambda s: self.prev_batch())
    self.viewer.actions.add('dec-batch', lambda s: self.batch_dec())
    self.viewer.actions.add('inc-batch', lambda s: self.batch_inc())
    self.viewer.actions.add('mark-bad', lambda s: self.mark_bad())
    self.viewer.actions.add('mark-removed-bad',
                            lambda s: self.mark_removed_bad())
    self.viewer.actions.add('toggle-equiv', lambda s: self.toggle_equiv())

    with self.viewer.config_state.txn() as s:
      s.input_event_bindings.viewer['keyj'] = 'next-batch'
      s.input_event_bindings.viewer['keyk'] = 'prev-batch'
      s.input_event_bindings.viewer['keym'] = 'dec-batch'
      s.input_event_bindings.viewer['keyp'] = 'inc-batch'
      s.input_event_bindings.viewer['keyv'] = 'mark-bad'
      s.input_event_bindings.viewer['keyt'] = 'toggle-equiv'
      s.input_event_bindings.viewer['keya'] = 'mark-removed-bad'

  def custom_msg(self) -> str:
    """Construct a custom message for the current state.

    Returns:
        A formatted message indicating the number of bad objects.
    """
    return 'num_bad: %d' % len(self.bad)

  def mark_bad(self) -> None:
    """Mark an object or group of objects as bad.

    If the batch size is greater than 1, the user is prompted to decrease
    the batch size.
    """
    if self.batch > 1:
      self.update_msg('decrease batch to 1 to mark objects bad')
      return

    sids = self.todo[self.index]['seg']
    if len(sids) == 1:
      self.bad.add(list(sids)[0])
    else:
      self.bad.add(frozenset(sids))

    self.update_msg('marked bad: %r' % (sids,))
    self.next_batch()

  def mark_removed_bad(self) -> None:
    """From the set of original objects mark those bad that are not displayed.
    Update the message with the IDs of the newly marked bad objects.
    """
    original = set(self.list_segments())
    new_bad = original - set(self.viewer.state.layers['seg'].segments)
    if new_bad:
      self.bad |= new_bad
      self.update_msg('marked bad: %r' % (new_bad,))


class ObjectReviewStoreLocation(ObjectReview):
  """Class to mark and store locations of errors in the segmentation

  To mark a merger, move the cursor to a spot of the false merger and press 'w'.
  Then, move the cursor to a spot within the object that should belong to a
  separate object and press 'shift + W'. Yellow point annotations indicate the
  merger. For split errors, proceed in similar manner but press 'd' and
  'shift + D', which will display blue annotations.
  Marked locations can be deleted either by pressing 'ctrl + Z' (to delete the
  last marked location) or by hovering the cursor over one of the point
  annotations and pressing 'ctrl + v'.

  Attributes:
      seg_error_coordinates: A mapping of annotation identifier substrings to
                             error coordinate pairs.
          Example: {'m0': [[x1,y1,z1],[x2,y2,z2]], 's0':[[x1,y1,z1],[x2,y2,z2]], ...}
          - Keys starting with 'm' indicate merge errors.
          - Keys starting with 's' indicate split errors.
      temp_coord_list: Temporary storage for coordinates.
  """

  def __init__(self,
               objects: list,
               bad: list,
               seg_error_coordinates: Optional[
                 list[str, list[list[int]]]] = {},
               load_annotations: bool = False) -> None:
    """Initialize the ObjectReviewStoreLocation class.

    Args:
        objects: A list of objects.
        bad: A list of bad objects or markers.
        seg_error_coordinates: A dictionary of error coordinates.
        load_annotations: A flag to indicate if annotations should be loaded.
    """
    super(ObjectReviewStoreLocation, self).__init__(objects, bad)
    self.seg_error_coordinates = seg_error_coordinates
    if load_annotations and seg_error_coordinates:
      for k, v in seg_error_coordinates.items():
        self.annotate_error_locations(v, k)
    self.temp_coord_list = []

  def set_keybindings(self) -> None:
    """Set key bindings for the viewer."""
    super().set_keybindings()
    self.viewer.actions.add('merge0',
                            lambda s: self.store_error_location(s, index=0,
                                                                mode='merger'))
    self.viewer.actions.add('merge1',
                            lambda s: self.store_error_location(s, index=1,
                                                                mode='merger'))
    self.viewer.actions.add('split0',
                            lambda s: self.store_error_location(s, index=0,
                                                                mode='split'))
    self.viewer.actions.add('split1',
                            lambda s: self.store_error_location(s, index=1,
                                                                mode='split'))
    self.viewer.actions.add('delete_from_annotation',
                            self.delete_location_from_annotation)
    self.viewer.actions.add('delete_last_entry',
                            lambda s: self.delete_last_location())

    with self.viewer.config_state.txn() as s:
      s.input_event_bindings.viewer['keyw'] = 'merge0'
      s.input_event_bindings.viewer['shift+keyw'] = 'merge1'
      s.input_event_bindings.viewer['keyd'] = 'split0'
      s.input_event_bindings.viewer['shift+keyd'] = 'split1'
      s.input_event_bindings.viewer[
        'control+keyv'] = 'delete_from_annotation'
      s.input_event_bindings.viewer[
        'control+keyz'] = 'delete_last_entry'

  def get_id(self, mode: str) -> str:
    """Generate a unique identifier for an error based on its type.

    Args:
        mode: Error type, either 'merge' or 'split'.

    Returns:
        A unique identifier string.
    """
    id_ = mode[0]
    if any(self.seg_error_coordinates):
      counter = int(
        max([x[1:] for x in self.seg_error_coordinates.keys()])) + 1
    else:
      counter = 0
    id_ = id_ + str(counter)
    return id_

  def store_error_location(self,
                           action_state: neuroglancer.viewer_config_state.ActionState,
                           mode: str,
                           index: int = 0) -> None:
    """Store error locations.

    Args:
        action_state: State of the viewer during the action.
        mode: Type of the error ('merger' or 'split').
        index: Indicates if it's the first or second coordinate (0 or 1).
    """
    location = self.get_cursor_position(action_state)
    if location is None:
      return

    if index == 1 and not self.temp_coord_list:
      self.update_msg('You have not entered a first coord yet')
      return

    if index == 0 and self.temp_coord_list:
      self.temp_coord_list = []

    self.temp_coord_list.append(location)

    if index == 1:
      if self.temp_coord_list[0] == self.temp_coord_list[1]:
        self.update_msg(
          'You entered the same coordinate twice. Try again!')
        self.temp_coord_list = []
        return

      identifier = self.get_id(mode=mode)
      self.seg_error_coordinates.update(
        {identifier: self.temp_coord_list})
      self.annotate_error_locations(self.temp_coord_list, identifier)
      self.temp_coord_list = []

  def annotate_error_locations(self,
                               coordinate_lst: list[list[int]],
                               id_: str) -> None:
    """Annotate the error locations in the viewer.

    Args:
        coordinate_lst: List of coordinates to be annotated.
        id_: Unique identifier for the error.
    """
    for i, coord in enumerate(coordinate_lst):
      annotation_id = id_ + f'_{i}'
      self.mk_point_annotation(coord, annotation_id)

  def mk_point_annotation(self,
                          coordinate: list[int],
                          annotation_id: str) -> None:
    """Create a point annotation in the viewer.

    Args:
        coordinate: 3D coordinate of the annotation point.
        annotation_id: Unique identifier for the annotation.
    """
    if annotation_id.startswith('m'):
      color = '#fae505'
    else:
      color = '#05f2fa'
    annotation = neuroglancer.PointAnnotation(id=annotation_id,
                                              point=coordinate,
                                              props=[color])
    with self.viewer.txn() as s:
      annotations = s.layers['annotation'].annotations
      annotations.append(annotation)

  def get_annotation_id(self,
                        action_state: neuroglancer.viewer_config_state.ActionState) -> \
          Optional[str]:
    """Retrieve the ID of a selected annotation.

    Args:
        action_state: neuroglancer.viewer_config_state.ActionState.

    Returns:
        The selected object's ID or None if retrieval fails.
    """
    try:
      selection_state = action_state.selected_values[
        'annotation'].to_json()
      selected_object = selection_state['annotationId']
    except Exception:
      self.update_msg('Could not retrieve annotation id')
      return

    return selected_object

  def delete_location_from_annotation(self,
                                      action_state: neuroglancer.viewer_config_state.ActionState) -> None:
    """Delete the error location pair associated with the annotation at the cursor position

    Args:
        action_state: State of the viewer during the action.
    """
    id_ = self.get_annotation_id(action_state)
    if id_ is None:
      return

    target_key = id_[:2]
    del self.seg_error_coordinates[target_key]

    to_remove = [target_key + '_0', target_key + '_1']
    self.delete_annotation(to_remove)

  def delete_annotation(self, to_remove: list[str]) -> None:
    """Delete specified annotations from the viewer.

    Args:
        to_remove: list of annotation IDs to be removed.
    """
    with self.viewer.txn() as s:
      annotations = s.layers['annotation'].annotations
      annotations = [a for a in annotations if a.id not in to_remove]
      s.layers['annotation'].annotations = annotations

  def delete_last_location(self):
    """Delete the last error location pair tagged."""
    last_key = list(self.seg_error_coordinates.keys())[-1]
    del self.seg_error_coordinates[last_key]

    to_remove = [last_key + '_0', last_key + '_1']
    self.delete_annotation(to_remove)


class ObjectClassification(Base):
  """Base class for object classification."""

  def __init__(self, objects, key_to_class, num_to_prefetch=10,
               locations=None):
    """Constructor.

    Args:
      objects: iterable of object IDs
      key_to_class: dict mapping keys to class labels
      num_to_prefetch: number of `objects` to prefetch
    """
    super().__init__(
      num_to_prefetch=num_to_prefetch, locations=locations,
      objects=objects)

    self.results = defaultdict(set)  # class -> ids

    self.viewer.actions.add('mr-next-batch', lambda s: self.next_batch())
    self.viewer.actions.add('mr-prev-batch', lambda s: self.prev_batch())
    self.viewer.actions.add('unclassify', lambda s: self.classify(None))

    for key, cls in key_to_class.items():
      self.viewer.actions.add(
        'classify-%s' % cls, lambda s, cls=cls: self.classify(cls))

    with self.viewer.config_state.txn() as s:
      for key, cls in key_to_class.items():
        s.input_event_bindings.viewer[
          'key%s' % key] = 'classify-%s' % cls

      # Navigation without classification.
      s.input_event_bindings.viewer['keyj'] = 'mr-next-batch'
      s.input_event_bindings.viewer['keyk'] = 'mr-prev-batch'
      s.input_event_bindings.viewer['keyv'] = 'unclassify'

    self.update_batch()

  def custom_msg(self):
    return ' '.join('%s:%d' % (k, len(v)) for k, v in self.results.items())

  def classify(self, cls):
    sid = list(self.todo[self.index]['seg'])[0]
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

  def __init__(self, graph, objects, bad, num_to_prefetch=0):
    super().__init__(objects=objects, num_to_prefetch=num_to_prefetch)
    self.graph = graph
    self.split_objects = []
    self.split_path = []
    self.split_index = 1
    self.sem = threading.Semaphore()

    self.bad = bad
    self.viewer.actions.add('add-ccs', lambda s: self.add_ccs())
    self.viewer.actions.add('clear-splits', lambda s: self.clear_splits())
    self.viewer.actions.add('add-split', self.add_split)
    self.viewer.actions.add('accept-split', lambda s: self.accept_split())
    self.viewer.actions.add('split-inc', lambda s: self.inc_split())
    self.viewer.actions.add('split-dec', lambda s: self.dec_split())
    self.viewer.actions.add('merge-segments',
                            lambda s: self.merge_segments())
    self.viewer.actions.add('mark-bad', lambda s: self.mark_bad())
    self.viewer.actions.add('next-batch', lambda s: self.next_batch())
    self.viewer.actions.add('prev-batch', lambda s: self.prev_batch())

    with self.viewer.config_state.txn() as s:
      s.input_event_bindings.viewer['keyj'] = 'next-batch'
      s.input_event_bindings.viewer['keyk'] = 'prev-batch'
      s.input_event_bindings.viewer['keyc'] = 'add-ccs'
      s.input_event_bindings.viewer['keya'] = 'clear-splits'
      s.input_event_bindings.viewer['keym'] = 'merge-segments'
      s.input_event_bindings.viewer['shift+bracketleft'] = 'split-dec'
      s.input_event_bindings.viewer['shift+bracketright'] = 'split-inc'
      s.input_event_bindings.viewer['keys'] = 'accept-split'
      s.input_event_bindings.data_view['shift+mousedown0'] = 'add-split'
      s.input_event_bindings.viewer['keyv'] = 'mark-bad'

    with self.viewer.txn() as s:
      s.layers['split'] = neuroglancer.SegmentationLayer(
        source=s.layers['seg'].source)
      s.layers['split'].visible = False

  def merge_segments(self):
    sids = [sid for sid in self.viewer.state.layers['seg'].segments if
            sid > 0]
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
    if self.sem.acquire(blocking=False):
      curr = set(self.viewer.state.layers['seg'].segments)
      for sid in self.viewer.state.layers['seg'].segments:
        if sid in self.graph:
          curr |= set(nx.node_connected_component(self.graph, sid))

      self.update_segments(curr)
      self.sem.release()

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
      self.split_objects.append(s.selected_values['seg'].value)
    self.update_msg(
      'split: %s' % (':'.join(str(x) for x in self.split_objects)))

    if len(self.split_objects) == 2:
      self.start_split()

  def mark_bad(self):
    if self.batch > 1:
      self.update_msg('decrease batch to 1 to mark objects bad')
      return

    sids = self.todo[self.index]['seg']
    if len(sids) == 1:
      self.bad.add(list(sids)[0])
    else:
      self.bad.add(frozenset(sids))

    self.update_msg('marked bad: %r' % (sids,))
    self.next_batch()
