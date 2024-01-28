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
"""FFN model execution.

FFN inference is performed via a client-server 'executor' interface,
with the server owning the model and accelerator resources, and one
or more clients maintaining the segmentation state and orchestrating
inference calls.

This permits batching in the server, which is generally necessary for
good utilization of accelerator resources.
"""

import _thread
from concurrent import futures
import os
import queue
import threading
import time
from typing import Callable, Optional, Sequence

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v1 as tf
from ..training import model as ffn_model
from . import inference_utils
from .inference_utils import timer_counter


class TerminationException(Exception):  # pylint: disable=g-bad-exception-name
  """Indicates that the program has been requested to shut down."""
  pass


class ExecutorInterface:
  """Provides a client/server interface.

  Owns the communication channels and provides methods of communication
  with the server.
  """

  def __init__(self):
    self.lock = threading.Lock()
    self.outputs = {}  # Will be populated by Queues as clients register.
    # Used by clients to communiate with the executor. The protocol is
    # as follows:
    #  - 'exit': indicates a request to kill the executor
    #  - N >= 0: registration of a new client with the specified ID
    #  - N < 0: deregistration of an existing client with ID -N - 1
    #  - (client_id, seed, image, fetches): request to perform inference
    self._input_queue = queue.Queue()
    self.exit_request = threading.Event()

  def queue_put(self, x):
    if self.exit_request.is_set():
      raise TerminationException()

    return self._input_queue.put(x)

  def queue_get(self, **kwargs):
    return self._input_queue.get(**kwargs)

  def get_output(self,
                 client_id: int,
                 timeout: int = 0) -> dict[str, np.ndarray]:
    while True:
      try:
        return self.outputs[client_id].get(timeout=timeout)
      except queue.Empty:
        if self.exit_request.is_set():
          raise TerminationException()  # pylint:disable=raise-missing-from


class ExecutorClient:
  """Client interface for the FFN executor."""

  def __init__(self, counters: inference_utils.Counters,
               interface: ExecutorInterface):
    self._client_id = None
    self.counters = counters
    self._interface = interface

  def start(self) -> int:
    """Registers as a new client.

    Returns:
      client ID
    """
    raise NotImplementedError()

  def finish(self):
    """Deregisters the client."""
    raise NotImplementedError()

  def predict(self, seed: np.ndarray, image: np.ndarray,
              fetches: Sequence[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError()


class ThreadingExecutorClient(ExecutorClient):
  """Client interface for a same-process executor."""

  def start(self) -> int:
    with self._interface.lock:
      if not self._interface.outputs:
        client_id = 0
      else:
        client_id = max(self._interface.outputs.keys()) + 1

      self._interface.outputs[client_id] = queue.Queue()

    self._interface.queue_put(client_id)
    self._client_id = client_id
    return client_id

  def finish(self):
    if self._client_id is None:
      return
    with self._interface.lock:
      del self._interface.outputs[self._client_id]
    self._interface.queue_put(-1 - self._client_id)

  def predict(self, seed: np.ndarray, image: np.ndarray,
              fetches: Sequence[str]) -> dict[str, np.ndarray]:
    assert self._client_id is not None
    self._interface.queue_put((self._client_id, seed, image, fetches))
    with timer_counter(self.counters, 'client-wait'):
      return self._interface.get_output(self._client_id, timeout=1)


class BatchExecutor:
  """Base class for FFN executors.

  Owns the FFN model and any accelerator resources.
  """

  def __init__(self, interface: ExecutorInterface,
               model: ffn_model.FFNModel,
               model_info: ffn_model.ModelInfo, session: tf.Session,
               counters: inference_utils.Counters, batch_size: int):
    self._interface = interface
    self.session = session
    self.model = model
    self.counters = counters

    self.batch_size = batch_size
    self.active_clients = 0
    self.registered_clients = set()

    # Cache input/output sizes.
    self._input_seed_size = np.array(model_info.input_seed_size[::-1]).tolist()
    self._input_image_size = np.array(
        model_info.input_image_size[::-1]).tolist()
    self._pred_size = np.array(model_info.pred_mask_size[::-1]).tolist()

    self._initialize_model()

  def __del__(self):
    self.stop_server()

  def start_server(self):
    raise NotImplementedError()

  def stop_server(self):
    raise NotImplementedError()

  def get_client(self, subvol_counters):
    return ThreadingExecutorClient(subvol_counters, self._interface)

  def _initialize_model(self):
    self.model.define_tf_graph()

  def _run_executor(self):
    raise NotImplementedError()

  def _run_executor_log_exceptions(self):
    """Runs the main loop of the executor.

    Logs any exceptions and re-raises them.
    """
    try:
      self._run_executor()
    except Exception as e:  # pylint:disable=broad-except
      logging.exception(e)
      # If the executor fails, the whole process becomes useless and we need
      # to make sure it gets terminated.
      _thread.interrupt_main()  # pytype: disable=module-attr
      time.sleep(10)
      os._exit(1)  # pylint:disable=protected-access

  @property
  def num_devices(self):
    return 1


class ThreadingBatchExecutor(BatchExecutor):
  """Thread-based BatchExecutor for TF models.

  It is recommended to start the client threads as daemons, so that failures
  of the server thread will result in termination of the whole program.

  Note that the number of clients can (and for efficient utilization of ML
  accelerators, should) exceed the batch size. This makes sense to do even
  if the batch size is 1.
  """

  def __init__(self,
               interface: ExecutorInterface,
               model: Optional[ffn_model.FFNModel],
               model_info: ffn_model.ModelInfo,
               session: Optional[tf.Session],
               counters: inference_utils.Counters,
               batch_size: int,
               expected_clients: int = 1):
    super(ThreadingBatchExecutor, self).__init__(interface, model, model_info,
                                                 session, counters, batch_size)

    # Total clients seen during the lifetime of the executor.
    self.total_clients = 0

    # This many clients need to register themselves during the lifetime of
    # the executor in order for it be allowed to terminate.
    self.expected_clients = expected_clients

    # Arrays fed to TF.
    self.input_seed = np.zeros(
        [batch_size] + self._input_seed_size + [1], dtype=np.float32)
    self.input_image = np.zeros(
        [batch_size] + self._input_image_size + [1], dtype=np.float32)
    self.th_executor = None

  def start_server(self):
    """Starts the server which will evaluate TF models.

    The server will automatically terminate after no more clients are
    registered, and after at least one client has registered and
    deregistered.
    """
    if self.th_executor is None:
      self.th_executor = threading.Thread(
          target=self._run_executor_log_exceptions)
      self._interface.exit_request.clear()
      self.th_executor.start()

  def stop_server(self):
    if self.th_executor is None:
      return
    logging.info('Requesting executor shutdown.')
    self._interface.queue_put('exit')
    self._interface.exit_request.set()
    self.th_executor.join()
    self.th_executor = None
    logging.info('Executor shutdown complete.')

  def _run_executor(self):
    """Main loop of the server thread which runs TF code."""
    logging.info('Executor starting, batch_size=%d.', self.batch_size)

    while self.active_clients or self.total_clients < self.expected_clients:
      self.counters.get(
          'executor-clients', cumulative=False).Set(self.active_clients)

      with timer_counter(self.counters, 'executor-input'):
        ready = []
        while (len(ready) < min(self.active_clients, self.batch_size) or
               not self.active_clients):
          try:
            data = self._interface.queue_get(timeout=5)
          except queue.Empty:
            continue
          if data == 'exit':
            logging.info('Executor shut down requested.')
            return
          elif isinstance(data, int):
            client_id = data
            if client_id >= 0:
              self.registered_clients.add(client_id)
              self.total_clients += 1
              self.active_clients += 1
              logging.info('client %d starting', client_id)
            else:
              self.registered_clients.remove(-client_id - 1)
              logging.info('client %d terminating', -client_id - 1)
              self.active_clients -= 1
          else:
            client_id, seed, image, fetches = data
            l = len(ready)
            self.input_seed[l, ..., 0] = seed
            self.input_image[l, ..., 0] = image
            ready.append(client_id)

      if ready:
        self._schedule_batch(ready, fetches)

    logging.info('Executor terminating.')

  def _schedule_batch(self, client_ids: Sequence[int], fetches: Sequence[str]):
    """Schedules a single batch for execution."""

    to_fetch = {f: getattr(self.model, f) for f in fetches}
    with timer_counter(self.counters, 'executor-inference'):
      try:
        ret = self.session.run(
            to_fetch, {
                self.model.input_seed: self.input_seed,
                self.model.input_patches: self.input_image
            })
      except Exception as e:  # pylint:disable=broad-except
        logging.exception(e)
        # If calling TF didn't work (faulty hardware, misconfiguration, etc),
        # we want to terminate the whole program.
        _thread.interrupt_main()  # pytype: disable=module-attr
        raise e

    with timer_counter(self.counters, 'executor-output'):
      with self._interface.lock:
        for i, client_id in enumerate(client_ids):
          try:
            self._interface.outputs[client_id].put(
                {k: v[i, ...] for k, v in ret.items()})
          except KeyError:
            # This could happen if a client unregistered itself
            # while inference was running.
            pass


class TPUExecutor(ThreadingBatchExecutor):
  """ThreadingBatchExecutor for TF models on TPUs."""

  CLIENT_ID_PAD = -1
  CLIENT_ID_SHUTDOWN = -2

  @property
  def num_devices(self):
    return self.num_tpus

  def _tpu_loop(self, loop_op):
    try:
      self.session.run(loop_op)
    except Exception as e:
      logging.exception(e)
      raise e
    logging.info('TPU loop done.')

  def _dispatcher(self, results_op):
    """Reads data from TPU and dispatches it to clients."""
    try:
      while True:
        results = self.session.run(results_op)
        ids, logits = results
        if -2 in ids:
          logging.info('Terminating dispatcher.')
          return

        logits = np.reshape(logits, self.model.logits.shape_as_list())
        with self._interface.lock:
          for i, client_id in enumerate(ids):
            try:
              self._interface.outputs[client_id].put({'logits': logits[i, ...]})
            except KeyError:
              # This could happen if a client unregistered itself
              # while inference was running.
              pass
    except Exception as e:
      logging.exception(e)
      raise e

  def _run_executor(self):
    fs_loop, fs_dispatch = set(), set()
    self._curr_infeed = 0

    with futures.ThreadPoolExecutor(max_workers=self.num_tpus * 2) as e:
      # Runs the TPU main loop in separate threads.
      for loop_op in self.tpu_loop_ops:
        fs_loop.add(e.submit(self._tpu_loop, loop_op))
      # Gets results from the TPU and distributes them to the clients.
      for results_op in self.tpu_outfeed_results:
        fs_dispatch.add(e.submit(self._dispatcher, results_op))

      super(TPUExecutor, self)._run_executor()
      logging.info('TPU executor done. Shutting down.')

      # TODO(mjanusz): Fix this awkward shut down procedure.
      # Experiments show that the following alternatives do not work:
      #  - scheduling a single batch per TPU (no loops are actually terminated)
      #  - not scheduling more work on infeeds corresponding to loops that have
      #    terminated, assuming a 1:1 mapping between self.tpu_loop_ops and
      #    self.tpu_infeed_enquee_ops
      while fs_loop:
        for _ in range(self.num_tpus):
          # Terminate the main loop on the TPU.
          self._schedule_batch([TPUExecutor.CLIENT_ID_SHUTDOWN] *
                               self.batch_size, None)
          logging.info('Scheduling termination request.')

        fs_done, fs_loop = futures.wait(
            fs_loop, timeout=1, return_when=futures.FIRST_COMPLETED)
        for f in fs_done:
          f.result()

    # Check for exceptions.
    for f in fs_dispatch:
      f.result()
    self.session.run(self.tpu_shutdown_system)
    logging.info('TPU executor shutdown complete.')

  def _schedule_batch_on_feed(self, client_ids, fetches, feed_id):
    del fetches  # TODO(mjanusz): Support this.
    self.session.run(
        self.tpu_infeed_enqueue_ops[feed_id],
        {
            # Pad client IDs to full batch size.
            self.client_ids:
                np.array(
                    client_ids + [TPUExecutor.CLIENT_ID_PAD] *
                    (self.batch_size - len(client_ids)),
                    dtype=np.int32),
            self.plc_input_seed:
                self.input_seed,
            self.plc_input_patches:
                self.input_image
        })

  def _schedule_batch(self, client_ids, fetches):
    self._schedule_batch_on_feed(client_ids, fetches, self._curr_infeed)

    # Distribute batches across available TPUs in a round-robin fashion.
    self._curr_infeed += 1
    self._curr_infeed %= len(self.tpu_infeed_enqueue_ops)

  def _initialize_model(self):
    self.tpu_initialize_system = tf.tpu.initialize_system()
    self.tpu_shutdown_system = tf.tpu.shutdown_system()

    # This returns the global_tpu_id, which we don't use yet.
    self.session.run(self.tpu_initialize_system)
    self.client_ids = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,))

    # Define infeeds.
    infeed_placeholder_attrs = ('input_seed', 'input_patches')
    infeed_placeholders = [self.client_ids] + [
        getattr(self.model, attr) for attr in infeed_placeholder_attrs
    ]
    tpu_infeed_queue = tf.contrib.tpu.InfeedQueue(
        tuple_types=[t.dtype for t in infeed_placeholders],
        tuple_shapes=[t.shape for t in infeed_placeholders],
    )
    num_tpus = sum(
        dev.device_type == 'TPU' for dev in self.session.list_devices())
    tpu_infeed_queue.set_number_of_shards(num_tpus)
    logging.info('Found %d TPU cores.', num_tpus)

    self.num_tpus = num_tpus
    self.tpu_infeed_enqueue_ops = tpu_infeed_queue.generate_enqueue_ops(
        [infeed_placeholders] * num_tpus, lambda x: x)

    # Save the placeholders that we will feed on the host. These will
    # be replaced by infeed dequeue ops below.
    self.plc_input_seed = self.model.input_seed
    self.plc_input_patches = self.model.input_patches

    def loop_body(not_done):
      """Defines the graph that executes in a loop on the TPU."""
      del not_done
      inputs = tpu_infeed_queue.generate_dequeue_op()
      client_ids = inputs[0]
      inputs = inputs[1:]
      for attr, iteration_input in zip(infeed_placeholder_attrs, inputs):
        setattr(self.model, attr, iteration_input)

      # Define the graph for the FFN model.
      self.model.define_tf_graph()

      # Flat shape (or at least removal of the channel dimension) is necessary
      # for efficient outfeed as of Oct 2020).
      self.flat_logits = tf.reshape(self.model.logits, [-1])
      tpu_outfeed_enqueue_op = tf.contrib.tpu.outfeed_enqueue_tuple(
          [client_ids, self.flat_logits])

      with tf.control_dependencies([tpu_outfeed_enqueue_op]):
        return tf.greater(client_ids[0:1], TPUExecutor.CLIENT_ID_SHUTDOWN)

    def loop_condition(not_done):
      return tf.identity(not_done[0], name='not_done_reduce')

    # Note: this executes loop_body.
    self.tpu_loop_ops = tf.tpu.replicate(
        lambda: tf.while_loop(  # pylint:disable=g-long-lambda
            cond=loop_condition,
            body=loop_body,
            loop_vars=[tf.constant(True, dtype=tf.bool, shape=(1,))],
            parallel_iterations=1),
        inputs=[[]] * num_tpus)

    self.tpu_outfeed_results = []
    for i in range(num_tpus):
      with tf.device('/device:TPU:%d' % i):
        self.tpu_outfeed_results.append(
            tf.contrib.tpu.outfeed_dequeue_tuple(
                dtypes=[tf.int32, self.model.logits.dtype],
                shapes=[self.client_ids.shape, self.flat_logits.shape]))


class JAXExecutor(ThreadingBatchExecutor):
  """ThreadingBatchExecutor for JAX models."""

  def __init__(self,
               interface: ExecutorInterface,
               model_info: ffn_model.ModelInfo,
               apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
               counters: inference_utils.Counters,
               batch_size: int,
               expected_clients: int = 1):
    super().__init__(
        interface,
        model=None,
        model_info=model_info,
        session=None,
        counters=counters,
        batch_size=batch_size,
        expected_clients=expected_clients)
    self._apply_fn = jax.jit(apply_fn)
    self._curr_device = 0
    self.tpe = futures.ThreadPoolExecutor(max_workers=jax.device_count())

  @property
  def num_devices(self) -> int:
    return jax.device_count()

  def _initialize_model(self):
    # JAX models do not require initialization.
    return

  def _dispatcher(self, data: jnp.ndarray, client_ids: Sequence[int]):
    """Dispatches results to clients."""
    with timer_counter(self.counters, 'executor-inference'):
      try:
        ret = self._apply_fn(data)
      except Exception as e:  # pylint:disable=broad-except
        logging.exception(e)
        # Terminate the whole program on failure.
        _thread.interrupt_main()  # pytype: disable=module-attr
        raise e

      ret = np.array(ret)

    with timer_counter(self.counters, 'executor-output'):
      with self._interface.lock:
        for i, client_id in enumerate(client_ids):
          try:
            self._interface.outputs[client_id].put({'logits': ret[i, ...]})
          except KeyError:
            # This could happen if a client unregistered itself
            # while inference was running.
            pass

  def _schedule_batch(self, client_ids: Sequence[int], fetches: Sequence[str]):
    del fetches

    with jax.default_device(jax.devices()[self._curr_device]):
      data = jnp.concatenate([self.input_image, self.input_seed], axis=-1)

    self.tpe.submit(self._dispatcher, data, client_ids)
    self._curr_device = (self._curr_device + 1) % self.num_devices
