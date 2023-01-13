# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX module for normalization with accumulated statistics."""

import haiku as hk
import jax.numpy as jnp
import jraph


def get_accumulated_normalizer(name):
  return AccumulatedNormalizer(name=name)


class AccumulatedNormalizer(hk.Module):
  """Feature normalizer that accumulates statistics for normalization.

  It will accumulate statistics using float32 variables, and will return
  the mean and std. It accumulates statistics until the accumulate method is
  called `max_num_accumulations` times or the total number of batch elements
  processed is below `max_example_count`.

  To enable full GPU compatibility the number of accumulations is stored as a
  float32. As this number is incremented one by one, we require
  `max_num_accumulations` to be smaller than the highest float32 number that
  maintains integer precision (16777216).

  """

  def __init__(
      self,
      *,
      std_epsilon: float = 1e-5,
      name: str = 'accumulated_normalizer',
  ):
    """Inits the module.

    Args:
      std_epsilon: minimum value of the standard deviation to use.
      name: Name of the module.
    """
    super(AccumulatedNormalizer, self).__init__(name=name)
    self._accumulator_shape = None
    self._std_epsilon = std_epsilon

  def __call__(self, batched_data):
    """Direct transformation of the normalizer."""
    self._set_accumulator_shape(batched_data)
    return (batched_data - self.mean) / self.std_with_epsilon

  def inverse(self, normalized_batch_data):
    """Inverse transformation of the normalizer."""
    self._set_accumulator_shape(normalized_batch_data)
    return normalized_batch_data * self.std_with_epsilon + self.mean

  def _set_accumulator_shape(self, batched_sample_data):
    self._accumulator_shape = batched_sample_data.shape[-1]

  def _verify_module_connected(self):
    if self._accumulator_shape is None:
      raise RuntimeError(
          'Trying to read the mean before connecting the module.')

  @property
  def _acc_sum(self):
    return hk.get_state(
        'acc_sum', self._accumulator_shape, dtype=jnp.float32, init=jnp.zeros)

  @property
  def _acc_count(self):
    return hk.get_state('acc_count', (), dtype=jnp.float32, init=jnp.zeros)

  @property
  def _acc_sum_squared(self):
    return hk.get_state(
        'acc_sum_squared',
        self._accumulator_shape,
        dtype=jnp.float32,
        init=jnp.zeros)

  @property
  def _safe_count(self):
    # To ensure count is at least one and avoid nan's.
    return jnp.maximum(self._acc_count, 1.)

  @property
  def mean(self):
    self._verify_module_connected()
    return self._acc_sum / self._safe_count

  @property
  def std(self):
    self._verify_module_connected()
    var = self._acc_sum_squared / self._safe_count - self.mean**2
    var = jnp.maximum(var, 0.)  # Prevent negatives due to numerical precision.
    return jnp.sqrt(var)

  @property
  def std_with_epsilon(self):
    # To use in case the std is too small.
    return jnp.maximum(self.std, self._std_epsilon)


class GraphElementsNormalizer(hk.Module):
  """Online normalization of individual graph components of a GraphsTuple.


  Can be used to normalize individual node, edge, and global arrays.

  """

  def __init__(self,
               template_graph: jraph.GraphsTuple,
               is_padded_graph: bool,
               name: str = 'graph_elements_normalizer'):
    """Inits the module.

    Args:
      template_graph: Input template graph to compute edge/node/global padding
        masks.
      is_padded_graph: Whether the graph has padding.
      name: Name of the Haiku module.
    """

    super().__init__(name=name)
    self._node_mask = None
    self._edge_mask = None
    self._graph_mask = None
    if is_padded_graph:
      self._node_mask = jraph.get_node_padding_mask(template_graph)
      self._edge_mask = jraph.get_edge_padding_mask(template_graph)
      self._graph_mask = jraph.get_graph_padding_mask(template_graph)

    self._names_used = []

  def _run_normalizer(self, name, array, mask):
    if name in self._names_used:
      raise ValueError(
          f'Attempt to reuse name {name}. Used names: {self._names_used}')
    self._names_used.append(name)

    normalizer = get_accumulated_normalizer(name)
    return normalizer(array)

  def normalize_node_array(self, name, array):
    return self._run_normalizer(name, array, self._node_mask)
