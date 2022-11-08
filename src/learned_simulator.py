# Copyright 2022 DeepMind Technologies Limited
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

"""Graph Network Simulator implementation used in NeurIPS 2022 submission.

  Inverse Design for Fluid-Structure Interactions using Graph Network Simulators

  Kelsey R. Allen*, Tatiana Lopez-Guevera*, Kimberly Stachenfeld*,
  Alvaro Sanchez-Gonzalez, Peter Battaglia, Jessica Hamrick, Tobias Pfaff
"""

from typing import Any, Dict

import haiku as hk
import jraph

from inverse_design.src import graph_network
from inverse_design.src import normalizers


class LearnedSimulator(hk.Module):
  """Graph Network Simulator."""

  def __init__(self,
               connectivity_radius,
               *,
               graph_network_kwargs: Dict[str, Any],
               flatten_features_fn=None,
               name="LearnedSimulator"):
    """Initialize the model.

    Args:
      connectivity_radius: Radius of connectivity within which to connect
        particles with edges.
      graph_network_kwargs: Keyword arguments to pass to the learned part of the
        graph network `model.EncodeProcessDecode`.
      flatten_features_fn: Function that takes the input graph and dataset
        metadata, and returns a graph where node and edge features are a single
        array of rank 2, and without global features. The function will be
        wrapped in a haiku module, which allows the flattening fn to instantiate
        its own variable normalizers.
      name: Name of the Haiku module.
    """
    super().__init__(name=name)
    self._connectivity_radius = connectivity_radius
    self._graph_network_kwargs = graph_network_kwargs
    self._graph_network = None

    # Wrap flatten function in a Haiku module, so any haiku modules created
    # by the function are reused in case of multiple calls.
    self._flatten_features_fn = hk.to_module(flatten_features_fn)(
        name="flatten_features_fn")

  def _maybe_build_modules(self, input_graph):
    if self._graph_network is None:
      num_dimensions = input_graph.nodes["world_position"].shape[-1]
      self._graph_network = graph_network.EncodeProcessDecode(
          name="encode_process_decode",
          node_output_size=num_dimensions,
          **self._graph_network_kwargs)

      self._target_normalizer = normalizers.get_accumulated_normalizer(
          name="target_normalizer")

  def __call__(self, input_graph: jraph.GraphsTuple, padded_graph=True):
    self._maybe_build_modules(input_graph)

    flat_graphs_tuple = self._encoder_preprocessor(
        input_graph, padded_graph=padded_graph)
    normalized_prediction = self._graph_network(flat_graphs_tuple).nodes
    next_position = self._decoder_postprocessor(normalized_prediction,
                                                input_graph)
    return input_graph._replace(
        nodes={"p:world_position": next_position},
        edges={},
        globals={},
        senders=input_graph.senders[:0],
        receivers=input_graph.receivers[:0],
        n_edge=input_graph.n_edge * 0), {}

  def _encoder_preprocessor(self, input_graph, padded_graph):
    # Flattens the input graph
    graph_with_flat_features = self._flatten_features_fn(
        input_graph,
        connectivity_radius=self._connectivity_radius,
        is_padded_graph=padded_graph)
    return graph_with_flat_features

  def _decoder_postprocessor(self, normalized_prediction, input_graph):
    # Un-normalize and integrate
    position_sequence = input_graph.nodes["world_position"]

    # The model produces the output in normalized space so we apply inverse
    # normalization.
    prediction = self._target_normalizer.inverse(normalized_prediction)

    new_position = euler_integrate_position(position_sequence, prediction)
    return new_position


def euler_integrate_position(position_sequence, finite_diff_estimate):
  """Integrates finite difference estimate to position (assuming dt=1)."""
  # Uses an Euler integrator to go from acceleration to position,
  # assuming dt=1 corresponding to the size of the finite difference.
  previous_position = position_sequence[:, -1]
  previous_velocity = previous_position - position_sequence[:, -2]
  next_acceleration = finite_diff_estimate
  next_velocity = previous_velocity + next_acceleration
  next_position = previous_position + next_velocity
  return next_position


def euler_integrate_position_inverse(position_sequence, next_position):
  """Computes a finite difference estimate from current position and history."""
  previous_position = position_sequence[:, -1]
  previous_velocity = previous_position - position_sequence[:, -2]
  next_velocity = next_position - previous_position
  acceleration = next_velocity - previous_velocity
  return acceleration
