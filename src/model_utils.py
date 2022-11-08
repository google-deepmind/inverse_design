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

"""Utility functions for the LearnedSimulator model."""

import jax
import jax.numpy as jnp
import jraph
import tree

from inverse_design.src import normalizers


def flatten_features(input_graph,
                     connectivity_radius,
                     is_padded_graph,
                     apply_normalization=False):
  """Returns a graph with a single array of node and edge features."""

  # Normalize the eleements of the graph.
  if apply_normalization:
    graph_elements_normalizer = normalizers.GraphElementsNormalizer(
        template_graph=input_graph,
        is_padded_graph=is_padded_graph)

  # Computing relative distances in the model.
  if "relative_world_position" not in input_graph.edges:
    input_graph = _add_relative_distances(
        input_graph)

  # Extract important features from the position_sequence.
  position_sequence = input_graph.nodes["world_position"]
  velocity_sequence = time_diff(position_sequence)  # Finite-difference.

  # Collect node features.
  node_features = []

  # Normalized velocity sequence, flattening spatial axis.
  flat_velocity_sequence = jnp.reshape(velocity_sequence,
                                       [velocity_sequence.shape[0], -1])

  if apply_normalization:
    flat_velocity_sequence = graph_elements_normalizer.normalize_node_array(
        "velocity_sequence", flat_velocity_sequence)

  node_features.append(flat_velocity_sequence)

  # Material types (one-hot, does not need normalization).
  node_features.append(jax.nn.one_hot(input_graph.nodes["material_type(9)"], 9))

  # Collect edge features.
  edge_features = []

  # Relative distances and norms.
  relative_world_position = input_graph.edges["relative_world_position"]
  relative_world_distance = safe_edge_norm(
      input_graph.edges["relative_world_position"],
      input_graph,
      is_padded_graph,
      keepdims=True)

  if apply_normalization:
    # Scaled determined by connectivity radius.
    relative_world_position = relative_world_position / connectivity_radius
    relative_world_distance = relative_world_distance / connectivity_radius

  edge_features.append(relative_world_position)
  edge_features.append(relative_world_distance)

  # Handle normalization.
  node_features = jnp.concatenate(node_features, axis=-1)
  edge_features = jnp.concatenate(edge_features, axis=-1)

  return input_graph._replace(
      nodes=node_features,
      edges=edge_features,
      globals=None,
  )


def time_diff(input_sequence):
  """Compute finnite time difference."""
  return input_sequence[:, 1:] - input_sequence[:, :-1]


def safe_edge_norm(array, graph, is_padded_graph, keepdims=False):
  """Compute vector norm, preventing nans in padding elements."""
  if is_padded_graph:
    padding_mask = jraph.get_edge_padding_mask(graph)
    epsilon = 1e-8
    perturb = jnp.logical_not(padding_mask) * epsilon
    array += jnp.expand_dims(perturb, range(1, len(array.shape)))
  return jnp.linalg.norm(array, axis=-1, keepdims=keepdims)


def _add_relative_distances(input_graph,
                            use_last_position_only=True):
  """Computes relative distances between particles and with walls."""

  # If these exist, there is probably something wrong.
  assert "relative_world_position" not in input_graph.edges
  assert "clipped_distance_to_walls" not in input_graph.nodes

  input_graph = tree.map_structure(lambda x: x, input_graph)  # Avoid mutating.
  particle_pos = input_graph.nodes["world_position"]

  if use_last_position_only:
    particle_pos = particle_pos[:, -1]

  input_graph.edges["relative_world_position"] = (
      particle_pos[input_graph.receivers] - particle_pos[input_graph.senders])

  return input_graph
