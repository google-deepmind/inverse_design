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

"""Tools to compute the connectivity of the graph."""
import functools

import jax
from jax.experimental import host_callback as hcb
import jax.numpy as jnp
import numpy as np
from sklearn import neighbors


def _cb_radius_query(args):
  """Host callback function to compute connectivity."""
  padded_pos, n_node, radius, max_edges, query_mask, node_mask = args
  edges = []
  offset = 0

  for num_nodes in n_node:
    pos_nodes = padded_pos[offset:offset+num_nodes]
    pos_query = padded_pos[offset:offset+num_nodes]
    pos_nodes = pos_nodes[node_mask[offset:offset+num_nodes]]
    pos_query = pos_query[query_mask[offset:offset+num_nodes]]

    # indices: [num_edges, 2] array of receivers ([:, 0]) and senders ([:, 1])
    indices = compute_fixed_radius_connectivity_np(pos_nodes, radius, pos_query)
    mask = query_mask[offset:offset+num_nodes]
    renumber = np.arange(num_nodes, dtype=np.int32)[mask]
    indices[:, 0] = renumber[indices[:, 0]]

    mask = node_mask[offset:offset+num_nodes]
    renumber = np.arange(num_nodes, dtype=np.int32)[mask]
    indices[:, 1] = renumber[indices[:, 1]]

    # remove self-edges
    mask = indices[:, 0] != indices[:, 1]
    indices = indices[mask]

    # create unique two way edges (only necessary in the masked case)
    indices = np.stack([np.min(indices, axis=1),
                        np.max(indices, axis=1)],
                       axis=1)
    indices = np.unique(indices, axis=0)
    indices = np.concatenate([indices, indices[:, [1, 0]]], axis=0)

    edges.append(indices + offset)
    offset += num_nodes

  n_edge = [x.shape[0] for x in edges]
  total_edges = np.sum(n_edge)

  # padding
  if total_edges >= max_edges:
    raise ValueError("%d edges found, max_edges: %d" % (total_edges, max_edges))

  # create a [n_p, 2] padding array, which connects the first dummy padding node
  # (with index `num_nodes`) to itself.
  padding_size = max_edges - total_edges
  padding = np.ones((padding_size, 2), dtype=np.int32) * offset
  edges = np.concatenate(edges + [padding], axis=0)
  n_edge = np.array(n_edge + [padding_size], dtype=np.int32)
  return n_edge, edges


@functools.partial(jax.custom_jvp, nondiff_argnums=(4, 5))
def compute_fixed_radius_connectivity_jax(positions, n_node, query_mask,
                                          node_mask, radius, max_edges):
  """Computes connectivity for batched graphs using a jax host callback.

  Args:
    positions: concatenated vector (N, 2) of node positions for all graphs
    n_node: array of num_nodes for each graph
    query_mask: defines the subset of nodes to query from (None=all)
    node_mask: defines the subset of nodes to query to (None=all)
    radius: connectivity radius
    max_edges: maximum total number of edges

  Returns:
    array of num_edges, senders, receivers
  """
  callback_arg = (positions, n_node, radius, max_edges, query_mask, node_mask)
  out_shape = (jax.ShapeDtypeStruct((len(n_node) + 1,), jnp.int32),
               jax.ShapeDtypeStruct((max_edges, 2), jnp.int32))
  n_edge, indices = hcb.call(_cb_radius_query, callback_arg,
                             result_shape=out_shape)

  senders = indices[:, 1]
  receivers = indices[:, 0]
  return n_edge, senders, receivers


@compute_fixed_radius_connectivity_jax.defjvp
def _compute_fixed_radius_connectivity_jax_jvp(radius, max_edges, primals,
                                               tangents):
  """Custom zero-jvp function for compute_fixed_radius_connectivity_jax."""
  del tangents
  primal_out = compute_fixed_radius_connectivity_jax(
      *primals, radius=radius, max_edges=max_edges)
  grad_out = tuple(jnp.zeros_like(x) for x in primal_out)
  return primal_out, grad_out


def compute_fixed_radius_connectivity_np(
    positions, radius, receiver_positions=None, remove_self_edges=False):
  """Computes connectivity between positions and receiver_positions."""

  # if removing self edges, receiver positions must be none
  assert not (remove_self_edges and receiver_positions is not None)

  if receiver_positions is None:
    receiver_positions = positions

  # use kdtree for efficient calculation of pairs within radius distance
  kd_tree = neighbors.KDTree(positions)
  receivers_list = kd_tree.query_radius(receiver_positions, r=radius)
  num_nodes = len(receiver_positions)
  senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
  receivers = np.concatenate(receivers_list, axis=0)

  if remove_self_edges:
    # Remove self edges.
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]

  return np.stack([senders.astype(np.int32),
                   receivers.astype(np.int32)],
                  axis=-1)
