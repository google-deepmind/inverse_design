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

"""JAX implementation of Encode Process Decode."""

from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
import jraph


class EncodeProcessDecode(hk.Module):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      *,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      num_processor_repetitions: int = 1,
      encode_nodes: bool = True,
      encode_edges: bool = True,
      node_output_size: Optional[int] = None,
      edge_output_size: Optional[int] = None,
      include_sent_messages_in_node_update: bool = False,
      use_layer_norm: bool = True,
      name: str = "EncodeProcessDecode"):
    """Inits the model.

    Args:
      latent_size: Size of the node and edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of unshared message passing steps
         in the processor steps.
      num_processor_repetitions: Number of times that the same processor is
         applied sequencially.
      encode_nodes: If False, the node encoder will be omitted.
      encode_edges: If False, the edge encoder will be omitted.
      node_output_size: Output size of the decoded node representations.
      edge_output_size: Output size of the decoded edge representations.
      include_sent_messages_in_node_update: Whether to include pooled sent
          messages from each node in the node update.
      use_layer_norm: Whether it uses layer norm or not.
      name: Name of the model.
    """

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._encode_nodes = encode_nodes
    self._encode_edges = encode_edges
    self._node_output_size = node_output_size
    self._edge_output_size = edge_output_size
    self._include_sent_messages_in_node_update = (
        include_sent_messages_in_node_update)
    self._use_layer_norm = use_layer_norm
    self._networks_builder()

  def __call__(self, input_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Forward pass of the learnable dynamics model."""

    # Encode the input_graph.
    latent_graph_0 = self._encode(input_graph)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0)

    # Decode from the last latent graph.
    return self._decode(latent_graph_m)

  def _networks_builder(self):

    def build_mlp(name, output_size=None):
      if output_size is None:
        output_size = self._latent_size
      mlp = hk.nets.MLP(
          output_sizes=[self._mlp_hidden_size] * self._mlp_num_hidden_layers + [
              output_size], name=name + "_mlp", activation=jax.nn.relu)
      return jraph.concatenated_args(mlp)

    def build_mlp_with_maybe_layer_norm(name, output_size=None):
      network = build_mlp(name, output_size)
      if self._use_layer_norm:
        layer_norm = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name=name + "_layer_norm")
        network = hk.Sequential([network, layer_norm])
      return jraph.concatenated_args(network)

    # The encoder graph network independently encodes edge and node features.
    encoder_kwargs = dict(
        embed_edge_fn=build_mlp_with_maybe_layer_norm("encoder_edges")
        if self._encode_edges else None,
        embed_node_fn=build_mlp_with_maybe_layer_norm("encoder_nodes")
        if self._encode_nodes else None,)
    self._encoder_network = jraph.GraphMapFeatures(**encoder_kwargs)

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    self._processor_networks = []
    for step_i in range(self._num_message_passing_steps):
      self._processor_networks.append(
          jraph.InteractionNetwork(
              update_edge_fn=build_mlp_with_maybe_layer_norm(
                  f"processor_edges_{step_i}"),
              update_node_fn=build_mlp_with_maybe_layer_norm(
                  f"processor_nodes_{step_i}"),
              include_sent_messages_in_node_update=(
                  self._include_sent_messages_in_node_update)))

    # The decoder MLP decodes edge/node latent features into the output sizes.
    decoder_kwargs = dict(
        embed_edge_fn=build_mlp("decoder_edges", self._edge_output_size)
        if self._edge_output_size else None,
        embed_node_fn=build_mlp("decoder_nodes", self._node_output_size)
        if self._node_output_size else None,
    )
    self._decoder_network = jraph.GraphMapFeatures(**decoder_kwargs)

  def _encode(
      self, input_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Encodes the input graph features into a latent graph."""

    # Copy the globals to all of the nodes, if applicable.
    if input_graph.globals is not None:
      broadcasted_globals = jnp.repeat(
          input_graph.globals, input_graph.n_node, axis=0,
          total_repeat_length=input_graph.nodes.shape[0])
      input_graph = input_graph._replace(
          nodes=jnp.concatenate(
              [input_graph.nodes, broadcasted_globals], axis=-1),
          globals=None)

    # Encode the node and edge features.
    latent_graph_0 = self._encoder_network(input_graph)
    return latent_graph_0

  def _process(
      self, latent_graph_0: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Processes the latent graph with several steps of message passing."""

    # Do `num_message_passing_steps` with each of the `self._processor_networks`
    # with unshared weights, and repeat that `self._num_processor_repetitions`
    # times.
    latent_graph = latent_graph_0
    for unused_repetition_i in range(self._num_processor_repetitions):
      for processor_network in self._processor_networks:
        latent_graph = self._process_step(processor_network, latent_graph,
                                          latent_graph_0)

    return latent_graph

  def _process_step(
      self, processor_network_k,
      latent_graph_prev_k: jraph.GraphsTuple,
      latent_graph_0: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Single step of message passing with node/edge residual connections."""

    input_graph_k = latent_graph_prev_k

    # One step of message passing.
    latent_graph_k = processor_network_k(input_graph_k)

    # Add residuals.
    latent_graph_k = latent_graph_k._replace(
        nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
        edges=latent_graph_k.edges+latent_graph_prev_k.edges)
    return latent_graph_k

  def _decode(self, latent_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Decodes from the latent graph."""
    return self._decoder_network(latent_graph)
