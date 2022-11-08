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

"""Watercourse 3D environment utils."""
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import tree

from inverse_design.src import connectivity_utils


NORMAL = 0
OBSTACLE = 1
INFLOW = 4

# for eliminating stray particles from pipe
OOB_AREA = 1.5


def _update_edges(input_graph, obstacle_edges, radius):
  """Recomputes particle edges, adds obstacle edges."""
  # get input graph nodes corresponding to fluid
  query_mask = ~input_graph.nodes["external_mask"]

  # get input graph ndoes that are either fluid or obstacle
  valid_mask = query_mask | input_graph.nodes["obstacle_mask"]
  max_edges = input_graph.senders.shape[0]
  num_obstacle_edges = obstacle_edges.shape[0]

  # compute the sender and receiver edges for fluid-fluid and fluid-obstacle
  # interactions.
  n_edge, senders, receivers = connectivity_utils.compute_fixed_radius_connectivity_jax(
      input_graph.nodes["world_position"][:, -1],
      n_node=input_graph.n_node[:-1], max_edges=max_edges - num_obstacle_edges,
      radius=radius, query_mask=query_mask, node_mask=valid_mask)

  # update edges to include obstacle edges and new fluid-fluid edges
  return input_graph._replace(
      senders=jnp.concatenate([obstacle_edges[:, 0], senders], axis=0),
      receivers=jnp.concatenate([obstacle_edges[:, 1], receivers], axis=0),
      n_edge=n_edge.at[0].set(n_edge[0] + num_obstacle_edges))


def forward(input_graph, new_particles, network, haiku_model, obstacle_edges,
            radius):
  """Runs model and post-processing steps in jax, returns position sequence."""
  @hk.transform_with_state
  def model(inputs):
    return haiku_model()(inputs)
  rnd_key = jax.random.PRNGKey(42)  # use a fixed random key

  # only run for a single graph (plus one padding graph), update graph with
  # obstacle edges
  assert len(input_graph.n_node) == 2, "Not a single padded graph."
  graph = tree.map_structure(lambda x: x, input_graph)
  graph = _update_edges(graph, obstacle_edges, radius)

  # build material type
  pattern = jnp.ones_like(graph.nodes["external_mask"], dtype=jnp.int32)
  inflow_mask = jnp.any(~graph.nodes["mask_stack"], axis=-1)
  graph.nodes["material_type(9)"] = jnp.where(
      graph.nodes["external_mask"], pattern * OBSTACLE,
      jnp.where(inflow_mask, pattern * INFLOW,
                pattern * NORMAL))
  graph.nodes["type/particles"] = None

  # run model
  prev_pos = input_graph.nodes["world_position"]
  model_out = model.apply(network["params"], network["state"], rnd_key, graph)
  pred_pos = model_out[0][0].nodes["p:world_position"]
  total_nodes = jnp.sum(input_graph.n_node[:-1])
  node_padding_mask = jnp.arange(prev_pos.shape[0]) < total_nodes

  # update history, reset external particles
  next_pos_seq = jnp.concatenate([prev_pos[:, 1:], pred_pos[:, None]], axis=1)
  mask = (~input_graph.nodes["external_mask"]) & node_padding_mask
  next_pos_seq = jnp.where(mask[:, None, None], next_pos_seq, prev_pos)

  # add new particles, remove old particles that go below the floor surface
  delete_particles = next_pos_seq[:, -1, 1] <= 0
  delete_particles &= graph.nodes["mask_stack"][:, -1]
  particle_mask = graph.nodes["mask_stack"][:, -1] & ~delete_particles
  particle_mask |= new_particles
  mask_stack = jnp.concatenate(
      [graph.nodes["mask_stack"][:, 1:], particle_mask[:, None]], axis=1)

  # create new node features and update graph
  new_node_features = {
      **input_graph.nodes,
      "world_position": next_pos_seq,
      "mask_stack": mask_stack,
      "external_mask": ~particle_mask,
      "deleted": graph.nodes["deleted"] | delete_particles,
  }
  return input_graph._replace(nodes=new_node_features)


def build_initial_graph(input_graphs, max_edges):
  """Builds initial padded graphs tuple from typed graph."""
  obstacle_edges = np.stack(
      [input_graphs[0].senders, input_graphs[0].receivers], axis=1)
  graph = tree.map_structure(lambda x: x.copy(), input_graphs[0])

  # clear graph edges
  dummy_edge = np.zeros((0,), dtype=np.int32)
  graph = graph._replace(
      senders=dummy_edge,
      receivers=dummy_edge,
      n_edge=np.array([0], dtype=np.int32))

  # build inflow stack
  inflow_stack = []
  init_pos = graph.nodes["world_position"]
  for cur_graph in input_graphs:
    mask_stack = cur_graph.nodes["mask_stack"]
    cur_pos = cur_graph.nodes["world_position"]
    new_particles = mask_stack[:, -1] & (~mask_stack[:, -2])
    init_pos[new_particles] = cur_pos[new_particles]
    new_particles = np.concatenate([new_particles, [False]])
    inflow_stack.append(new_particles)
  inflow_stack = np.stack(inflow_stack[1:], axis=0)
  graph.nodes["world_position"] = init_pos
  graph.nodes["deleted"] = np.zeros(init_pos.shape[0], dtype=np.bool)

  # fix stray particles
  stray_particles = init_pos[:, -1, 1] > OOB_AREA
  graph.nodes["mask_stack"][stray_particles] = False
  graph.nodes["external_mask"][stray_particles] = True

  # pad to maximum node, edge values and add padding graph
  max_n_node = graph.n_node.sum() + 1
  graphs_tuple = jraph.pad_with_graphs(graph, n_node=max_n_node,
                                       n_edge=max_edges, n_graph=2)
  return obstacle_edges, inflow_stack, graphs_tuple


def rollout(initial_graph, inflow_stack, network, haiku_model, obstacle_edges,
            radius):
  """Runs a jittable model rollout."""
  @jax.checkpoint
  def _step(graph, inflow_mask):
    out_graph = forward(graph, inflow_mask, network, haiku_model,
                        obstacle_edges, radius)
    out_data = dict(
        pos=out_graph.nodes["world_position"][:, -1],
        mask=out_graph.nodes["mask_stack"][:, -1])
    return out_graph, out_data
  final_graph, trajectory = jax.lax.scan(_step, init=initial_graph,
                                         xs=inflow_stack)
  return final_graph, trajectory


def make_plain_obstacles(num_side=25):
  """Create a mesh obstacle (landscape) with num_side squared control points."""
  px, pz = np.meshgrid(
      np.linspace(-0.5, 0.5, num_side), np.linspace(-0.5, 0.5, num_side))
  trans = np.array([0.5, 0.5, 0.5])

  # generate height map
  py = np.zeros_like(px)
  pos = np.stack([px, py, pz], axis=-1).reshape((-1, 3))
  pos += trans[None]
  return pos


def max_x_loss_fn(graph):
  """Example loss function for maximizing x position of particles when they hit the ground."""
  z_pos = graph.nodes["world_position"][:, -1, 2]
  z_var = jnp.std(z_pos, where=graph.nodes["deleted"])
  x_pos = graph.nodes["world_position"][:, -1, 0]
  x_max = jnp.mean(-x_pos, where=graph.nodes["deleted"])
  return x_max + z_var


def smooth_loss_fn(obs_pos, num_side=25):
  """Smoothing loss function for minimizing sharp changes across obstacle."""
  obs_grid = jnp.reshape(obs_pos, (num_side, num_side))
  obs_dx = jnp.diff(obs_grid, axis=0) ** 2
  obs_dy = jnp.diff(obs_grid, axis=1) ** 2
  return 0.5 * (jnp.mean(obs_dx) + jnp.mean(obs_dy))


def design_fn(params, graph, height_scale=0.15):
  """Convert parameters in params into landscape heightfield to be represented in graph."""
  graph = tree.map_structure(lambda x: x, graph)
  init_pos = jnp.array(graph.nodes["world_position"])
  # use tanh transformation to limit height to be within [-1, 1]
  raw_obs_pos = jnp.tanh(params) * height_scale

  # tile graph to have the same time history as the fluid particles
  obs_pos = jnp.tile(raw_obs_pos[:, None], [1, init_pos.shape[1]])

  # only controlling the height, so set other dimensions to 0
  obs_pos = jnp.stack(
      [jnp.zeros_like(obs_pos), obs_pos,
       jnp.zeros_like(obs_pos)], axis=-1)

  # add controlled height to initial heightfield and update graph nodes
  pos = jnp.concatenate(
      [init_pos[:obs_pos.shape[0]] + obs_pos, init_pos[obs_pos.shape[0]:]],
      axis=0)
  graph.nodes["world_position"] = pos
  return graph, raw_obs_pos
