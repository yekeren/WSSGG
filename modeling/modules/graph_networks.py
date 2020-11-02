# Copyright 2020 Keren Ye, University of Pittsburgh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Building graph networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging

import tensorflow as tf
import tf_slim as slim

import sonnet as snt
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import _base
from graph_nets import utils_tf

from protos import graph_network_pb2


class GraphNet(abc.ABC):
  """Graph interface."""

  def __init__(self, options, is_training=False):
    """Initializes the graph network.

    Args:
      options: proto to store the configs.
      is_training: if True, build the training graph.
    """
    self.options = options
    self.is_training = is_training
    self.add_bi_directional_edges = None
    self.add_self_loop_edges = None
    self.use_reverse_edges = None

  @staticmethod
  def _pack_graphs_tuple(batch_n_node,
                         batch_n_edge,
                         batch_nodes,
                         batch_edges,
                         batch_senders,
                         batch_receivers,
                         use_reverse_edges=False,
                         add_bi_directional_edges=False,
                         add_self_loop_edges=False):
    """Packs data into a GraphTuple instance.

    Args:
      batch_n_node: A [batch] int tensor.
      batch_n_edge: A [batch] int tensor.
      batch_nodes: A [batch, max_n_node, dims] float tensor.
      batch_edges: A [batch, max_n_edge, dims] float tensor.
      batch_senders: A [batch, max_n_edge] int tensor.
      batch_receivers: A [batch, max_n_edge] int tensor.
      add_bi_directional_edges: If true, add bi-directional edges.
      add_self_loop_edges: If true, add self-loop edges.
      
    Returns:
      graphs_tuple: A GraphTuple instance.
    """
    graph_dicts = []
    for (n_node, n_edge, nodes, edges,
         senders, receivers) in zip(tf.unstack(batch_n_node),
                                    tf.unstack(batch_n_edge),
                                    tf.unstack(batch_nodes),
                                    tf.unstack(batch_edges),
                                    tf.unstack(batch_senders),
                                    tf.unstack(batch_receivers)):
      nodes = nodes[:n_node]
      edges = edges[:n_edge]
      senders = senders[:n_edge]
      receivers = receivers[:n_edge]

      if use_reverse_edges:  # TODO: test
        senders, receivers = receivers, senders
        assert not add_bi_directional_edges

      if add_bi_directional_edges:
        (senders, receivers) = (tf.concat([senders, receivers], axis=0),
                                tf.concat([receivers, senders], axis=0))
        edges = tf.concat([edges, edges], 0)
        n_edge = 2 * n_edge

      if add_self_loop_edges:
        senders = tf.concat([senders, tf.range(n_node)], 0)
        receivers = tf.concat([receivers, tf.range(n_node)], 0)
        edge_padding_shape = tf.concat(
            [tf.expand_dims(n_node, 0), edges.shape[1:]], 0)
        edges = tf.concat(
            [edges, tf.zeros(edge_padding_shape, dtype=edges.dtype)], 0)
        n_edge += n_node

      graph_dicts.append({
          graphs.N_NODE: n_node,
          graphs.N_EDGE: n_edge,
          graphs.NODES: nodes,
          graphs.EDGES: edges,
          graphs.SENDERS: senders,
          graphs.RECEIVERS: receivers,
      })
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)
    return graphs_tuple

  @staticmethod
  def _unpack_graphs_tuple(graphs_tuple, max_n_node, max_n_edge):
    """Unpacks data from a GraphTuple instance.

    Args:
      graphs_tuple: A GraphTuple instance.
      max_n_node: Maximum number of nodes.
      max_n_edge: Maximum number of edges.

    Returns:
      batch_nodes: A [batch, max_n_node, dims] float tensor.
      batch_edges: A [batch, max_n_edge, dims] float tensor.
    """
    batch_nodes = []
    batch_edges = []

    batch_size = graphs_tuple.n_node.shape[0]
    for i in range(batch_size):
      graph_tuple_at_i = utils_tf.get_graph(graphs_tuple, i)
      n_node = tf.squeeze(graph_tuple_at_i.n_node)
      n_edge = tf.squeeze(graph_tuple_at_i.n_edge)
      nodes = graph_tuple_at_i.nodes
      edges = graph_tuple_at_i.edges

      if not None in [nodes, max_n_node]:
        batch_nodes.append(
            tf.pad(tensor=nodes, paddings=[[0, max_n_node - n_node], [0, 0]]))
      if not None in [edges, max_n_edge]:
        batch_edges.append(
            tf.pad(tensor=edges, paddings=[[0, max_n_edge - n_edge], [0, 0]]))

    batch_nodes = tf.stack(batch_nodes, 0) if batch_nodes else None
    batch_edges = tf.stack(batch_edges, 0) if batch_edges else None

    return batch_nodes, batch_edges

  def compute_graph_embeddings(self,
                               batch_n_node,
                               batch_n_edge,
                               batch_nodes,
                               batch_edges,
                               batch_senders,
                               batch_receivers,
                               regularizer=None):
    """Computes graph embeddings for the graph nodes and edges.

    Args:
      batch_n_node: A [batch] int tensor.
      batch_n_edge: A [batch] int tensor.
      batch_nodes: A [batch, max_n_node, dims] float tensor.
      batch_edges: A [batch, max_n_edge, dims] float tensor.
      batch_senders: A [batch, max_n_edge] int tensor.
      batch_receivers: A [batch, max_n_edge] int tensor.
      regularizer: Regularizer to be used in linear layers.
      
    Returns:
      updated_node_embs: A [batch, max_n_node, dims] float tensor.
      updated_edge_embs: A [batch, max_n_edge, dims] float tensor.
    """
    max_n_node = tf.shape(batch_nodes)[1]
    max_n_edge = tf.shape(batch_edges)[1]

    graphs_tuple = self._pack_graphs_tuple(
        batch_n_node,
        batch_n_edge,
        batch_nodes,
        batch_edges,
        batch_senders,
        batch_receivers,
        use_reverse_edges=self.use_reverse_edges,
        add_bi_directional_edges=self.add_bi_directional_edges,
        add_self_loop_edges=self.add_self_loop_edges)

    output_graphs_tuple = self._build_graph(graphs_tuple, regularizer)

    updated_node_embs, updated_edge_embs = self._unpack_graphs_tuple(
        output_graphs_tuple, max_n_node, max_n_edge)
    return updated_node_embs, updated_edge_embs

  @abc.abstractmethod
  def _build_graph(self, graphs_tuple, regularizer):
    """Builds graph network.

    Args:
      graphs_tuple: A GraphTuple instance.
      regularizer: Regularizer to be used in linear layers.

    Returns:
      output_graphs_tuple: A updated GraphTuple instance.
    """
    pass


class NoGraph(GraphNet):
  """Graph interface."""

  def __init__(self, options, is_training=False):
    """Initializes the graph network.

    Args:
      options: proto to store the configs.
      is_training: if True, build the training graph.
    """
    super(NoGraph, self).__init__(options, is_training)

    if not isinstance(options, graph_network_pb2.NoGraph):
      raise ValueError('Options has to be an NoGraph proto.')

    self.add_bi_directional_edges = False
    self.add_self_loop_edges = False

  def _build_graph(self, graphs_tuple, regularizer):
    """Builds graph network.

    Args:
      graphs_tuple: A GraphTuple instance.
      regularizer: Regularizer to be used in linear layers.

    Returns:
      output_graphs_tuple: A updated GraphTuple instance.
    """
    return graphs_tuple


class SelfAttention(GraphNet):
  """Self attention model using a RNN cell."""

  def __init__(self, options, is_training=False):
    """Initializes the graph network.

    Args:
      options: proto to store the configs.
      is_training: if True, build the training graph.
    """
    super(SelfAttention, self).__init__(options, is_training)

    if not isinstance(options, graph_network_pb2.SelfAttention):
      raise ValueError('Options has to be an SelfAttention proto.')

    self.add_bi_directional_edges = options.add_bi_directional_edges
    self.add_self_loop_edges = options.add_self_loop_edges

  def _build_graph(self, graphs_tuple, regularizer):
    """Builds graph network.

    Args:
      graphs_tuple: A GraphTuple instance.
      regularizer: Regularizer to be used in linear layers.

    Returns:
      output_graphs_tuple: A updated GraphTuple instance.
    """
    node_values = graphs_tuple.nodes

    # Check configuations.
    num_heads = self.options.n_head
    key_dims = self.options.key_dims
    value_dims = node_values.shape[-1].value
    assert key_dims % num_heads == 0
    assert value_dims % num_heads == 0

    key_size = key_dims // num_heads
    value_size = value_dims // num_heads

    # Compute the key/query tensors shared across layers.
    if self.options.n_layer:
      with tf.variable_scope('self_attention'):
        node_queries = slim.fully_connected(node_values,
                                            num_outputs=key_dims,
                                            activation_fn=None,
                                            biases_initializer=None,
                                            scope='node_queries')
        node_keys = slim.fully_connected(node_values,
                                         num_outputs=key_dims,
                                         activation_fn=None,
                                         biases_initializer=None,
                                         scope='node_keys')

    # Initial RNN states.
    rnn_nodes = snt.GRU(hidden_size=value_dims)
    node_states = rnn_nodes.initial_state(
        batch_size=tf.shape(graphs_tuple.nodes)[0])

    # Stack layers.
    self_attention = modules.SelfAttention()
    graphs_tuple = graphs_tuple.replace(nodes=None, edges=None)

    for _ in range(self.options.n_layer):
      _, node_states = rnn_nodes(node_values, node_states)

      # Call graph_nets SelfAttention model.
      graphs_tuple = self_attention(
          tf.reshape(node_values, [-1, num_heads, value_size]),
          tf.reshape(node_keys, [-1, num_heads, key_size]),
          tf.reshape(node_queries, [-1, num_heads, key_size]), graphs_tuple)

      node_values = tf.reshape(graphs_tuple.nodes, [-1, value_dims])
      graphs_tuple = graphs_tuple.replace(nodes=None, edges=None)

    # Pack results, FC layer to project states.
    _, node_states = rnn_nodes(node_values, node_states)

    node_states = slim.fully_connected(node_states,
                                       num_outputs=value_dims,
                                       activation_fn=None,
                                       scope='node_states')

    graphs_tuple = graphs_tuple.replace(nodes=node_states, edges=None)
    return graphs_tuple


class GNetMPNN(_base.AbstractModule):
  """Inherits the GraphNets abstract module."""

  def __init__(self, name='GNetMPNN'):
    super(GNetMPNN, self).__init__(name=name)
    self._normalizer = modules._unsorted_segment_softmax

  def _build(self,
             input_graph,
             hidden_size=50,
             dropout_rate=0.5,
             attn_scale=1.0,
             regularizer=None,
             is_training=False):

    node_values = input_graph.nodes
    edge_values = input_graph.edges

    value_dims = node_values.shape[-1].value
    assert value_dims == edge_values.shape[-1].value

    # Compute edge values, sender feature + edge feature.
    # - edge_values = [total_num_edges, value_dims]
    edge_value_block = blocks.EdgeBlock(edge_model_fn=lambda: snt.Linear(
        output_size=value_dims, regularizers={'w': regularizer}),
                                        use_edges=True,
                                        use_receiver_nodes=True,
                                        use_sender_nodes=True,
                                        use_globals=False,
                                        name='update_edge_values')
    edge_values = edge_value_block(input_graph).edges
    tf.summary.histogram('mpnn/edge_values', edge_values)

    logits_block = blocks.EdgeBlock(
        edge_model_fn=lambda: snt.nets.MLP(output_sizes=[hidden_size, 1],
                                           activation=tf.nn.tanh,
                                           regularizers={'w': regularizer}),
        use_edges=True,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=False,
        name='update_attention_logits')
    attention_weights_logits = attn_scale * logits_block(input_graph).edges
    tf.summary.histogram('mpnn/logits', attention_weights_logits)

    normalized_attention_weight = modules._received_edges_normalizer(
        input_graph.replace(edges=attention_weights_logits),
        normalizer=self._normalizer)

    # Attending to sender values according to the weights.
    # - attended_edges = [total_num_edges, value_dims]
    attended_edges = edge_values * normalized_attention_weight

    # Summing all of the attended values from each node.
    # aggregated_attended_values = [total_num_nodes, embedding_size]
    received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
        reducer=tf.math.unsorted_segment_sum)
    aggregated_attended_values = received_edges_aggregator(
        input_graph.replace(edges=attended_edges))

    return input_graph.replace(nodes=aggregated_attended_values,
                               edges=edge_values)


class MessagePassing(GraphNet):
  """Self attention model using a RNN cell."""

  def __init__(self, options, is_training=False):
    """Initializes the graph network.

    Args:
      options: proto to store the configs.
      is_training: if True, build the training graph.
    """
    super(MessagePassing, self).__init__(options, is_training)

    if not isinstance(options, graph_network_pb2.MessagePassing):
      raise ValueError('Options has to be an MessagePassing proto.')

    self.use_reverse_edges = options.use_reverse_edges
    self.add_bi_directional_edges = options.add_bi_directional_edges
    self.add_self_loop_edges = False

  def _build_graph(self, graphs_tuple, regularizer):
    """Builds graph network.

    Args:
      graphs_tuple: A GraphTuple instance.
      regularizer: Regularizer to be used in linear layers.

    Returns:
      output_graphs_tuple: A updated GraphTuple instance.
    """
    # Initial RNN states.
    rnn_nodes = snt.GRU(hidden_size=graphs_tuple.nodes.shape[-1].value,
                        name='node_rnn')
    rnn_edges = snt.GRU(hidden_size=graphs_tuple.edges.shape[-1].value,
                        name='edge_rnn')
    node_states = rnn_nodes.initial_state(
        batch_size=tf.shape(graphs_tuple.nodes)[0])
    edge_states = rnn_edges.initial_state(
        batch_size=tf.shape(graphs_tuple.edges)[0])

    _, node_states = rnn_nodes(graphs_tuple.nodes, node_states)
    _, edge_states = rnn_edges(graphs_tuple.edges, edge_states)
    graphs_tuple = graphs_tuple.replace(nodes=node_states, edges=edge_states)

    # Stack layers.
    network = GNetMPNN()
    for _ in range(self.options.n_layer):
      graphs_tuple = network(graphs_tuple,
                             hidden_size=self.options.hidden_size,
                             regularizer=regularizer,
                             attn_scale=self.options.attn_scale,
                             is_training=self.is_training)
      _, node_states = rnn_nodes(graphs_tuple.nodes, node_states)
      _, edge_states = rnn_edges(graphs_tuple.edges, edge_states)
      graphs_tuple = graphs_tuple.replace(nodes=node_states, edges=edge_states)

    # Projectthe RNN states.
    node_states = slim.fully_connected(
        graphs_tuple.nodes,
        num_outputs=graphs_tuple.nodes.shape[-1].value,
        activation_fn=None,
        scope='fc_nodes')
    edge_states = slim.fully_connected(
        graphs_tuple.edges,
        num_outputs=graphs_tuple.edges.shape[-1].value,
        activation_fn=None,
        scope='fc_edges')
    graphs_tuple = graphs_tuple.replace(nodes=node_states, edges=edge_states)
    return graphs_tuple


_MODELS = {
    'no_graph': NoGraph,
    'self_attention': SelfAttention,
    'message_passing': MessagePassing,
}


def build_graph_network(config, is_training=False):
  """Builds graph network according to the config.

  Args:
    config: An instance of graph_network_pb2.GraphNetwork.
    is_training: if True, build the training graph.

  Returns:
    An instance of graph network.
  """
  if not isinstance(config, graph_network_pb2.GraphNetwork):
    raise ValueError('Config has to be an instance of GraphNetwork proto.')

  network_oneof = config.WhichOneof('graph_network_oneof')
  if not network_oneof in _MODELS:
    raise ValueError('Invalid model %s!' % network_oneof)

  return _MODELS[network_oneof](getattr(config, network_oneof),
                                is_training=is_training)
