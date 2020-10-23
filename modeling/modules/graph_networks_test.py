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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format
from protos import graph_network_pb2

from modeling.modules import graph_networks

from graph_nets import graphs
from graph_nets import utils_tf

tf.compat.v1.enable_eager_execution()


class GraphNetsBuilderTest(tf.test.TestCase):

  def test_pack_graphs_tuple(self):
    graph = graph_networks.GraphNet._pack_graphs_tuple(
        batch_n_node=[4],
        batch_n_edge=[3],
        batch_nodes=[[0, 1, 2, 3]],
        batch_edges=[[1, 2, 3]],
        batch_senders=[[0, 0, 0]],
        batch_receivers=[[1, 2, 3]],
        add_bi_directional_edges=False,
        add_self_loop_edges=False)

    self.assertAllEqual(graph.n_node, [4])
    self.assertAllEqual(graph.n_edge, [3])
    self.assertAllEqual(graph.nodes, [0, 1, 2, 3])
    self.assertAllEqual(graph.edges, [1, 2, 3])
    self.assertAllEqual(graph.senders, [0, 0, 0])
    self.assertAllEqual(graph.receivers, [1, 2, 3])

  def test_pack_graphs_tuple_bi_directional(self):
    graph = graph_networks.GraphNet._pack_graphs_tuple(
        batch_n_node=[4],
        batch_n_edge=[3],
        batch_nodes=[[0, 1, 2, 3]],
        batch_edges=[[1, 2, 3]],
        batch_senders=[[0, 0, 0]],
        batch_receivers=[[1, 2, 3]],
        add_bi_directional_edges=True,
        add_self_loop_edges=False)

    self.assertAllEqual(graph.n_node, [4])
    self.assertAllEqual(graph.n_edge, [6])
    self.assertAllEqual(graph.nodes, [0, 1, 2, 3])
    self.assertAllEqual(graph.edges, [1, 2, 3, 1, 2, 3])
    self.assertAllEqual(graph.senders, [0, 0, 0, 1, 2, 3])
    self.assertAllEqual(graph.receivers, [1, 2, 3, 0, 0, 0])

  def test_pack_graphs_tuple_self_loop(self):
    graph = graph_networks.GraphNet._pack_graphs_tuple(
        batch_n_node=[4],
        batch_n_edge=[3],
        batch_nodes=[[0, 1, 2, 3]],
        batch_edges=[[1, 2, 3]],
        batch_senders=[[0, 0, 0]],
        batch_receivers=[[1, 2, 3]],
        add_bi_directional_edges=False,
        add_self_loop_edges=True)

    self.assertAllEqual(graph.n_node, [4])
    self.assertAllEqual(graph.n_edge, [7])
    self.assertAllEqual(graph.nodes, [0, 1, 2, 3])
    self.assertAllEqual(graph.edges, [1, 2, 3, 0, 0, 0, 0])
    self.assertAllEqual(graph.senders, [0, 0, 0, 0, 1, 2, 3])
    self.assertAllEqual(graph.receivers, [1, 2, 3, 0, 1, 2, 3])

  def test_pack_graphs_tuple_self_loop_multidimensions(self):
    graph = graph_networks.GraphNet._pack_graphs_tuple(
        batch_n_node=[4],
        batch_n_edge=[3],
        batch_nodes=[[[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]],
        batch_edges=[[[1.0, 1.2], [2.0, 2.2], [3.0, 3.2]]],
        batch_senders=[[0, 0, 0]],
        batch_receivers=[[1, 2, 3]],
        add_bi_directional_edges=False,
        add_self_loop_edges=True)

    self.assertAllEqual(graph.n_node, [4])
    self.assertAllEqual(graph.n_edge, [7])
    self.assertAllClose(graph.nodes,
                        [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    self.assertAllClose(graph.edges,
                        [[1.0, 1.2], [2.0, 2.2], [3.0, 3.2], [0.0, 0.0],
                         [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    self.assertAllEqual(graph.senders, [0, 0, 0, 0, 1, 2, 3])
    self.assertAllEqual(graph.receivers, [1, 2, 3, 0, 1, 2, 3])

  def test_unpack_graphs_tuple(self):
    graph_0 = {
        graphs.N_NODE: 4,
        graphs.N_EDGE: 3,
        graphs.NODES: [[0, 0.1], [1, 1.1], [2, 2.1], [3, 3.1]],
        graphs.EDGES: [[1, 1.2], [2, 2.2], [3, 3.2]],
        graphs.SENDERS: [0, 0, 0],
        graphs.RECEIVERS: [1, 2, 3],
    }
    graph_1 = {
        graphs.N_NODE: 3,
        graphs.N_EDGE: 2,
        graphs.NODES: [[0, 0.1], [1, 1.1], [2, 2.1]],
        graphs.EDGES: [[1, 1.2], [2, 2.2]],
        graphs.SENDERS: [0, 1],
        graphs.RECEIVERS: [1, 2],
    }
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([graph_0, graph_1])
    nodes, edges = graph_networks.GraphNet._unpack_graphs_tuple(
        graphs_tuple=graphs_tuple, max_n_node=5, max_n_edge=6)
    self.assertAllClose(nodes,
                        [[[0, 0.1], [1, 1.1], [2, 2.1], [3, 3.1], [0, 0]],
                         [[0, 0.1], [1, 1.1], [2, 2.1], [0, 0], [0, 0]]])
    self.assertAllClose(edges,
                        [[[1, 1.2], [2, 2.2], [3, 3.2], [0, 0], [0, 0], [0, 0]],
                         [[1, 1.2], [2, 2.2], [0, 0], [0, 0], [0, 0], [0, 0]]])

  def test_unpack_graphs_tuple_no_edge(self):
    graph_0 = {
        graphs.N_NODE: 4,
        graphs.N_EDGE: 3,
        graphs.NODES: [[0, 0.1], [1, 1.1], [2, 2.1], [3, 3.1]],
        graphs.EDGES: [[1, 1.2], [2, 2.2], [3, 3.2]],
        graphs.SENDERS: [0, 0, 0],
        graphs.RECEIVERS: [1, 2, 3],
    }
    graph_1 = {
        graphs.N_NODE: 3,
        graphs.N_EDGE: 2,
        graphs.NODES: [[0, 0.1], [1, 1.1], [2, 2.1]],
        graphs.EDGES: [[1, 1.2], [2, 2.2]],
        graphs.SENDERS: [0, 1],
        graphs.RECEIVERS: [1, 2],
    }
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([graph_0, graph_1])
    nodes, edges = graph_networks.GraphNet._unpack_graphs_tuple(
        graphs_tuple=graphs_tuple, max_n_node=5, max_n_edge=None)
    self.assertAllClose(nodes,
                        [[[0, 0.1], [1, 1.1], [2, 2.1], [3, 3.1], [0, 0]],
                         [[0, 0.1], [1, 1.1], [2, 2.1], [0, 0], [0, 0]]])
    self.assertIsNone(edges)

  def test_unpack_graphs_tuple_no_node(self):
    graph_0 = {
        graphs.N_NODE: 4,
        graphs.N_EDGE: 3,
        graphs.NODES: [[0, 0.1], [1, 1.1], [2, 2.1], [3, 3.1]],
        graphs.EDGES: [[1, 1.2], [2, 2.2], [3, 3.2]],
        graphs.SENDERS: [0, 0, 0],
        graphs.RECEIVERS: [1, 2, 3],
    }
    graph_1 = {
        graphs.N_NODE: 3,
        graphs.N_EDGE: 2,
        graphs.NODES: [[0, 0.1], [1, 1.1], [2, 2.1]],
        graphs.EDGES: [[1, 1.2], [2, 2.2]],
        graphs.SENDERS: [0, 1],
        graphs.RECEIVERS: [1, 2],
    }
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([graph_0, graph_1])
    nodes, edges = graph_networks.GraphNet._unpack_graphs_tuple(
        graphs_tuple=graphs_tuple, max_n_node=None, max_n_edge=6)
    self.assertIsNone(nodes)
    self.assertAllClose(edges,
                        [[[1, 1.2], [2, 2.2], [3, 3.2], [0, 0], [0, 0], [0, 0]],
                         [[1, 1.2], [2, 2.2], [0, 0], [0, 0], [0, 0], [0, 0]]])

  def test_edge_dropout(self):
    (n_edge, edges, senders,
     receivers) = graph_networks.GraphNet._edge_dropout(n_edge=4,
                                                        edges=[
                                                            [0.0, 0.0],
                                                            [1.0, 1.0],
                                                            [2.0, 2.0],
                                                            [3.0, 3.0],
                                                        ],
                                                        senders=[0, 0, 0, 1],
                                                        receivers=[1, 2, 3, 0],
                                                        dropout_keep_prob=1.0)
    self.assertAllEqual(n_edge, 4)
    self.assertAllEqual(edges.shape, [4, 2])
    self.assertAllEqual(senders.shape, [4])
    self.assertAllEqual(receivers.shape, [4])

  def test_edge_dropout_half(self):
    (n_edge, edges, senders,
     receivers) = graph_networks.GraphNet._edge_dropout(n_edge=4,
                                                        edges=[
                                                            [0.0, 0.0],
                                                            [1.0, 1.0],
                                                            [2.0, 2.0],
                                                            [3.0, 3.0],
                                                        ],
                                                        senders=[0, 0, 0, 1],
                                                        receivers=[1, 2, 3, 0],
                                                        dropout_keep_prob=0.5)
    self.assertAllEqual(n_edge, 2)
    self.assertAllEqual(edges.shape, [2, 2])
    self.assertAllEqual(senders.shape, [2])
    self.assertAllEqual(receivers.shape, [2])

    (n_edge, edges, senders,
     receivers) = graph_networks.GraphNet._edge_dropout(n_edge=4,
                                                        edges=[
                                                            [0.0, 0.0],
                                                            [1.0, 1.0],
                                                            [2.0, 2.0],
                                                            [3.0, 3.0],
                                                        ],
                                                        senders=[0, 0, 0, 1],
                                                        receivers=[1, 2, 3, 0],
                                                        dropout_keep_prob=0.5)
    self.assertAllEqual(n_edge, 2)
    self.assertAllEqual(edges.shape, [2, 2])
    self.assertAllEqual(senders.shape, [2])
    self.assertAllEqual(receivers.shape, [2])

  def test_self_attention(self):
    options_str = r"""
      self_attention {
      }
    """
    options = graph_network_pb2.GraphNetwork()
    text_format.Merge(options_str, options)

    network = graph_networks.build_graph_network(options, is_training=False)
    self.assertIsInstance(network, graph_networks.SelfAttention)


if __name__ == '__main__':
  tf.test.main()
