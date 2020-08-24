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

import reader
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple

from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


class COCOReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    options_str = r"""
      coco_reader {
        input_pattern: "data-mscoco/tfrecords/scenegraphs_val2017.tfrecord-00000-of-00005"
        batch_size: 17
        shuffle_buffer_size: 10
        prefetch_buffer_size: 10
      }
    """
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      self.assertAllEqual(elem['id'].shape, [17])
      self.assertAllEqual(elem['url'].shape, [17])
      self.assertAllEqual(elem['image/n_proposal'].shape, [17])
      self.assertAllEqual(elem['image/proposals'].shape, [17, 2000, 4])
      max_n_node = tf.reduce_max(elem['graph/n_node'])
      max_n_edge = tf.reduce_max(elem['graph/n_edge'])
      self.assertAllEqual(elem['graph/n_node'].shape, [17])
      self.assertAllEqual(elem['graph/n_edge'].shape, [17])
      self.assertAllEqual(elem['graph/nodes'].shape, [17, max_n_node])
      self.assertAllEqual(elem['graph/edges'].shape, [17, max_n_edge])
      self.assertAllEqual(elem['graph/senders'].shape, [17, max_n_edge])
      self.assertAllEqual(elem['graph/receivers'].shape, [17, max_n_edge])

      self.assertDTypeEqual(elem['id'], np.int64)
      self.assertDTypeEqual(elem['url'], np.object)
      self.assertDTypeEqual(elem['image/n_proposal'], np.int32)
      self.assertDTypeEqual(elem['image/proposals'], np.float32)
      self.assertDTypeEqual(elem['graph/n_node'], np.int32)
      self.assertDTypeEqual(elem['graph/n_edge'], np.int32)
      self.assertDTypeEqual(elem['graph/nodes'], np.object)
      self.assertDTypeEqual(elem['graph/edges'], np.object)
      self.assertDTypeEqual(elem['graph/senders'], np.int32)
      self.assertDTypeEqual(elem['graph/receivers'], np.int32)

  def test_coco_graphs(self):
    options_str = r"""
      coco_reader {
        input_pattern: "data-mscoco/tfrecords/scenegraphs_val2017.tfrecord-00000-of-00005"
        batch_size: 1
        shuffle_buffer_size: 10
        prefetch_buffer_size: 10
      }
    """
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      graphs = GraphsTuple(globals=None,
                           n_node=elem['graph/n_node'],
                           n_edge=elem['graph/n_edge'],
                           nodes=tf.reshape(elem['graph/nodes'], [-1]),
                           edges=tf.reshape(elem['graph/edges'], [-1]),
                           receivers=tf.reshape(elem['graph/receivers'], [-1]),
                           senders=tf.reshape(elem['graph/senders'], [-1]))
      graphs_nx = utils_np.graphs_tuple_to_networkxs(graphs)
      self.assertEqual(len(graphs_nx), 1)

      # Check whether the numbers of node/edges are correct.
      graph_nx = graphs_nx[0]
      self.assertEqual(graph_nx.number_of_nodes(),
                       elem['graph/n_node'][0].numpy())
      self.assertEqual(graph_nx.number_of_edges(),
                       elem['graph/n_edge'][0].numpy())

      # Check whether the senders and receivers are correct.
      for (from1, to1), from2, to2 in zip(graph_nx.edges(),
                                          elem['graph/senders'][0].numpy(),
                                          elem['graph/receivers'][0].numpy()):
        self.assertEqual(from1, from2)
        self.assertEqual(to1, to2)


if __name__ == '__main__':
  tf.test.main()
