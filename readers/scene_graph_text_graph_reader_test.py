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

from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


class SceneGraphTextGraphReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    batch_size = 17
    # input_pattern: "data-vspnet/tfrecords/caption-graph-hanwang/val.tfrecord-00000-of-00001"
    options_str = r"""
      scene_graph_text_graph_reader {
        input_pattern: "data-mscoco/tfrecords/scenegraphs_train2017.tfreocrd-00000-of-00020"
        batch_size: %i
        shuffle_buffer_size: 500
        prefetch_buffer_size: 500
        feature_dimensions: 1536
        max_n_proposal: 20
      }
    """ % (batch_size)
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      self.assertAllEqual(elem['id'].shape, [batch_size])
      self.assertAllEqual(elem['image/n_proposal'].shape, [batch_size])
      self.assertAllEqual(elem['image/proposal'].shape, [batch_size, 20, 4])
      self.assertAllEqual(elem['image/proposal/feature'].shape,
                          [batch_size, 20, 1536])

      max_n_triple = elem['scene_graph/n_triple'].numpy().max()
      self.assertAllEqual(elem['scene_graph/predicate'].shape,
                          [batch_size, max_n_triple])
      self.assertAllEqual(elem['scene_graph/subject'].shape,
                          [batch_size, max_n_triple])
      self.assertAllEqual(elem['scene_graph/subject/box'].shape,
                          [batch_size, max_n_triple, 4])
      self.assertAllEqual(elem['scene_graph/object'].shape,
                          [batch_size, max_n_triple])
      self.assertAllEqual(elem['scene_graph/object/box'].shape,
                          [batch_size, max_n_triple, 4])

      self.assertDTypeEqual(elem['id'], np.int64)
      self.assertDTypeEqual(elem['image/n_proposal'], np.int32)
      self.assertDTypeEqual(elem['image/proposal'], np.float32)
      self.assertDTypeEqual(elem['image/proposal/feature'], np.float32)
      self.assertDTypeEqual(elem['scene_graph/n_triple'], np.int32)
      self.assertDTypeEqual(elem['scene_graph/predicate'], np.object)
      self.assertDTypeEqual(elem['scene_graph/subject'], np.object)
      self.assertDTypeEqual(elem['scene_graph/subject/box'], np.float32)
      self.assertDTypeEqual(elem['scene_graph/object'], np.object)
      self.assertDTypeEqual(elem['scene_graph/object/box'], np.float32)

      # Check graphs.
      for scope in ['scene_pseudo_graph', 'scene_text_graph']:
        max_n_node = elem[scope + '/n_node'].numpy().max()
        max_n_edge = elem[scope + '/n_edge'].numpy().max()
        self.assertAllEqual(elem[scope + '/nodes'].shape,
                            [batch_size, max_n_node])
        self.assertAllEqual(elem[scope + '/edges'].shape,
                            [batch_size, max_n_edge])
        self.assertAllEqual(elem[scope + '/senders'].shape,
                            [batch_size, max_n_edge])
        self.assertAllEqual(elem[scope + '/receivers'].shape,
                            [batch_size, max_n_edge])

        self.assertDTypeEqual(elem[scope + '/n_node'], np.int32)
        self.assertDTypeEqual(elem[scope + '/n_edge'], np.int32)
        self.assertDTypeEqual(elem[scope + '/nodes'], np.object)
        self.assertDTypeEqual(elem[scope + '/edges'], np.object)
        self.assertDTypeEqual(elem[scope + '/senders'], np.int32)
        self.assertDTypeEqual(elem[scope + '/receivers'], np.int32)


if __name__ == '__main__':
  tf.test.main()
