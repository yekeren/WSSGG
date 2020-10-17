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

import matplotlib
matplotlib.use('TkAgg')

from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import networkx as nx

from readers import reader
from google.protobuf import text_format

from modeling.utils import visualization

flags.DEFINE_string(
    'tf_record_file',
    'data-vspnet/tfrecords/graph-hanwang/val.tfrecord-00000-of-00001',
    'Path to the input record file.')

flags.DEFINE_string('image_directory', 'data-vspnet/images',
                    'Path to the directory saving images.')

FLAGS = flags.FLAGS

from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


def main(_):
  assert FLAGS.tf_record_file, '`tf_record_file` missing.'
  assert FLAGS.image_directory, '`image_directory` missing.'

  options_str = r"""
    scene_graph_pseudo_graph_reader {
      input_pattern: "%s"
      batch_size: 1
      shuffle_buffer_size: 10
      prefetch_buffer_size: 10
      feature_dimensions: 1536
    }
  """ % (FLAGS.tf_record_file)
  options = text_format.Merge(options_str, reader_pb2.Reader())

  dataset = reader.get_input_fn(options, is_training=False)()
  for elem_id, elem in enumerate(dataset.take(50)):
    if elem_id < 31:
      continue
    # Read image.
    image_id = elem['id'].numpy()[0]
    image = Image.open(
        os.path.join(FLAGS.image_directory, '{}.jpg'.format(image_id)))
    image = np.array(image)

    # Visualization.
    plt.figure(figsize=(10, 20))

    # - Original image.
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.axis('off')

    # - Proposals.
    plt.subplot(3, 3, 2)
    n_proposal = elem['image/n_proposal'].numpy()[0]
    proposals = elem['image/proposal'].numpy()[0]

    image_with_proposals = visualization.draw_bounding_box_py_func_fn(
        image.copy(), n_proposal, proposals, None, None)
    plt.imshow(image_with_proposals)
    plt.axis('off')

    # - Relation triples.
    id_offset = 3
    n_triple = elem['scene_graph/n_triple'].numpy()[0]
    for i in range(n_triple):
      sub = elem['scene_graph/subject'].numpy()[0, i]
      obj = elem['scene_graph/object'].numpy()[0, i]
      pred = elem['scene_graph/predicate'].numpy()[0, i]

      sub_box = elem['scene_graph/subject/box'].numpy()[0, i]
      obj_box = elem['scene_graph/object/box'].numpy()[0, i]

      image_with_triple = visualization.draw_bounding_box_py_func_fn(
          image.copy(), 2, [sub_box, obj_box], [sub, obj], None)
      image_with_triple = visualization.draw_arrow_py_func_fn(
          image_with_triple, 1, [sub_box[0]], [sub_box[1]], [obj_box[0]],
          [obj_box[1]], [pred], None)

      plt.subplot(3, 3, id_offset)
      plt.imshow(image_with_triple)
      plt.axis('off')

      id_offset += 1
      if id_offset >= 8:
        break

    # - Pseudo graph.
    n_node = elem['scene_pseudo_graph/n_node'].numpy()[0]
    n_edge = elem['scene_pseudo_graph/n_edge'].numpy()[0]
    nodes = elem['scene_pseudo_graph/nodes'].numpy()[0]
    edges = elem['scene_pseudo_graph/edges'].numpy()[0]
    senders = elem['scene_pseudo_graph/senders'].numpy()[0]
    receivers = elem['scene_pseudo_graph/receivers'].numpy()[0]

    plt.subplot(3, 3, 9)

    g = nx.DiGraph()
    for sender, receiver, edge_label in zip(senders[:n_edge],
                                            receivers[:n_edge], edges[:n_edge]):
      edge_label = edge_label.decode('ascii')
      from_node = '%s:%i' % (nodes[sender].decode('ascii'), sender)
      to_node = '%s:%i' % (nodes[receiver].decode('ascii'), receiver)
      g.add_edge(from_node, to_node, label=edge_label, weight=0)

    pos = nx.spring_layout(g, iterations=100)
    nx.draw_networkx_nodes(g, pos, node_size=200, node_shape='o')
    nx.draw_networkx_labels(g, pos, font_size=8, font_family='sans-serif')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=dict([((u, v), d['label']) for u, v, d in g.edges(data=True)
                         ]))

    print('n_triple=%i' % n_triple)
    print('n_edge=%i' % n_edge)
    plt.tight_layout()
    plt.show()

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
