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

import os
import json

from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt
import networkx as nx
import textwrap

flags.DEFINE_string('scenegraph_annotations_file', '',
                    'Scene graph annotations JSON file.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.scenegraph_annotations_file, '`scenegraph_annotations_file` missing.'

  logging.set_verbosity(logging.INFO)

  with open(FLAGS.scenegraph_annotations_file, 'r') as fid:
    images = json.load(fid)

  for image in images:
    image_data = plt.imread(image['coco_url'], format='jpg')

    plt.figure(figsize=(30, 20))

    # Plot image.
    plt.subplot(2, 3, 1)
    plt.imshow(image_data)
    plt.axis('off')

    # Plot scene graphs.
    plt_id = 2
    for sg in image['scene_graphs']:
      ax = plt.subplot(2, 3, plt_id)
      ax.set_title('\n'.join(textwrap.wrap(sg['phrase'], 40)), fontsize=8)
      plt_id += 1

      print(sg)

      g = nx.DiGraph()

      # Relations.
      relation_edges = []
      for r in sg['relationships']:
        assert len(r['text']) == 3
        from_node, to_node = (r['text'][0] + str(r['subject']),
                              r['text'][2] + str(r['object']))
        g.add_edge(from_node, to_node, label=r['predicate'], weight=0)
        relation_edges.append((from_node, to_node))

      # Attributes.
      attribute_edges = []
      for a in sg['attributes']:
        assert len(a['text']) == 3
        from_node, to_node = (a['text'][0] + str(a['subject']), a['text'][2])
        g.add_edge(from_node, to_node, label=a['predicate'], weight=1)
        attribute_edges.append((from_node, to_node))

      # Draw graph.
      pos = nx.spring_layout(g, iterations=200)
      nx.draw_networkx_nodes(g, pos, node_size=200, node_shape='o')
      nx.draw_networkx_edges(g, pos, edgelist=relation_edges)
      nx.draw_networkx_edges(g,
                             pos,
                             edgelist=attribute_edges,
                             arrows=False,
                             style='dashed')

      nx.draw_networkx_edge_labels(
          g,
          pos,
          edge_labels=dict([
              ((u, v), d['label']) for u, v, d in g.edges(data=True)
          ]))
      nx.draw_networkx_labels(g, pos, font_family='sans-serif')

    plt.show()


if __name__ == '__main__':
  app.run(main)
