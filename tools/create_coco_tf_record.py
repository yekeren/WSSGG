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

import os
import json

from absl import app
from absl import flags
from absl import logging

import zipfile
import numpy as np
import tensorflow as tf
from graph_nets import utils_np

flags.DEFINE_string('train_scenegraph_annotations_file', '',
                    'Scene graph annotations JSON file.')
flags.DEFINE_string('val_scenegraph_annotations_file', '',
                    'Scene graph annotations JSON file.')
flags.DEFINE_string('proposal_npz_directory', '',
                    'Path to the directory saving proposal data.')
flags.DEFINE_string('output_directory', '',
                    'Path to store the output annotation file.')

FLAGS = flags.FLAGS

_NUM_PROPOSAL_SUBDIRS = 10


def _create_tf_example(annot, num_proposals, proposals, proposal_features):
  """Creates tf example proto.

  Args:
    annot: JSON object containing scene graph annotations.
    num_proposals: An integer denoting the number of proposals.
    proposals: A [num_proposals, 4] np array.
    proposal_features: A [num_proposals, dims] np array.

  Returns:
    tf_example: A tf.train.Example proto.
  """

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode('utf8')]))

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _string_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[x.encode('utf8') for x in value]))

  def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  # Set the basic fields.
  feature_dict = {}
  for key, value in annot.items():
    if isinstance(value, int):
      feature_dict[key] = _int64_feature(value)
    elif isinstance(value, str):
      feature_dict[key] = _string_feature(value)

  # Set the image and proposals.
  feature_dict.update({
      'image/n_proposal':
          _int64_feature(len(proposals)),
      'image/proposal/bbox/ymin':
          _float_feature_list(proposals[:, 0].tolist()),
      'image/proposal/bbox/xmin':
          _float_feature_list(proposals[:, 1].tolist()),
      'image/proposal/bbox/ymax':
          _float_feature_list(proposals[:, 2].tolist()),
      'image/proposal/bbox/xmax':
          _float_feature_list(proposals[:, 3].tolist()),
      'image/proposal/feature':
          _float_feature_list(proposal_features.flatten().tolist()),
  })

  # Set the scene graph info.
  data_dict_list = []
  n_entity = []
  n_relation = []
  captions = []
  for sg in annot['scene_graphs']:
    assert all((len(x['names']) == 1 for x in sg['objects']))

    # Nodes (objects + attributes).
    objects = [x['names'][0] for x in sg['objects']]
    attributes = [x['attribute'] for x in sg['attributes']]
    nodes = objects + attributes
    captions.append(sg['phrase'])

    n_entity.append(len(objects))
    n_relation.append(len(sg['relationships']))

    # Edges (relationships + attributes), bi-directional.
    senders, receivers, edges = [], [], []
    for r in sg['relationships']:
      senders.append(r['subject'])
      receivers.append(r['object'])
      edges.append(r['predicate'])
    for a_id, a in enumerate(sg['attributes']):
      senders.append(a['subject'])
      receivers.append(len(objects) + a_id)
      edges.append(a['predicate'])
    data_dict_list.append({
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers
    })

  graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)
  feature_dict.update({
      'scene_text_graph/caption': _string_feature_list(captions),
      'scene_text_graph/n_entity': _int64_feature_list(n_entity),
      'scene_text_graph/n_relation': _int64_feature_list(n_relation),
      'scene_text_graph/n_node': _int64_feature_list(graphs_tuple.n_node),
      'scene_text_graph/n_edge': _int64_feature_list(graphs_tuple.n_edge),
      'scene_text_graph/nodes': _string_feature_list(graphs_tuple.nodes),
      'scene_text_graph/edges': _string_feature_list(graphs_tuple.edges),
      'scene_text_graph/senders': _int64_feature_list(graphs_tuple.senders),
      'scene_text_graph/receivers': _int64_feature_list(graphs_tuple.receivers)
  })

  tf_example = tf.train.Example(features=tf.train.Features(
      feature=feature_dict))
  return tf_example


def _create_tf_record_from_annotations(scenegraph_annotations_file,
                                       proposal_npz_directory, tf_record_file,
                                       num_output_parts):
  """Creates tf record files from scenegraphs annotations.

  Args:
    scenegraph_annotations_file: JSON file containing scene graph annotations.
    proposal_npz_directory: Path to the directory saving proposal data.
    tf_record_file: Tf record file containing tf.example protos.
    num_output_parts: Number of output partitions.
  """
  with tf.io.gfile.GFile(scenegraph_annotations_file, 'r') as fid:
    annots = json.load(fid)

  writers = []
  for i in range(num_output_parts):
    filename = tf_record_file + '-%05d-of-%05d' % (i, num_output_parts)
    writers.append(tf.io.TFRecordWriter(filename))

  for i, annot in enumerate(annots):
    # Read proposals.
    npz_path = os.path.join(proposal_npz_directory,
                            str(annot['id'] % _NUM_PROPOSAL_SUBDIRS),
                            '%012d.npz' % (annot['id']))

    with tf.io.gfile.GFile(npz_path, 'rb') as fid:
      data = np.load(fid)
      num_proposals = data['num_proposals']
      proposals = data['proposals']
      proposal_features = data['proposal_features']

      assert num_proposals == 20

    # Encode tf example.
    tf_example = _create_tf_example(annot, num_proposals, proposals,
                                    proposal_features)
    writers[i % num_output_parts].write(tf_example.SerializeToString())
    if (i + 1) % 500 == 0:
      logging.info('On example %i/%i', i + 1, len(annots))

  for writer in writers:
    writer.close()
  logging.info('Done')


def main(_):
  assert FLAGS.train_scenegraph_annotations_file, '`train_scenegraph_annotations_file` missing.'
  assert FLAGS.val_scenegraph_annotations_file, '`val_scenegraph_annotations_file` missing.'
  assert FLAGS.proposal_npz_directory, '`proposal_npz_directory` missing.'
  assert FLAGS.output_directory, '`output_directory` missing.'

  logging.set_verbosity(logging.INFO)

  tf.gfile.MakeDirs(FLAGS.output_directory)

  output_train_file = os.path.join(FLAGS.output_directory,
                                   'scenegraphs_train2017.tfreocrd')
  # output_val_file = os.path.join(FLAGS.output_directory,
  #                                'scenegraphs_val2017.tfrecord')

  _create_tf_record_from_annotations(FLAGS.train_scenegraph_annotations_file,
                                     FLAGS.proposal_npz_directory,
                                     output_train_file, 20)
  # _create_tf_record_from_annotations(FLAGS.val_scenegraph_annotations_file,
  #                                    output_val_file, 5)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
