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

flags.DEFINE_string('train_image_file', '', 'Training image zip file.')
flags.DEFINE_string('val_image_file', '', 'Validation image zip file.')
flags.DEFINE_string('train_scenegraph_annotations_file', '',
                    'Scene graph annotations JSON file.')
flags.DEFINE_string('val_scenegraph_annotations_file', '',
                    'Scene graph annotations JSON file.')
flags.DEFINE_string('proposal_nparray_directory', '',
                    'Path to the directory saving proposal data.')
flags.DEFINE_string('output_directory', '',
                    'Path to store the output annotation file.')
flags.DEFINE_integer('max_proposals', 2000,
                     'Maximum number of proposals to be used.')

FLAGS = flags.FLAGS

_NUM_PROPOSAL_SUBDIRS = 10


def _create_tf_example(annot, encoded_jpg, proposals):
  """Creates tf example proto.

  Args:
    annot: JSON object containing scene graph annotations.
    encoded_jpg: A string represent the encoded jpg data.
    proposals: A [num_proposals, 4] np array.

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
      'image/encoded': _bytes_feature(encoded_jpg),
      'image/format': _string_feature('jpeg'),
      'image/proposal/bbox/ymin': _float_feature_list(proposals[:, 0].tolist()),
      'image/proposal/bbox/xmin': _float_feature_list(proposals[:, 1].tolist()),
      'image/proposal/bbox/ymax': _float_feature_list(proposals[:, 2].tolist()),
      'image/proposal/bbox/xmax': _float_feature_list(proposals[:, 3].tolist())
  })

  # Set the scene graph info.
  data_dict_list = []
  for sg in annot['scene_graphs']:

    assert all((len(x['names']) == 1 for x in sg['objects']))

    # Nodes (objects + attributes).
    objects = [x['names'][0] for x in sg['objects']]
    attributes = [x['attribute'] for x in sg['attributes']]
    nodes = objects + attributes

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
      'graphs/n_node': _int64_feature_list(graphs_tuple.n_node),
      'graphs/n_edge': _int64_feature_list(graphs_tuple.n_edge),
      'graphs/nodes': _string_feature_list(graphs_tuple.nodes),
      'graphs/edges': _string_feature_list(graphs_tuple.edges),
      'graphs/senders': _int64_feature_list(graphs_tuple.senders),
      'graphs/receivers': _int64_feature_list(graphs_tuple.receivers)
  })

  tf_example = tf.train.Example(features=tf.train.Features(
      feature=feature_dict))
  return tf_example


def _create_tf_record_from_annotations(image_zip_file,
                                       scenegraph_annotations_file,
                                       tf_record_file, num_output_parts):
  """Creates tf record files from scenegraphs annotations.

  Args:
    image_zip_file: Path to the .zip images.
    scenegraph_annotations_file: JSON file containing caption annotations.
    tf_record_file: Tf record file containing tf.example protos.
    num_output_parts: Number of output partitions.
  """
  with tf.io.gfile.GFile(scenegraph_annotations_file, 'r') as fid:
    annots = json.load(fid)

  image_dir = os.path.split(image_zip_file)[1].split('.')[0]

  writers = []
  for i in range(num_output_parts):
    filename = tf_record_file + '-%05d-of-%05d' % (i, num_output_parts)
    writers.append(tf.io.TFRecordWriter(filename))

  with zipfile.ZipFile(image_zip_file) as zf:
    for i, annot in enumerate(annots):
      # Read image and proposals.
      with zf.open(os.path.join(image_dir, annot['file_name'])) as fid:
        encoded_jpg = fid.read()

      nparray_path = os.path.join(FLAGS.proposal_nparray_directory,
                                  str(annot['id'] % _NUM_PROPOSAL_SUBDIRS),
                                  '%012d.npy' % (annot['id']))

      with tf.io.gfile.GFile(nparray_path, 'rb') as fid:
        proposals = np.load(fid)
        proposals = proposals[:FLAGS.max_proposals, :]

      # Encode tf example.
      tf_example = _create_tf_example(annot, encoded_jpg, proposals)
      writers[i % num_output_parts].write(tf_example.SerializeToString())
      if (i + 1) % 500 == 0:
        logging.info('On example %i/%i', i + 1, len(annots))

  for writer in writers:
    writer.close()
  logging.info('Done')


def main(_):
  assert FLAGS.train_image_file, '`train_image_file` missing.'
  assert FLAGS.val_image_file, '`val_image_file` missing.'
  assert FLAGS.val_scenegraph_annotations_file, '`val_scenegraph_annotations_file` missing.'
  assert FLAGS.train_scenegraph_annotations_file, '`train_scenegraph_annotations_file` missing.'
  assert FLAGS.proposal_nparray_directory, '`proposal_nparray_directory` missing.'
  assert FLAGS.output_directory, '`output_directory` missing.'

  logging.set_verbosity(logging.INFO)

  tf.gfile.MakeDirs(FLAGS.output_directory)

  output_val_file = os.path.join(FLAGS.output_directory,
                                 'scenegraphs_val2017.tfrecord')
  output_train_file = os.path.join(FLAGS.output_directory,
                                   'scenegraphs_train2017.tfreocrd')

  _create_tf_record_from_annotations(FLAGS.val_image_file,
                                     FLAGS.val_scenegraph_annotations_file,
                                     output_val_file, 5)
  _create_tf_record_from_annotations(FLAGS.train_image_file,
                                     FLAGS.train_scenegraph_annotations_file,
                                     output_train_file, 20)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
