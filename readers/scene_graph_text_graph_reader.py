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
"""Reads from tfrecord files and yields batched tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from protos import reader_pb2
from tf_slim import tfexample_decoder
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple


def _parse_single_example(example, options):
  """Parses a single tf.Example proto.

  Args:
    example: An Example proto.
    options: An instance of reader_pb2.Reader.

  Returns:
    A dictionary indexed by tensor name.
  """
  # Initialize `keys_to_features`.
  example_fmt = {
      'id':
          tf.io.FixedLenFeature([], tf.int64),
      # Proposals
      'image/n_proposal':
          tf.io.FixedLenFeature([], tf.int64),
      'image/proposal/bbox/ymin':
          tf.io.VarLenFeature(tf.float32),
      'image/proposal/bbox/xmin':
          tf.io.VarLenFeature(tf.float32),
      'image/proposal/bbox/ymax':
          tf.io.VarLenFeature(tf.float32),
      'image/proposal/bbox/xmax':
          tf.io.VarLenFeature(tf.float32),
      'image/proposal/feature':
          tf.io.VarLenFeature(tf.float32),
      # Scene graph triplets.
      'scene_graph/n_triple':
          tf.io.FixedLenFeature([], tf.int64, default_value=0),
      # - Predicate.
      'scene_graph/predicate':
          tf.io.VarLenFeature(tf.string),
      # - Subject.
      'scene_graph/subject':
          tf.io.VarLenFeature(tf.string),
      'scene_graph/subject/bbox/ymin':
          tf.io.VarLenFeature(tf.float32),
      'scene_graph/subject/bbox/xmin':
          tf.io.VarLenFeature(tf.float32),
      'scene_graph/subject/bbox/ymax':
          tf.io.VarLenFeature(tf.float32),
      'scene_graph/subject/bbox/xmax':
          tf.io.VarLenFeature(tf.float32),
      # - Object.
      'scene_graph/object':
          tf.io.VarLenFeature(tf.string),
      'scene_graph/object/bbox/ymin':
          tf.io.VarLenFeature(tf.float32),
      'scene_graph/object/bbox/xmin':
          tf.io.VarLenFeature(tf.float32),
      'scene_graph/object/bbox/ymax':
          tf.io.VarLenFeature(tf.float32),
      'scene_graph/object/bbox/xmax':
          tf.io.VarLenFeature(tf.float32),
      # Scene graph pseudo graph.
      'scene_pseudo_graph/n_node':
          tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'scene_pseudo_graph/n_edge':
          tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'scene_pseudo_graph/nodes':
          tf.io.VarLenFeature(tf.string),
      'scene_pseudo_graph/edges':
          tf.io.VarLenFeature(tf.string),
      'scene_pseudo_graph/senders':
          tf.io.VarLenFeature(tf.int64),
      'scene_pseudo_graph/receivers':
          tf.io.VarLenFeature(tf.int64),
      # Scene graph text graph.
      'scene_text_graph/caption':
          tf.io.VarLenFeature(tf.string),
      'scene_text_graph/n_entity':
          tf.io.VarLenFeature(tf.int64),
      'scene_text_graph/n_relation':
          tf.io.VarLenFeature(tf.int64),
      'scene_text_graph/n_node':
          tf.io.VarLenFeature(tf.int64),
      'scene_text_graph/n_edge':
          tf.io.VarLenFeature(tf.int64),
      'scene_text_graph/nodes':
          tf.io.VarLenFeature(tf.string),
      'scene_text_graph/edges':
          tf.io.VarLenFeature(tf.string),
      'scene_text_graph/senders':
          tf.io.VarLenFeature(tf.int64),
      'scene_text_graph/receivers':
          tf.io.VarLenFeature(tf.int64),
  }

  parsed = tf.parse_single_example(example, example_fmt)

  # Decode bounding boxes.
  proposals = tfexample_decoder.BoundingBox(
      prefix='image/proposal/bbox/').tensors_to_item(parsed)
  subject_boxes = tfexample_decoder.BoundingBox(
      prefix='scene_graph/subject/bbox/').tensors_to_item(parsed)
  object_boxes = tfexample_decoder.BoundingBox(
      prefix='scene_graph/object/bbox/').tensors_to_item(parsed)

  # Decode and randomly get one caption graph.
  graphs = GraphsTuple(
      globals=None,
      n_node=tf.sparse_tensor_to_dense(parsed['scene_text_graph/n_node'], 0),
      n_edge=tf.sparse_tensor_to_dense(parsed['scene_text_graph/n_edge'], 0),
      nodes=tf.sparse_tensor_to_dense(parsed['scene_text_graph/nodes'], ''),
      edges=tf.sparse_tensor_to_dense(parsed['scene_text_graph/edges'], ''),
      senders=tf.sparse_tensor_to_dense(parsed['scene_text_graph/senders'], 0),
      receivers=tf.sparse_tensor_to_dense(parsed['scene_text_graph/receivers'],
                                          0))
  num_graphs = utils_tf.get_num_graphs(graphs)
  index = tf.random.uniform([], minval=0, maxval=num_graphs, dtype=tf.int32)
  text_graph = utils_tf.get_graph(graphs, index)
  text_graph_n_entity = tf.sparse_tensor_to_dense(
      parsed['scene_text_graph/n_entity'])[index]
  text_graph_n_relation = tf.sparse_tensor_to_dense(
      parsed['scene_text_graph/n_relation'])[index]
  text_graph_caption = tf.sparse_tensor_to_dense(
      parsed['scene_text_graph/caption'])[index]

  feature_dict = {
      'id':
          parsed['id'],
      # Proposals.
      'image/n_proposal':
          tf.minimum(parsed['image/n_proposal'], options.max_n_proposal),
      'image/proposal':
          proposals[:options.max_n_proposal, :],
      'image/proposal/feature':
          tf.reshape(
              tf.sparse_tensor_to_dense(parsed['image/proposal/feature']),
              [-1, options.feature_dimensions])[:options.max_n_proposal, :],
      # Scene graph triplets.
      'scene_graph/n_triple':
          parsed['scene_graph/n_triple'],
      # - Predicate.
      'scene_graph/predicate':
          tf.sparse_tensor_to_dense(parsed['scene_graph/predicate'], ''),
      # - Subject.
      'scene_graph/subject':
          tf.sparse_tensor_to_dense(parsed['scene_graph/subject'], ''),
      'scene_graph/subject/box':
          subject_boxes,
      # - Object.
      'scene_graph/object':
          tf.sparse_tensor_to_dense(parsed['scene_graph/object'], ''),
      'scene_graph/object/box':
          object_boxes,
      # Scene graph pseudo graph.
      'scene_pseudo_graph/n_node':
          parsed['scene_pseudo_graph/n_node'],
      'scene_pseudo_graph/n_edge':
          parsed['scene_pseudo_graph/n_edge'],
      'scene_pseudo_graph/nodes':
          tf.sparse_tensor_to_dense(parsed['scene_pseudo_graph/nodes']),
      'scene_pseudo_graph/edges':
          tf.sparse_tensor_to_dense(parsed['scene_pseudo_graph/edges']),
      'scene_pseudo_graph/senders':
          tf.sparse_tensor_to_dense(parsed['scene_pseudo_graph/senders']),
      'scene_pseudo_graph/receivers':
          tf.sparse_tensor_to_dense(parsed['scene_pseudo_graph/receivers']),
      # Scene graph text graph.
      'scene_text_graph/caption':
          text_graph_caption,
      'scene_text_graph/n_entity':
          text_graph_n_entity,
      'scene_text_graph/n_relation':
          text_graph_n_relation,
      'scene_text_graph/n_node':
          text_graph.n_node[0],
      'scene_text_graph/n_edge':
          text_graph.n_edge[0],
      'scene_text_graph/nodes':
          text_graph.nodes,
      'scene_text_graph/edges':
          text_graph.edges,
      'scene_text_graph/senders':
          text_graph.senders,
      'scene_text_graph/receivers':
          text_graph.receivers,
  }
  feature_dict['scene_graph/n_triple'] = tf.shape(
      feature_dict['scene_graph/subject'])[0]

  for key in feature_dict.keys():
    if key != 'id' and feature_dict[key].dtype == tf.int64:
      feature_dict[key] = tf.cast(feature_dict[key], tf.int32)

  return feature_dict


def _create_dataset(options, is_training, input_pipeline_context=None):
  """Creates dataset object based on options.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.
    input_pipeline_context: A tf.distribute.InputContext instance.

  Returns:
    A tf.data.Dataset object.
  """
  batch_size = options.batch_size
  dataset = tf.data.Dataset.list_files(options.input_pattern[:],
                                       shuffle=is_training)
  dataset = dataset.interleave(tf.data.TFRecordDataset,
                               cycle_length=options.interleave_cycle_length)

  parse_fn = lambda x: _parse_single_example(x, options)
  dataset = dataset.map(map_func=parse_fn,
                        num_parallel_calls=options.num_parallel_calls)
  # dataset = dataset.cache()

  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(options.shuffle_buffer_size)

  padded_shapes = {
      'id': [],
      'image/n_proposal': [],
      'image/proposal': [None, 4],
      'image/proposal/feature': [None, options.feature_dimensions],
      'scene_graph/n_triple': [],
      'scene_graph/predicate': [None],
      'scene_graph/subject': [None],
      'scene_graph/subject/box': [None, 4],
      'scene_graph/object': [None],
      'scene_graph/object/box': [None, 4],
      'scene_pseudo_graph/n_node': [],
      'scene_pseudo_graph/n_edge': [],
      'scene_pseudo_graph/nodes': [None],
      'scene_pseudo_graph/edges': [None],
      'scene_pseudo_graph/senders': [None],
      'scene_pseudo_graph/receivers': [None],
      'scene_text_graph/caption': [],
      'scene_text_graph/n_entity': [],
      'scene_text_graph/n_relation': [],
      'scene_text_graph/n_node': [],
      'scene_text_graph/n_edge': [],
      'scene_text_graph/nodes': [None],
      'scene_text_graph/edges': [None],
      'scene_text_graph/senders': [None],
      'scene_text_graph/receivers': [None],
  }
  dataset = dataset.padded_batch(batch_size,
                                 padded_shapes=padded_shapes,
                                 drop_remainder=True)
  dataset = dataset.prefetch(options.prefetch_buffer_size)
  return dataset


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.SceneGraphTextGraphReader):
    raise ValueError(
        'options has to be an instance of SceneGraphTextGraphReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
