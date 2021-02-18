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
          tf.io.FixedLenFeature([], tf.int64, default_value=-1),
      # Proposals
      'image/n_proposal':
          tf.io.FixedLenFeature([], tf.int64, default_value=0),
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
      # Caption graph.
      'caption_graph/caption':
          tf.io.VarLenFeature(tf.string),
      'caption_graph/n_node':
          tf.io.VarLenFeature(tf.int64),
      'caption_graph/n_edge':
          tf.io.VarLenFeature(tf.int64),
      'caption_graph/nodes':
          tf.io.VarLenFeature(tf.string),
      'caption_graph/edges':
          tf.io.VarLenFeature(tf.string),
      'caption_graph/senders':
          tf.io.VarLenFeature(tf.int64),
      'caption_graph/receivers':
          tf.io.VarLenFeature(tf.int64),
      # Ground-truth.
      'scene_graph/n_relation':
          tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'scene_graph/predicate':
          tf.io.VarLenFeature(tf.string),
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
  }

  # Decode proposals.
  parsed = tf.parse_single_example(example, example_fmt)
  proposals = tfexample_decoder.BoundingBox(
      prefix='image/proposal/bbox/').tensors_to_item(parsed)

  n_proposal = tf.minimum(parsed['image/n_proposal'], options.max_n_proposal)
  proposals = proposals[:options.max_n_proposal, :]
  proposal_features = tf.reshape(
      tf.sparse_tensor_to_dense(parsed['image/proposal/feature']),
      [-1, options.feature_dimensions])[:options.max_n_proposal, :]

  # Decode and randomly sample a caption graph.
  graphs = GraphsTuple(
      globals=None,
      n_node=tf.sparse_tensor_to_dense(parsed['caption_graph/n_node'], 0),
      n_edge=tf.sparse_tensor_to_dense(parsed['caption_graph/n_edge'], 0),
      nodes=tf.sparse_tensor_to_dense(parsed['caption_graph/nodes'], ''),
      edges=tf.sparse_tensor_to_dense(parsed['caption_graph/edges'], ''),
      senders=tf.sparse_tensor_to_dense(parsed['caption_graph/senders'], 0),
      receivers=tf.sparse_tensor_to_dense(parsed['caption_graph/receivers'], 0))
  num_graphs = utils_tf.get_num_graphs(graphs)
  index = tf.random.uniform([], minval=0, maxval=num_graphs, dtype=tf.int32)

  caption = tf.sparse_tensor_to_dense(parsed['caption_graph/caption'])[index]
  caption_graph = utils_tf.get_graph(graphs, index)

  # Decode the ground-truth.
  subject_boxes = tfexample_decoder.BoundingBox(
      prefix='scene_graph/subject/bbox/').tensors_to_item(parsed)
  object_boxes = tfexample_decoder.BoundingBox(
      prefix='scene_graph/object/bbox/').tensors_to_item(parsed)

  feature_dict = {
      'id':
          parsed['id'],
      # Proposals.
      'image/n_proposal':
          n_proposal,
      'image/proposal':
          proposals,
      'image/proposal/feature':
          proposal_features,
      # Caption graph.
      'caption_graph/caption':
          caption,
      'caption_graph/n_node':
          caption_graph.n_node[0],
      'caption_graph/n_edge':
          caption_graph.n_edge[0],
      'caption_graph/nodes':
          caption_graph.nodes,
      'caption_graph/edges':
          caption_graph.edges,
      'caption_graph/senders':
          caption_graph.senders,
      'caption_graph/receivers':
          caption_graph.receivers,
      # Ground-truth.
      'scene_graph/n_relation':
          parsed['scene_graph/n_relation'],
      'scene_graph/predicate':
          tf.sparse_tensor_to_dense(parsed['scene_graph/predicate'], ''),
      'scene_graph/subject':
          tf.sparse_tensor_to_dense(parsed['scene_graph/subject'], ''),
      'scene_graph/subject/box':
          subject_boxes,
      'scene_graph/object':
          tf.sparse_tensor_to_dense(parsed['scene_graph/object'], ''),
      'scene_graph/object/box':
          object_boxes,
  }

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

  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(options.shuffle_buffer_size)

  padded_shapes = {
      'id': [],
      'image/n_proposal': [],
      'image/proposal': [None, 4],
      'image/proposal/feature': [None, options.feature_dimensions],
      'caption_graph/caption': [],
      'caption_graph/n_node': [],
      'caption_graph/n_edge': [],
      'caption_graph/nodes': [None],
      'caption_graph/edges': [None],
      'caption_graph/senders': [None],
      'caption_graph/receivers': [None],
      'scene_graph/n_relation': [],
      'scene_graph/predicate': [None],
      'scene_graph/subject': [None],
      'scene_graph/subject/box': [None, 4],
      'scene_graph/object': [None],
      'scene_graph/object/box': [None, 4],
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
  if not isinstance(options, reader_pb2.CaptionGraphReader):
    raise ValueError(
        'options has to be an instance of SceneGraphTextGraphReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
