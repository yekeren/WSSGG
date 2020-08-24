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
      'id': tf.io.FixedLenFeature([], tf.int64),
      'coco_url': tf.io.FixedLenFeature([], tf.string),
      'image/proposal/bbox/ymin': tf.io.VarLenFeature(tf.float32),
      'image/proposal/bbox/xmin': tf.io.VarLenFeature(tf.float32),
      'image/proposal/bbox/ymax': tf.io.VarLenFeature(tf.float32),
      'image/proposal/bbox/xmax': tf.io.VarLenFeature(tf.float32),
      'graphs/n_node': tf.io.VarLenFeature(tf.int64),
      'graphs/n_edge': tf.io.VarLenFeature(tf.int64),
      'graphs/nodes': tf.io.VarLenFeature(tf.string),
      'graphs/edges': tf.io.VarLenFeature(tf.string),
      'graphs/senders': tf.io.VarLenFeature(tf.int64),
      'graphs/receivers': tf.io.VarLenFeature(tf.int64),
  }

  parsed = tf.parse_single_example(example, example_fmt)

  # Decode proposals.
  bbox_decoder = tfexample_decoder.BoundingBox(prefix='image/proposal/bbox/')
  proposals = bbox_decoder.tensors_to_item(parsed)

  # Decode scene graphs.

  graphs = GraphsTuple(
      globals=None,
      nodes=tf.sparse_tensor_to_dense(parsed['graphs/nodes'], ''),
      edges=tf.sparse_tensor_to_dense(parsed['graphs/edges'], ''),
      receivers=tf.sparse_tensor_to_dense(parsed['graphs/receivers'], 0),
      senders=tf.sparse_tensor_to_dense(parsed['graphs/senders'], 0),
      n_node=tf.sparse_tensor_to_dense(parsed['graphs/n_node'], 0),
      n_edge=tf.sparse_tensor_to_dense(parsed['graphs/n_edge'], 0))

  num_graphs = utils_tf.get_num_graphs(graphs)
  index = tf.random.uniform([], minval=0, maxval=num_graphs, dtype=tf.int32)
  graph = utils_tf.get_graph(graphs, index)

  feature_dict = {
      'id': parsed['id'],
      'url': parsed['coco_url'],
      'image/n_proposal': tf.shape(proposals)[0],
      'image/proposals': proposals,
      'graph/n_node': tf.reshape(graph.n_node, []),
      'graph/n_edge': tf.reshape(graph.n_edge, []),
      'graph/nodes': graph.nodes,
      'graph/edges': graph.edges,
      'graph/senders': graph.senders,
      'graph/receivers': graph.receivers,
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
      'url': [],
      'image/n_proposal': [],
      'image/proposals': [None, 4],
      'graph/n_node': [],
      'graph/n_edge': [],
      'graph/nodes': [None],
      'graph/edges': [None],
      'graph/senders': [None],
      'graph/receivers': [None],
  }
  dataset = dataset.padded_batch(options.batch_size,
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
  if not isinstance(options, reader_pb2.COCOReader):
    raise ValueError('options has to be an instance of COCOReader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
