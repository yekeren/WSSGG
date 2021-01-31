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
""" 
Weakly supervised scene graph generation model inspired by:
  - Zareian et al. 2020 (VSPNet)
  - Ye et al. 2019 (Cap2det) (our previous work).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json

import numpy as np
import tensorflow as tf

import tf_slim as slim

from protos import model_pb2

import sonnet as snt
from graph_nets import graphs
from graph_nets import modules
from graph_nets import blocks
from graph_nets import _base
from graph_nets import utils_tf

from models.graph_mps import GraphMPS
from models.graph_nms import GraphNMS

from modeling.layers import id_to_token
from modeling.layers import token_to_id
from modeling.modules import graph_networks
from modeling.utils import box_ops
from modeling.utils import hyperparams
from modeling.utils import masked_ops

from models import model_base
from models import utils

from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields

from model_utils.scene_graph_evaluation import SceneGraphEvaluator


def parse_entity_and_attributes(entity_and_attributes):
  """Parses entity name and attributes from tensor `strings`.

  Args:
    entity_and_attributes: A [batch, max_n_entity] string tensor 
      mixing entity and attributes. E.g., 'suitcase:small,packed'.

  Returns:
    entity: Entity, a [batch, max_n_entity] string tensor.
    n_attribute: Number of attributes, a [batch, max_n_entity] int tensor.
    attributes: Attributes, a [batch, max_n_entity, max_n_attribute] string tensor.
  """
  batch = entity_and_attributes.shape[0].value

  split_res = tf.sparse_tensor_to_dense(
      tf.strings.split(entity_and_attributes, sep=':', maxsplit=1), '')
  entity, attributes = split_res[:, :, 0], split_res[:, :, 1]

  # Attributes.
  attributes = tf.sparse_tensor_to_dense(
      tf.strings.split(attributes, sep=',', maxsplit=-1), '')
  n_attribute = tf.reduce_sum(tf.cast(tf.not_equal(attributes, ''), tf.int32),
                              -1)

  # Set shape.
  entity.set_shape([batch, None])
  n_attribute.set_shape([batch, None])
  attributes.set_shape([batch, None, None])

  return entity, n_attribute, attributes


def compute_entity_repr_with_attributes(entity, per_entity_n_attribute,
                                        per_entity_attributes):
  """Computes node embeddings.

  Args:
    entity: A [batch, max_n_entity, dims] string tensor.
    per_entity_n_attribute: A [batch, max_n_entity] int tensor.
    per_entity_attributes: A [batch, max_n_entity, max_n_attribute, dims] tensor.

  Returns:
    entity_with_attributes: A [batch, max_n_entity, dims] string tensor.
  """
  max_n_attribute = tf.shape(per_entity_attributes)[2]
  attribute_masks = tf.sequence_mask(per_entity_n_attribute,
                                     max_n_attribute,
                                     dtype=tf.float32)
  # Sum up the representations.
  attr_repr = masked_ops.masked_sum_nd(per_entity_attributes,
                                       attribute_masks,
                                       dim=2)
  entity_with_attributes = tf.squeeze(attr_repr, 2) + entity
  return entity_with_attributes


class Cap2SGGrounding(model_base.ModelBase):
  """Cap2SGGrounding model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the Cap2SGGrounding instance. """
    super(Cap2SGGrounding, self).__init__(options, is_training)

    if not isinstance(options, model_pb2.Cap2SGGrounding):
      raise ValueError('Options has to be an Cap2SGGrounding proto.')

    # Initialize the arg_scope for FC layers.
    self.arg_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                      is_training)

    # Initialize vocabulary.
    token2id = {'OOV': 0}
    with open(options.vocabulary_file, 'r') as f:
      for i, line in enumerate(f):
        token, freq = line.strip('\n').split('\t')
        if int(freq) < options.minimum_frequency:
          break
        token2id[token] = 1 + i  # ZERO is reserved for OOV.
    self.vocab_size = len(token2id)
    self.token2id = token_to_id.TokenToIdLayer(token2id, oov_id=0)

    # Embeddings.
    self.embeddings = tf.get_variable(
        'embeddings',
        shape=[len(token2id), options.embedding_dims],
        regularizer=None,
        trainable=True)

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
        - `id`: A [batch] int64 tensor.
        - `image/n_proposal`: A [batch] int32 tensor.
        - `image/proposal`: A [batch, max_n_proposal, 4] float tensor.
        - `image/proposal/feature`: A [batch, max_proposal, feature_dims] float tensor.
        - `caption_graph/caption`: A [batch] string tensor.
        - `caption_graph/n_node`: A [batch] int tensor.
        - `caption_graph/n_edge`: A [batch] int tensor.
        - `caption_graph/nodes`: A [batch, max_n_node] string tensor.
        - `caption_graph/edges`: A [batch, max_n_edge] string tensor.
        - `caption_graph/senders`: A [batch, max_n_edge] float tensor.
        - `caption_graph/receivers`: A [batch, max_n_edge] float tensor.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    # Extract proposal features using FC layers.
    n_proposal, proposals, proposal_features = (
        inputs['image/n_proposal'], inputs['image/proposal'],
        inputs['image/proposal/feature'])
    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]

    attention_head, entity_head, attribute_head = [
        tf.layers.Dense(self.options.embedding_dims,
                        activation=None,
                        use_bias=True)(proposal_features) for i in range(3)
    ]

    # Extract linguistic information, e.g., nodes=`suitcase:small,packed`.
    n_node, n_edge, nodes, edges, senders, receivers = (
        inputs['caption_graph/n_node'], inputs['caption_graph/n_edge'],
        inputs['caption_graph/nodes'], inputs['caption_graph/edges'],
        inputs['caption_graph/senders'], inputs['caption_graph/receivers'])
    max_n_node, max_n_edge = tf.shape(nodes)[1], tf.shape(edges)[1]

    # Parse entity and attributes from strings such as 'suitcase:small,packed':
    (entity, per_ent_n_att, per_ent_atts) = parse_entity_and_attributes(nodes)

    # Embedding lookup, compute entity representations with attributes.
    embed_fn = lambda x: tf.nn.embedding_lookup(
        self.embeddings, x, max_norm=None)
    entity_ids, per_ent_atts_ids, edges_ids = (self.token2id(entity),
                                               self.token2id(per_ent_atts),
                                               self.token2id(edges))
    entity_embs, per_ent_atts_embs, edges_embs = (embed_fn(entity_ids),
                                                  embed_fn(per_ent_atts_ids),
                                                  embed_fn(edges_ids))
    node_embs = compute_entity_repr_with_attributes(entity_embs, per_ent_n_att,
                                                    per_ent_atts_embs)

    # Compute attention along the visual proposal axis.
    #   attention = [batch, max_n_node, max_n_proposal].
    proposal_masks = tf.expand_dims(
        tf.sequence_mask(n_proposal, max_n_proposal, dtype=tf.float32), 1)
    attention = tf.matmul(node_embs, attention_head, transpose_b=True)
    attention = masked_ops.masked_softmax(attention, proposal_masks, dim=2)

    # Treat top-1 box as the instance box.
    batch_indices = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                    [batch, max_n_node])
    box_indices = tf.math.argmax(attention, axis=2, output_type=tf.int32)
    box_indices = tf.stack([batch_indices, box_indices], -1)
    predictions = {
        'grounding/entity_boxes': tf.gather_nd(proposals, box_indices),
        'grounding/entity_scores': tf.reduce_max(attention, 2),
    }

    # Compute the classification logits from the entity and attribute heads.
    #   entity_labels/attribute_labels = [batch, max_n_node, vocab_size].
    entity_labels = tf.one_hot(entity_ids, depth=self.vocab_size)
    attribute_labels = tf.reduce_max(
        tf.one_hot(per_ent_atts_ids, depth=self.vocab_size), 2)

    for cls_name, cls_head, cls_labels in [
        ('grounding/entity', entity_head, entity_labels),
        ('grounding/attribute', attribute_head, attribute_labels)
    ]:
      # Attention weighting, cls_repr = [batch, max_n_node, dims].
      cls_repr = tf.matmul(attention, cls_head)

      # Convert representation to classification logits.
      cls_logits = tf.einsum('bnd,vd->bnv', cls_repr, self.embeddings)
      predictions.update({
          cls_name + '/logits': cls_logits,
          cls_name + '/labels': cls_labels,
      })

    return predictions

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    # Compute entity node masks.
    n_node, nodes = (inputs['caption_graph/n_node'],
                     inputs['caption_graph/nodes'])
    max_n_node = tf.shape(nodes)[1]
    node_masks = tf.sequence_mask(n_node, max_n_node, dtype=tf.float32)

    loss_dict = {}
    for cls_name in ['grounding/entity', 'grounding/attribute']:
      cls_logits = predictions[cls_name + '/logits']
      cls_labels = predictions[cls_name + '/labels']
      per_entity_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=cls_labels, logits=cls_logits)
      per_example_losses = masked_ops.masked_sum_nd(per_entity_losses,
                                                    mask=node_masks,
                                                    dim=1)
      losses = tf.div(tf.reduce_sum(per_example_losses, [0, 1]),
                      1e-6 + tf.cast(tf.reduce_sum(n_node), tf.float32))
      loss_dict.update({cls_name + '/loss': tf.reduce_sum(losses)})
    return loss_dict

  def build_metrics(self, inputs, predictions, **kwargs):
    """Computes evaluation metrics.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    metric_dict = {}
    const_metric = tf.keras.metrics.Mean()
    const_metric.update_state(1.0)
    metric_dict['metrics/accuracy'] = const_metric
    return metric_dict
