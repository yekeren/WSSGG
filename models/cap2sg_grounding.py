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

from absl import logging

import numpy as np
import tensorflow as tf

import tf_slim as slim

from protos import model_pb2

from models import model_base
from modeling.utils import masked_ops
from models.cap2sg_data import DataTuple


def ground_entities(options, dt, is_training):
  """Grounds entities.

  Args:
    options: A Cap2SGGrounding proto.
    dt: A DataTuple object.

  Returns:
    dt.attention
    dt.entity_image_logits
    dt.attribute_image_logits
    dt.entity_proposal_id
    dt.entity_proposal_box
    dt.entity_proposal_score
  """
  if not isinstance(options, model_pb2.Cap2SGGrounding):
    raise ValueError('Options has to be a Cap2SGGrounding proto.')

  if not isinstance(dt, DataTuple):
    raise ValueError('Invalid DataTuple object.')

  hidden_size = dt.dims
  attention_mask = tf.expand_dims(dt.proposal_masks, 1)

  # Compute the attention, entity, and attribute heads.
  attention_head, entity_head, attribute_head = [
      tf.layers.Dense(dt.dims,
                      activation=None,
                      use_bias=True,
                      kernel_initializer='glorot_normal',
                      name=name)(dt.proposal_features)
      for name in ['attention_head', 'entity_head', 'attribute_head']
  ]

  # Image-level classification using an attention model.
  if is_training:
    random_mask = tf.greater(
        tf.random.uniform(tf.shape(attention_mask), minval=0, maxval=1.0),
        1.0 - options.midn_attention_dropout_keep_prob)
    attention_mask = tf.multiply(attention_mask,
                                 tf.cast(random_mask, tf.float32))
  dt.attention = _compute_attention(
      tf.add(
          dt.refined_entity_embs
          if dt.refined_entity_embs is not None else dt.entity_embs,
          _compute_attribute_embeddings(dt.per_ent_n_att, dt.per_ent_att_embs)),
      attention_head, attention_mask)
  dt.entity_image_logits = _apply_attention(dt.attention, entity_head,
                                            dt.embeddings, dt.bias_entity)
  dt.attribute_image_logits = _apply_attention(dt.attention, attribute_head,
                                               dt.embeddings, dt.bias_attribute)

  # Multiply two branches.
  #   dt.attention: [batch, max_n_node, max_n_proposal]
  #   class_scores: [batch, vocab_size, max_n_proposal]
  dummy_attention = dt.attention
  if options.two_branches:
    class_scores = tf.nn.bias_add(
        tf.matmul(entity_head, dt.embeddings, transpose_b=True), dt.bias_entity)
    class_scores = tf.transpose(tf.nn.softmax(class_scores), [0, 2, 1])
    dummy_attention = tf.multiply(
        dt.attention,
        tf.gather_nd(class_scores, indices=_get_full_indices(dt.entity_ids)))

  # Set the grounding results.
  dt.grounding.entity_proposal_id = tf.math.argmax(dummy_attention,
                                                   axis=2,
                                                   output_type=tf.int32)
  indices = tf.stack([
      tf.broadcast_to(tf.expand_dims(tf.range(dt.batch), 1),
                      [dt.batch, dt.max_n_entity]),
      dt.grounding.entity_proposal_id,
  ], -1)
  dt.grounding.entity_proposal_box = tf.gather_nd(dt.proposals, indices)
  dt.grounding.entity_proposal_feature = tf.gather_nd(dt.proposal_features,
                                                      indices)
  dt.grounding.entity_proposal_score = tf.reduce_max(dt.attention, 2)
  return dt


def _compute_attribute_embeddings(per_entity_n_attribute,
                                  per_entity_attributes):
  """Computes node embeddings.

  Args:
    per_entity_n_attribute: A [batch, max_n_entity] int tensor.
    per_entity_attributes: A [batch, max_n_entity, max_per_entity_n_attribute, dims] tensor.

  Returns:
    entity_with_attributes: A [batch, max_n_entity, dims] string tensor.
  """
  max_per_entity_n_attribute = tf.shape(per_entity_attributes)[2]
  attribute_masks = tf.sequence_mask(per_entity_n_attribute,
                                     max_per_entity_n_attribute,
                                     dtype=tf.float32)
  # Sum up the representations.
  attr_repr = masked_ops.masked_sum_nd(per_entity_attributes,
                                       attribute_masks,
                                       dim=2)
  return tf.squeeze(attr_repr, 2)


def _compute_attention(class_embs, attention_head, attention_mask):
  """Predicts attention score.

  Assuming attention model SOFTMAX(Q K) V.
  class_embs, attention_head are analogous to Q and K.
  This function returns SOFTMAX(Q K)

  Args:
    class_embs: A [batch, max_n_node, dims] float tensor.
    attention_head: A [batch, max_n_proposal, dims] float tensor.
    attention_mask: A [batch, 1, max_n_proposal] float tensor.

  Returns:
    attention_score: A [batch, max_n_node, max_n_proposal] float tensor.
  """
  attention_logits = tf.matmul(class_embs, attention_head, transpose_b=True)
  return masked_ops.masked_softmax(attention_logits, attention_mask, dim=2)


def _apply_attention(attention_score, class_head, embeddings, bias):
  """Applies attention_score for classification.

  Assuming attention model SOFTMAX(Q K) V.
  class_head is analogous to V.
  This function based on weighted V, predict the label.

  Args:
    attention_score: A [batch, max_n_node, max_n_proposal] float tensor.
    class_head: A [batch, max_n_proposal, dims] float tensor.
    embeddings: A [vocab_size, dims] float tensor.
    bias: A [vocab_size] float tensor.
  """
  # Compute image-level representation: SOFTMAX(Q K) V.
  #   class_repr shape = [batch, max_n_node, dims].
  class_repr = tf.matmul(attention_score, class_head)

  # Compute image-level classification score.
  #   class_logits shape = [batch, max_n_node, vocab_size].
  class_logits = tf.einsum('bnd,vd->bnv', class_repr, embeddings)
  return tf.nn.bias_add(class_logits, bias)


def _get_full_indices(index):
  """Gets full indices from a single index.

  Args:
    index: A single index, a [batch, max_n_elem] int tensor.

  Returns:
    indices: Full indices with batch dimension added.
  """
  batch, max_n_elem = index.shape[0].value, index.shape[1].value
  if max_n_elem is None:
    max_n_elem = tf.shape(index)[1]

  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_elem])
  return tf.stack([batch_index, index], -1)
