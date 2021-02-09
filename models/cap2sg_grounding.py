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

import numpy as np
import tensorflow as tf

import tf_slim as slim

from protos import model_pb2

from modeling.layers import id_to_token
from modeling.layers import token_to_id
from modeling.utils import masked_ops

from models import model_base

from bert.modeling import transformer_model
import tensorflow.keras.backend as K


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


def compute_attribute_embeddings(per_entity_n_attribute, per_entity_attributes):
  """Computes node embeddings.

  Args:
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
  return tf.squeeze(attr_repr, 2)


def compute_attention(class_embs, attention_head, attention_mask):
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


def apply_attention(attention_score, class_head, embeddings, bias):
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


def sigmoid_focal_crossentropy(y_true,
                               y_pred,
                               alpha=0.25,
                               gamma=2.0,
                               from_logits=True):
  # Get the cross_entropy for each entry
  ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

  # If logits are provided then convert the predictions into probabilities
  if from_logits:
    pred_prob = tf.sigmoid(y_pred)
  else:
    pred_prob = y_pred

  p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
  alpha_factor = 1.0
  modulating_factor = 1.0

  if alpha:
    alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

  if gamma:
    gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
    modulating_factor = tf.pow((1.0 - p_t), gamma)

  # compute the final loss and return
  return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


def load_glove(glove_vocabulary_file, glove_embedding_file):
  """Loads GloVe embedding vectors.

  Args:
    glove_vocabulary_file: GloVe vocabulary file.
    glove_embedding_file: GloVe word embedding file.

  Returns:
    glove_dict: GloVe embeddings. A dict keyed by tokens.
  """
  glove_vectors = np.load(glove_embedding_file).astype(np.float32)
  with tf.gfile.GFile(glove_vocabulary_file, 'r') as f:
    glove_tokens = [x.strip('\n') for x in f]
  return dict((k, v) for k, v in zip(glove_tokens, glove_vectors))


def initialize_from_glove(glove_dict, token2id, embedding_dims):
  """Initializes token embeddings from GloVe.

  Args:
    glove_dict: GloVe embeddings. A dict keyed by tokens.
    token2id: A dict mapping from multi-words tokens to id.
    embedding_dims: embedding dimensions.

  Returns:
    embeddings: A [len(token2id), embedding_dims] np.float32 array.
  """

  embeddings = np.zeros((len(token2id), embedding_dims), dtype=np.float32)

  for multi_word_token, token_id in token2id.items():
    for word in multi_word_token.split(' '):
      if word in glove_dict:
        embeddings[token_id] += glove_dict[word]
  return embeddings


class Cap2SGGrounding(model_base.ModelBase):
  """Cap2SGGrounding model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the Cap2SGGrounding instance. """
    if not isinstance(options, model_pb2.Cap2SGGrounding):
      raise ValueError('Options has to be an Cap2SGGrounding proto.')
    super(Cap2SGGrounding, self).__init__(options, is_training)

    # Load GloVe embeddings.
    glove_dict = load_glove(options.glove_vocabulary_file,
                            options.glove_embedding_file)

    # Initialize vocabulary.
    token2id, id2token = {'OOV': 0}, {0: 'OOV'}
    id_offset = 1  # ZERO is reserved for OOV.
    with open(options.vocabulary_file, 'r') as f:
      for line in f:
        token, freq = line.strip('\n').split('\t')
        if int(freq) < options.minimum_frequency:
          break
        if any(word in glove_dict for word in token.split(' ')):
          id2token[id_offset] = token
          token2id[token] = id_offset
          id_offset += 1
    self.vocab_size = len(token2id)
    self.token2id = token_to_id.TokenToIdLayer(token2id, oov_id=0)
    self.id2token = id_to_token.IdToTokenLayer(id2token, oov='OOV')

    # Create word embeddings.
    self.embeddings = tf.get_variable('embeddings',
                                      initializer=initialize_from_glove(
                                          glove_dict, token2id,
                                          options.embedding_dims),
                                      trainable=options.embedding_trainable)

    # Create the bias tensor.
    if model_pb2.BIAS_MODE_ZERO == options.bias_mode:
      zeros = tf.zeros(shape=[len(token2id)], dtype=tf.float32)
      self.entity_bias = self.attribute_bias = zeros
    elif model_pb2.BIAS_MODE_TRADITION == options.bias_mode:
      self.entity_bias = tf.get_variable('entity_bias',
                                         initializer=tf.zeros_initializer(),
                                         shape=[len(token2id)],
                                         trainable=True)
      self.attribute_bias = tf.get_variable('attribute_bias',
                                            initializer=tf.zeros_initializer(),
                                            shape=[len(token2id)],
                                            trainable=True)
    elif model_pb2.BIAS_MODE_TRAIN_FROM_EMBEDDING == options.bias_mode:
      self.entity_bias, self.attribute_bias = [
          tf.squeeze(
              tf.layers.Dense(1, activation=None, use_bias=True,
                              name=name)(self.embeddings), -1)
          for name in ['entity_bias', 'attribute_bias']
      ]

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

    # Linear projection.
    entity_head, attribute_head = [
        tf.layers.Dense(self.options.embedding_dims,
                        activation=tf.nn.leaky_relu,
                        use_bias=True,
                        name=name)(proposal_features)
        for name in ['entity_head', 'attribute_head']
    ]

    # Compute the shared attention head.
    hidden_size = self.options.embedding_dims
    attention_input = tf.layers.Dense(hidden_size,
                                      activation=None,
                                      use_bias=True,
                                      name='bert_input')(proposal_features)
    attention_head = transformer_model(
        attention_input,
        hidden_size=hidden_size,
        num_hidden_layers=self.options.contextualization_config.
        num_hidden_layers,
        num_attention_heads=self.options.contextualization_config.
        num_attention_heads,
        intermediate_size=self.options.contextualization_config.
        intermediate_size)

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
    attribute_embs = compute_attribute_embeddings(per_ent_n_att,
                                                  per_ent_atts_embs)

    # Compute attention_mask, shape = [batch, 1, max_n_proposal].
    attention_mask = tf.expand_dims(
        tf.sequence_mask(n_proposal, max_n_proposal, dtype=tf.float32), 1)

    # One-hot encoding of both entity and attribute labels.
    entity_labels = tf.one_hot(entity_ids, depth=self.vocab_size)
    attribute_labels = tf.reduce_max(
        tf.one_hot(per_ent_atts_ids, depth=self.vocab_size), 2)

    # Entity attention model.
    attention = compute_attention(entity_embs + attribute_embs, attention_head,
                                  attention_mask)
    entity_logits = apply_attention(attention, entity_head, self.embeddings,
                                    self.entity_bias)
    attribute_logits = apply_attention(attention, attribute_head,
                                       self.embeddings, self.attribute_bias)

    # Treat top-1 box as the instance box.
    index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_n_node])
    index_entity_box = tf.math.argmax(attention, axis=2, output_type=tf.int32)
    index_entity_full = tf.stack([index_batch, index_entity_box], -1)

    predictions = {
        'grounding/entity/logits':
            entity_logits[:, :, 1:],
        'grounding/entity/labels':
            entity_labels[:, :, 1:],
        'grounding/entity/id':
            entity_ids,
        'grounding/entity/attribute_id':
            per_ent_atts_ids,
        'grounding/entity/proposal_id':
            index_entity_box,
        'grounding/entity/proposal_box':
            tf.gather_nd(proposals, index_entity_full),
        'grounding/entity/proposal_score':
            tf.reduce_max(attention, 2),
        'grounding/attribute/logits':
            attribute_logits[:, :, 1:],
        'grounding/attribute/labels':
            attribute_labels[:, :, 1:],
    }
    return predictions

  def _compute_cross_entropy_loss(self, n_node, logits, labels):
    """Computes cross-entropy loss.

    Args:
      n_node: A [batch] int tensor denoting the entity nodes in the example.
      logits: Entity logits, A [batch, max_n_node, vocab_size - 1] float tensor.
      labels: Entity labels, A [batch, max_n_node, vocab_size - 1] float tensor.

    Returns:
      A scalar loss tensor.
    """
    # Create mask to drop unlabeled example (e.g., OOV entity name, no attributes).
    #   label_mask shape = [batch, max_n_node].
    node_mask = tf.sequence_mask(n_node, tf.shape(logits)[1], dtype=tf.float32)
    label_mask = tf.cast(tf.greater(tf.reduce_sum(labels, -1), 0), tf.float32)
    label_mask = tf.multiply(label_mask, node_mask)

    # Normalize label and apply softmax cross-entropy loss.
    labels = tf.div(labels, 1e-6 + tf.reduce_sum(labels, -1, keepdims=True))
    per_entity_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                logits=logits)
    per_example_losses = masked_ops.masked_sum(per_entity_losses,
                                               mask=label_mask,
                                               dim=1)
    loss = tf.div(tf.reduce_sum(per_example_losses),
                  1e-6 + tf.reduce_sum(label_mask))
    return loss

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    n_node = inputs['caption_graph/n_node']

    loss_dict = {}
    for cls_name in ['grounding/entity', 'grounding/attribute']:
      cls_logits = predictions[cls_name + '/logits']
      cls_labels = predictions[cls_name + '/labels']

      loss_dict.update({
          cls_name + '/loss':
              tf.multiply(
                  self.options.loss_weight,
                  self._compute_cross_entropy_loss(n_node, cls_logits,
                                                   cls_labels))
      })
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
