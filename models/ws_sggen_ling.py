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
  - Our previous work of Ye et al. 2019 (Cap2det),
  - Online instance classifier refinement, proposed by Tang et al. 2017 (OICR),
Data are preprocessed by:
  - Zareian et al. 2020 (VSPNet)
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


class WSSGGenLing(model_base.ModelBase):
  """WSSGGenLing model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the WSSGGenLing instance. """
    super(WSSGGenLing, self).__init__(options, is_training)

    if not isinstance(options, model_pb2.WSSGGenLing):
      raise ValueError('Options has to be an WSSGGenLing proto.')

    # Load token2id mapping, convert 1-based index to 0-based index.
    with tf.io.gfile.GFile(options.token_to_id_meta_file, 'r') as fid:
      meta = json.load(fid)
      (entity2id, predicate2id) = (meta['label_to_idx'],
                                   meta['predicate_to_idx'])

    categories = []
    id2entity = {}
    id2predicate = {}
    for key in entity2id.keys():
      entity2id[key] -= 1
      id2entity[entity2id[key]] = key
      categories.append({'id': entity2id[key], 'name': key})
    self.entity_categories = categories

    for key in predicate2id.keys():
      predicate2id[key] -= 1
      id2predicate[predicate2id[key]] = key

    self.entity2id = token_to_id.TokenToIdLayer(entity2id, oov_id=0)
    self.predicate2id = token_to_id.TokenToIdLayer(predicate2id, oov_id=0)
    self.id2entity = id_to_token.IdToTokenLayer(id2entity, oov='OOV')
    self.id2predicate = id_to_token.IdToTokenLayer(id2predicate, oov='OOV')

    self.n_entity = len(entity2id)
    self.n_predicate = len(predicate2id)

    logging.info('#Entity=%s, #Predicate=%s', self.n_entity, self.n_predicate)

    # Load pre-trained GloVe word embeddings.
    w2v_dict, _, _ = utils.read_word_embeddings(
        self.options.glove_vocab_file, self.options.glove_embedding_file)
    self.entity_emb_weights = utils.lookup_word_embeddings(
        w2v_dict, id2entity, sorted(id2entity.keys()))
    self.predicate_emb_weights = utils.lookup_word_embeddings(
        w2v_dict, id2predicate, sorted(id2predicate.keys()))

    # Initialize the arg_scope for FC layers.
    self.arg_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                      is_training)

  def _refine_entity_scores_once(self, n_proposal, proposals, proposal_features,
                                 scope):
    """Refines scene graph entity score predictions.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, dims] float tensor.
      scope: Name of the variable scope.

    Returns:
      logits_entity_given_proposal: A [batch, max_n_proposal, 1 + n_entity] tensor.
    """
    with slim.arg_scope(self.arg_scope_fn()):
      logits_entity_given_proposal = slim.fully_connected(proposal_features,
                                                          num_outputs=1 +
                                                          self.n_entity,
                                                          activation_fn=None,
                                                          scope=scope)
    return logits_entity_given_proposal

  def _refine_relation_scores_once(self, n_proposal, proposals,
                                   proposal_features, scope):
    """Classifies relations given the proposal features.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.
      scope: Name of the variable scope.

    Returns:
      relation_logits: A [batch, max_n_proposal**2, 1 + n_predicate] float tensor.
    """
    with slim.arg_scope(self.arg_scope_fn()):
      with tf.variable_scope(scope):
        logits_predicate_given_subject = slim.fully_connected(
            proposal_features,
            num_outputs=1 + self.n_predicate,
            activation_fn=None,
            scope='attach_to_subject')
        logits_predicate_given_object = slim.fully_connected(
            proposal_features,
            num_outputs=1 + self.n_predicate,
            activation_fn=None,
            scope='attach_to_object')

    relation_logits = tf.minimum(
        tf.expand_dims(logits_predicate_given_subject, 2),
        tf.expand_dims(logits_predicate_given_object, 1))

    return tf.reshape(relation_logits,
                      [relation_logits.shape[0], -1, relation_logits.shape[-1]])

  def predict(self, inputs, **kwargs):
    """Predicts the resulting tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
        - `id`: A [batch] int64 tensor.
        - `image/n_proposal`: A [batch] int32 tensor.
        - `image/proposal`: A [batch, max_n_proposal, 4] float tensor.
        - `image/proposal/feature`: A [batch, max_proposal, feature_dims] float tensor.
        - `scene_graph/n_triple`: A [batch] int32 tensor.
        - `scene_graph/predicate`: A [batch, max_n_triple] string tensor.
        - `scene_graph/subject`: A [batch, max_n_triple] string tensor.
        - `scene_graph/object`: A [batch, max_n_triple] string tensor.
        - `scene_graph/subject/box`: A [batch, max_n_triple, 4] float tensor.
        - `scene_graph/object/box`: A [batch, max_n_triple, 4] float tensor.

    Returns:
      predictions: A dictionary of prediction tensors keyed by name.
    """
    predictions = {}

    # Checking the refinement score normalizer.
    score_normalizer = (tf.nn.softmax
                        if self.options.refine_use_softmax else tf.nn.sigmoid)

    # Extract proposal features.
    # - proposals = [batch, max_n_proposal, 4].
    # - proposal_features = [batch, max_n_proposal, dims].
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]

    with slim.arg_scope(self.arg_scope_fn()):
      predictions['features/shared_hidden'] = slim.fully_connected(
          proposal_features,
          num_outputs=300,
          activation_fn=None,
          scope='shared')
      predictions['image/proposal/feature'] = predictions[
          'features/shared_hidden']

    for i in range(1, 1 + self.options.n_refine_iteration):
      # Entity Score Refinement (ESR).
      # - `entity_score`=[batch, max_n_proposal, 1 + n_entity].
      entity_scope = 'refine_%i/entity' % i
      entity_score = self._refine_entity_scores_once(n_proposal, proposals,
                                                     proposal_features,
                                                     entity_scope)

      # Relation Score Refinement (RSR).
      # - `relation_score`=[batch, max_n_proposal**2, 1 + n_predicate].
      relation_scope = 'refine_%i/relation' % i
      relation_score = self._refine_relation_scores_once(
          n_proposal, proposals, proposal_features, relation_scope)

      # Update results.
      predictions.update({
          'refine@%i/entity_logits' % i: entity_score,
          'refine@%i/entity_probas' % i: score_normalizer(entity_score),
          'refine@%i/relation_logits' % i: relation_score,
          'refine@%i/relation_probas' % i: score_normalizer(relation_score),
      })

    # Post-process the predictions.
    if not self.is_training:
      graph_proposal_scores = predictions['refine@%i/entity_probas' %
                                          self.options.n_refine_iteration]
      graph_relation_scores = predictions['refine@%i/relation_probas' %
                                          self.options.n_refine_iteration]
      graph_proposal_scores = graph_proposal_scores[:, :, 1:]
      graph_relation_scores = graph_relation_scores[:, :, 1:]

      # Search for the triple proposals.
      triple_proposal = GraphNMS(
          n_proposal=n_proposal,
          proposals=proposals,
          proposal_scores=graph_proposal_scores,
          relation_scores=tf.reshape(
              graph_relation_scores,
              [batch, max_n_proposal, max_n_proposal, self.n_predicate]),
          max_size_per_class=self.options.post_process.max_size_per_class,
          max_total_size=self.options.post_process.max_total_size,
          iou_thresh=self.options.post_process.iou_thresh,
          score_thresh=self.options.post_process.score_thresh,
          use_class_agnostic_nms=False,
          use_log_prob=self.options.use_log_prob)

      # Predict the pseudo boxes for visualization purpose.
      predictions.update({
          'object_detection/num_detections':
              triple_proposal.num_detections,
          'object_detection/detection_boxes':
              triple_proposal.detection_boxes,
          'object_detection/detection_scores':
              triple_proposal.detection_scores,
          'object_detection/detection_classes':
              triple_proposal.detection_classes,
          'triple_proposal/n_triple':
              triple_proposal.n_triple,
          'triple_proposal/subject':
              self.id2entity(triple_proposal.subject_id),
          'triple_proposal/subject/score':
              triple_proposal.subject_score,
          'triple_proposal/subject/box':
              triple_proposal.get_subject_box(proposals),
          'triple_proposal/subject/box_index':
              triple_proposal.subject_proposal_index,
          'triple_proposal/object':
              self.id2entity(triple_proposal.object_id),
          'triple_proposal/object/score':
              triple_proposal.object_score,
          'triple_proposal/object/box':
              triple_proposal.get_object_box(proposals),
          'triple_proposal/object/box_index':
              triple_proposal.object_proposal_index,
          'triple_proposal/predicate':
              self.id2predicate(triple_proposal.predicate_id),
          'triple_proposal/predicate/score':
              triple_proposal.predicate_score,
      })

    return predictions

  def get_pseudo_triplets_from_inputs(self, inputs):
    """Extracts text triplets from inputs dict.

    WSSGGenLing model shall use `scene_pseudo_graph`.

    Args:
      inputs: A dictionary of input tensors keyed by names.

    Returns:
      n_triple: A [batch] tensor denoting the triples in each image.
      subject_ids: A [batch, max_n_triple] int tensor.
      predicate_ids: A [batch, max_n_triple] int tensor.
      object_ids: A [batch, max_n_triple] int tensor.
    """
    n_node = inputs['scene_pseudo_graph/n_node']
    n_edge = inputs['scene_pseudo_graph/n_edge']
    nodes = inputs['scene_pseudo_graph/nodes']
    edges = inputs['scene_pseudo_graph/edges']
    senders = inputs['scene_pseudo_graph/senders']
    receivers = inputs['scene_pseudo_graph/receivers']

    batch = n_edge.shape[0].value
    max_n_edge = tf.shape(edges)[1]

    # Sender/Receiver indices.
    batch_indices = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                    [batch, max_n_edge])
    sender_indices = tf.stack([batch_indices, senders], -1)
    receiver_indices = tf.stack([batch_indices, receivers], -1)

    edge_mask = tf.sequence_mask(n_edge, maxlen=max_n_edge)

    subject_label = tf.gather_nd(nodes, sender_indices)
    subject_label = tf.where(edge_mask, subject_label,
                             tf.zeros_like(subject_label, dtype=tf.string))
    object_label = tf.gather_nd(nodes, receiver_indices)
    object_label = tf.where(edge_mask, object_label,
                            tf.zeros_like(object_label, dtype=tf.string))

    # Returns image-level labels.
    assert_cond = tf.reduce_all([
        tf.reduce_all(tf.equal(n_edge, inputs['scene_graph/n_triple'])),
        tf.reduce_all(tf.equal(subject_label, inputs['scene_graph/subject'])),
        tf.reduce_all(tf.equal(edges, inputs['scene_graph/predicate'])),
        tf.reduce_all(tf.equal(object_label, inputs['scene_graph/object']))
    ])
    assert_op = tf.Assert(assert_cond,
                          ['Pseudo graph should be identical to triplets.'])
    with tf.control_dependencies([assert_op]):
      subject_ids = self.entity2id(subject_label)
      object_ids = self.entity2id(object_label)
      predicate_ids = self.predicate2id(edges)
    return n_edge, subject_ids, predicate_ids, object_ids

  def _multiple_entity_detection(self, n_proposal, proposals, proposal_features,
                                 n_node, node_ids, node_embs):
    """Detects multiple entities given the ground-truth.

    The function is called in the `build_losses` because `node_ids`,
    `node_embs` are from graph annotations.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, dims] float tensor.
      n_node: A [batch] int tensor.
      node_ids: A [batch, max_n_node] int tensor.
      node_embs: A [batch, max_n_node, feature_dims] float tensor.

    Returns:
      detection_score: Initial detection score, shape=[batch, max_n_proposal, n_entity]
      med_loss: The loss term to be optimized, to get the initial detection score.
    """
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)
    max_n_node = tf.shape(node_ids)[1]
    node_masks = tf.sequence_mask(n_node, maxlen=max_n_node, dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      proposal_features_projected = slim.fully_connected(
          proposal_features,
          num_outputs=node_embs.shape[-1].value,
          activation_fn=None,
          scope='MED/detection')

    # MED network: detection branch.
    # - attn_proposal_given_node = [batch, max_n_proposal, max_n_node].
    logits_proposal_given_node = tf.multiply(
        self.options.attn_scale,
        tf.matmul(proposal_features_projected, node_embs, transpose_b=True))

    attn_proposal_given_node = masked_ops.masked_softmax(
        logits_proposal_given_node,
        mask=tf.expand_dims(proposal_masks, -1),
        dim=1)
    attn_proposal_given_node = slim.dropout(attn_proposal_given_node,
                                            self.options.attn_dropout_keep_prob,
                                            is_training=self.is_training)

    # MED network: classification branch.
    # - logits_entity_given_proposal = [batch, max_n_proposal, n_entity].
    with slim.arg_scope(self.arg_scope_fn()):
      logits_entity_given_proposal = slim.fully_connected(
          proposal_features,
          num_outputs=self.n_entity,
          activation_fn=None,
          scope='MED/classification')

    # Apply attention weighing to get image-level classification for each node.
    # - logits_entity_given_node = [batch, max_n_node, n_entity].
    logits_entity_given_node = tf.matmul(attn_proposal_given_node,
                                         logits_entity_given_proposal,
                                         transpose_a=True)

    # Compute the MED loss.
    #   losses = [batch, max_n_node].
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=node_ids, logits=logits_entity_given_node)
    losses = masked_ops.masked_avg(losses, node_masks, dim=1)
    loss = tf.reduce_mean(losses)

    # Max-pooling.
    # This process use max-pooling to aggregate attention, to deal with multiple
    # instances of the same class. E.g., two person nodes in the graph.
    #   node_id_onehot / attn = [batch, max_n_proposal, max_n_node, n_entity]
    #   attn_maxpool = [batch, max_n_propoal, n_entity]
    attn_maxpool = tf.expand_dims(attn_proposal_given_node, -1)
    node_id_onehot = tf.one_hot(node_ids, depth=self.n_entity, dtype=tf.float32)
    node_id_onehot = tf.expand_dims(node_id_onehot, 1)
    attn_maxpool = tf.multiply(node_id_onehot, attn_maxpool)

    node_masks = tf.expand_dims(node_masks, 1)
    node_masks = tf.expand_dims(node_masks, -1)
    attn_maxpool = masked_ops.masked_maximum(attn_maxpool,
                                             mask=node_masks,
                                             dim=2)
    attn_maxpool = tf.squeeze(attn_maxpool, 2)

    # Detection score.
    detection_score = tf.multiply(attn_maxpool,
                                  tf.nn.softmax(logits_entity_given_proposal))

    return detection_score, loss

  def _compute_multiple_entity_detection_losses(self, n_triple, subject_labels,
                                                subject_logits, object_labels,
                                                object_logits):
    """Computes MED and MRD losses.

    Args:
      n_triple: A [batch] int tensor.
      subject_labels: A [batch, max_n_triple] int tensor.
      subject_logits: A [batch, max_n_triple, n_entity] float tensor.
      object_labels: A [batch, max_n_triple] int tensor.
      object_logits: A [batch, max_n_triple, n_entity] float tensor.

    Returns:
      loss_dict involving the following fields:
        - `losses/med_subject_loss`: MED loss for subject.
        - `losses/med_object_loss`: MED loss for object.
    """
    max_n_triple = tf.shape(subject_labels)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    loss_dict = {}
    for name, labels, logits in [
        ('losses/med_subject_loss', subject_labels, subject_logits),
        ('losses/med_object_loss', object_labels, object_logits),
    ]:
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=logits)
      loss_dict[
          name] = self.options.multiple_entity_detection_loss_weight * tf.reduce_mean(
              masked_ops.masked_avg(losses, triple_mask, dim=1))
    return loss_dict

  def _compute_proposal_refinement_losses(self, n_proposal, labels, logits):
    """Computes SGR losses.

    Args:
      n_proposal: A [batch] int tensor.
      labels: Pseudo instance labels, a [batch, max_n_proposal, 1 + n_entity] 
        float tensor involving background indicator.
      logits: A [batch, max_n_proposal, 1 + n_entity] float tensor.

    Returns:
      loss_dict involving the following fields:
        - `losses/SGR/detection_loss`: SGR detection loss.
    """
    max_n_proposal = tf.shape(labels)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
    losses = masked_ops.masked_avg(losses, proposal_masks, dim=1)
    return tf.reduce_mean(losses)

  def _compute_relation_refinement_losses(self, n_proposal, max_n_proposal,
                                          labels, logits):
    """Computes relation refinement losses.

    Args:
      n_proposal: A [batch] int tensor.
      max_n_proposal: A scalar int tensor.
      labels: Pseudo instance labels, a [batch, max_n_proposal * max_n_proposal, 1 + n_predicate] 
        float tensor involving background indicator.
      logits: A [batch, max_n_proposal * max_n_proposal, 1 + n_entity] float tensor.
    """
    batch = n_proposal.shape[0].value
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)
    relation_masks = tf.multiply(tf.expand_dims(proposal_masks, 1),
                                 tf.expand_dims(proposal_masks, 2))
    relation_masks = tf.reshape(relation_masks,
                                [batch, max_n_proposal * max_n_proposal])

    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
    losses = masked_ops.masked_avg(losses, relation_masks, dim=1)
    return tf.reduce_mean(losses)

  def _compute_pseudo_instance_labels(self,
                                      n_entity,
                                      n_proposal,
                                      proposals,
                                      subject_index,
                                      subject_proposal_index,
                                      object_index,
                                      object_proposal_index,
                                      iou_threshold_to_propogate=0.5):
    """Computes SGR losses.

    Args:
      n_entity: Total number of entities.
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      subject_index: A [batch, max_n_triple] int tensor.
      subject_proposal_index: A [batch, max_n_triple] int tensor.
      object_index: A [batch, max_n_triple] int tensor.
      object_proposal_index: A [batch, max_n_triple] int tensor.
      iou_threshold_to_propogate: IoU threshold for propogating annotations.

    Returns:
      Pseudo instance labels of shape [batch, max_n_proposal, 1 + n_entity].
    """
    instance_labels = []
    for entity_index, entity_proposal_index in [
        (subject_index, subject_proposal_index),
        (object_index, object_proposal_index)
    ]:
      instance_labels.append(
          utils.scatter_pseudo_entity_detection_labels(
              n_entity=n_entity,
              n_proposal=n_proposal,
              proposals=proposals,
              entity_index=1 + entity_index,  # Background.
              proposal_index=entity_proposal_index,
              iou_threshold=iou_threshold_to_propogate))

    instance_labels = utils.post_process_pseudo_detection_labels(
        tf.add_n(instance_labels), normalize=self.options.refine_use_softmax)
    return instance_labels

  def _compute_pseudo_relation_labels(self,
                                      n_predicate,
                                      n_proposal,
                                      proposals,
                                      predicate_index,
                                      subject_proposal_index,
                                      object_proposal_index,
                                      iou_threshold_to_propogate=0.5):
    """Computes SGR losses.

    Args:
      n_predicate: Total number of predicates.
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      predicate_index: A [batch, max_n_triple] int tensor.
      subject_proposal_index: A [batch, max_n_triple] int tensor.
      object_proposal_index: A [batch, max_n_triple] int tensor.
      iou_threshold_to_propogate: IoU threshold for propogating annotations.
    """
    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposals)[1]

    relation_labels = utils.scatter_pseudo_relation_detection_labels(
        n_predicate=n_predicate,
        n_proposal=n_proposal,
        proposals=proposals,
        predicate_index=1 + predicate_index,
        subject_proposal_index=subject_proposal_index,
        object_proposal_index=object_proposal_index,
        iou_threshold=iou_threshold_to_propogate)

    relation_labels_reshaped = tf.reshape(
        relation_labels,
        [batch, max_n_proposal * max_n_proposal, 1 + n_predicate])
    relation_labels_reshaped = utils.post_process_pseudo_detection_labels(
        relation_labels_reshaped, normalize=self.options.refine_use_softmax)
    return relation_labels_reshaped

  def _create_selection_indices(self, subject_ids, object_ids, predicate_ids):
    """Create indices to select predictions.

    Args:
      subject_ids: A [batch, max_n_triple] int tensor.
      object_ids: A [batch, max_n_triple] int tensor.
      predicate_ids: A [batch, max_n_triple] int tensor.

    Returns:
      subject_index: A [batch, max_n_triple, 2] int tensor.
      object_index: A [batch, max_n_triple, 2] int tensor.
      predicate_index: A [batch, max_n_triple, 2] int tensor.
    """
    batch = subject_ids.shape[0].value
    max_n_triple = tf.shape(subject_ids)[1]

    batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_n_triple])
    subject_index = tf.stack([batch_index, subject_ids], -1)
    object_index = tf.stack([batch_index, object_ids], -1)
    predicate_index = tf.stack([batch_index, predicate_ids], -1)

    return subject_index, object_index, predicate_index

  def _get_triplets_from_pseudo_graph(self, n_node, n_edge, nodes, node_embs,
                                      edges, senders, receivers, inputs):
    """Get triplets from pseudo graph.
    
      A helper function to check the compatibility of the pseudo graph and the
      triplets annotation. Note `n_edge == n_triple`.

    Args:
      n_node: A [batch] int tensor.
      n_edge: A [batch] int tensor.
      nodes: A [batch, max_n_node] string tensor.
      node_embs: A [batch, max_n_node, dims] float tensor.
      edges: A [batch, max_n_edge] string tensor.
      senders: A [batch, max_n_edge] int tensor.
      receivers: A [batch, max_n_edge] int tensor.
      inputs: The input dictionary used to check the compatibility.

    Returns:
      n_triple: A [batch] int tensor.
      subject_ids: A [batch, max_n_triple] int tensor.
      object_ids: A [batch, max_n_triple] int tensor.
      predicate_ids: A [batch, max_n_triple] int tensor.
    """

    def _get_gather_nd_indices(some_ids):
      batch = some_ids.shape[0].value
      max_n_edge = tf.shape(some_ids)[1]
      batch_indices = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                      [batch, max_n_edge])
      return tf.stack([batch_indices, some_ids], -1)

    def _gather_and_pad(n_edge, nodes, node_ids):
      max_n_edge = tf.shape(node_ids)[1]
      edge_mask = tf.sequence_mask(n_edge, maxlen=max_n_edge)
      name = tf.gather_nd(nodes, _get_gather_nd_indices(node_ids))
      return tf.where(edge_mask, name, tf.zeros_like(name, dtype=tf.string))

    subject_name = _gather_and_pad(n_edge, nodes, senders)
    object_name = _gather_and_pad(n_edge, nodes, receivers)
    assert_cond = tf.reduce_all([
        tf.reduce_all(tf.equal(n_edge, inputs['scene_graph/n_triple'])),
        tf.reduce_all(tf.equal(subject_name, inputs['scene_graph/subject'])),
        tf.reduce_all(tf.equal(object_name, inputs['scene_graph/object'])),
        tf.reduce_all(tf.equal(edges, inputs['scene_graph/predicate']))
    ])
    assert_op = tf.Assert(assert_cond,
                          [subject_name, inputs['scene_graph/subject']])
    with tf.control_dependencies([assert_op]):
      subject_ids = self.entity2id(subject_name)
      object_ids = self.entity2id(object_name)
      predicate_ids = self.predicate2id(edges)

    subject_embs = tf.gather_nd(node_embs, _get_gather_nd_indices(senders))
    object_embs = tf.gather_nd(node_embs, _get_gather_nd_indices(receivers))

    return n_edge, subject_ids, object_ids, predicate_ids, subject_embs, object_embs

  def _compute_initial_embeddings(self,
                                  tokens,
                                  initializer,
                                  scope,
                                  max_norm=None,
                                  trainable=False):
    """Computes the initial node embeddings.

    Args:
      tokens: Token ids of the nodes/edges, a [batch, max_n_token] int tensor.
      initializer: Initial value of the embedding weights.
      scope: Variable scope for the token embedding weights.
      max_norm: Maximum norm of the embedding weights.
      trainable: If true, set the embedding weights trainable.

    Returns:
      token_embeddings: A [batch, max_n_token, dims] float tensor.
    """
    # regularizer = hyperparams._build_slim_regularizer(
    #     self.options.fc_hyperparams.regularizer)
    regularizer = None

    with tf.variable_scope(scope):
      embedding_weights = tf.get_variable('embeddings',
                                          initializer=initializer,
                                          regularizer=None,
                                          trainable=trainable)
    embeddings = tf.nn.embedding_lookup(embedding_weights,
                                        tokens,
                                        max_norm=max_norm)
    return embeddings

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    loss_dict = {}

    # Parse proposals.
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = n_proposal.shape[0]
    max_n_proposal = tf.shape(proposals)[1]

    # Parse pseudo graph annotations.
    n_node = inputs['scene_pseudo_graph/n_node']
    n_edge = inputs['scene_pseudo_graph/n_edge']
    nodes = inputs['scene_pseudo_graph/nodes']
    edges = inputs['scene_pseudo_graph/edges']
    senders = inputs['scene_pseudo_graph/senders']
    receivers = inputs['scene_pseudo_graph/receivers']

    max_n_edge = tf.shape(edges)[1]

    (n_triple, subject_ids, predicate_ids,
     object_ids) = self.get_pseudo_triplets_from_inputs(inputs)

    # Using graph network to update node and edge embeddings.
    max_norm = None
    if self.options.HasField('graph_initial_embedding_max_norm'):
      max_norm = self.options.graph_initial_embedding_max_norm
    regularizer = hyperparams._build_slim_regularizer(
        self.options.fc_hyperparams.regularizer)

    graph_network = graph_networks.build_graph_network(
        self.options.text_graph_network, is_training=self.is_training)
    with slim.arg_scope(self.arg_scope_fn()):
      node_embs, edge_embs = graph_network.compute_graph_embeddings(
          n_node,
          n_edge,
          self._compute_initial_embeddings(
              self.entity2id(nodes),
              initializer=self.entity_emb_weights,
              scope='linguistic/node/embeddings',
              max_norm=max_norm,
              trainable=self.options.train_graph_initial_embedding),
          self._compute_initial_embeddings(
              self.predicate2id(edges),
              initializer=self.predicate_emb_weights,
              scope='linguistic/edge/embeddings',
              max_norm=max_norm,
              trainable=self.options.train_graph_initial_embedding),
          senders,
          receivers,
          regularizer=regularizer)

    # Multiple Entity Detection (MED).
    # - `initial_detection_scores`=[batch, max_n_proposal, n_entity].
    # - `logits_entity_given_entity`=[batch, max_n_node, n_entity].
    (initial_detection_scores,
     med_loss) = self._multiple_entity_detection(n_proposal,
                                                 proposals,
                                                 proposal_features,
                                                 n_node=n_triple,
                                                 node_ids=self.entity2id(nodes),
                                                 node_embs=node_embs)
    predictions.update({
        'refine@0/entity_probas': initial_detection_scores,
    })
    loss_dict.update({'losses/med_loss': med_loss})

    # Entity Refinement loss.
    # - `proposal_scores_0`=[batch, max_n_proposal, n_entity].
    # - `relation_scores_0`=[batch, max_n_proposal**2, 1 + n_predicate].
    (subject_index, object_index,
     predicate_index) = self._create_selection_indices(subject_ids, object_ids,
                                                       predicate_ids)

    max_n_triple = tf.shape(subject_ids)[1]
    proposal_scores_0 = initial_detection_scores
    relation_scores_0 = tf.zeros_like(
        predictions['refine@1/relation_probas'])[:, :, 1:]

    proposal_to_proposal_weight = self.options.joint_inferring_relation_weight
    for i in range(1, 1 + self.options.n_refine_iteration):
      # DP solution for the optimal subject/object boxe.
      mps = GraphMPS(
          n_triple=n_triple,
          n_proposal=n_proposal,
          subject_to_proposal=tf.gather_nd(
              tf.transpose(proposal_scores_0, [0, 2, 1]), subject_index),
          proposal_to_proposal=tf.reshape(
              tf.gather_nd(tf.transpose(relation_scores_0, [0, 2, 1]),
                           predicate_index),
              [batch, max_n_triple, max_n_proposal, max_n_proposal]),
          proposal_to_object=tf.gather_nd(
              tf.transpose(proposal_scores_0, [0, 2, 1]), object_index),
          proposal_to_proposal_weight=proposal_to_proposal_weight,
          use_log_prob=self.options.use_log_prob)

      predictions.update({
          'pseudo/subject/box': mps.get_subject_box(proposals),
          'pseudo/object/box': mps.get_object_box(proposals),
      })

      # Entity detection loss.
      pseudo_instance_labels = self._compute_pseudo_instance_labels(
          n_entity=self.n_entity,
          n_proposal=n_proposal,
          proposals=proposals,
          subject_index=subject_ids,
          subject_proposal_index=mps.subject_proposal_index,
          object_index=object_ids,
          object_proposal_index=mps.object_proposal_index,
          iou_threshold_to_propogate=self.options.iou_threshold_to_propogate)
      sgr_proposal_loss = self._compute_proposal_refinement_losses(
          n_proposal=n_proposal,
          labels=pseudo_instance_labels,
          logits=predictions['refine@%i/entity_logits' % i])

      # Relation detection los.
      pseudo_relation_labels = self._compute_pseudo_relation_labels(
          n_predicate=self.n_predicate,
          n_proposal=n_proposal,
          proposals=proposals,
          predicate_index=predicate_ids,
          subject_proposal_index=mps.subject_proposal_index,
          object_proposal_index=mps.object_proposal_index,
          iou_threshold_to_propogate=self.options.
          iou_threshold_to_propogate_relation)
      sgr_relation_loss = self._compute_relation_refinement_losses(
          n_proposal=n_proposal,
          max_n_proposal=max_n_proposal,
          labels=pseudo_relation_labels,
          logits=predictions['refine@%i/relation_logits' % i])

      loss_dict.update({
          'losses/sgr_proposal_loss_%i' % i: sgr_proposal_loss,
          'losses/sgr_relation_loss_%i' % i: sgr_relation_loss
      })

      proposal_scores_0 = predictions['refine@%i/entity_probas' % i]
      proposal_scores_0 = proposal_scores_0[:, :, 1:]
      relation_scores_0 = predictions['refine@%i/relation_probas' % i]
      relation_scores_0 = relation_scores_0[:, :, 1:]

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

    # Parse ground-truth annotations.
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    batch = n_proposal.shape[0]
    max_n_proposal = tf.shape(proposals)[1]

    subject_ids = self.entity2id(inputs['scene_graph/subject'])
    object_ids = self.entity2id(inputs['scene_graph/object'])
    predicate_ids = self.predicate2id(inputs['scene_graph/predicate'])

    n_triple = inputs['scene_graph/n_triple']
    max_n_triple = tf.shape(subject_ids)[1]

    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    gt_subject_box = inputs['scene_graph/subject/box']
    gt_object_box = inputs['scene_graph/object/box']

    # Compute the object detection metrics.
    eval_dict = {
        'key':
            inputs['id'],
        'num_groundtruth_boxes_per_image':
            inputs['scene_graph/n_triple'],
        'groundtruth_classes':
            subject_ids,
        'groundtruth_boxes':
            inputs['scene_graph/subject/box'],
        'num_det_boxes_per_image':
            predictions['object_detection/num_detections'],
        'detection_boxes':
            predictions['object_detection/detection_boxes'],
        'detection_scores':
            predictions['object_detection/detection_scores'],
        'detection_classes':
            predictions['object_detection/detection_classes'],
    }
    det_evaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=self.entity_categories)
    det_metrics = det_evaluator.get_estimator_eval_metric_ops(eval_dict)
    metric_dict.update({
        'metrics/subject_detection/recall@100':
            det_metrics['DetectionBoxes_Recall/AR@100'],
        'metrics/subject_detection/mAP@0.50IOU':
            det_metrics['DetectionBoxes_Precision/mAP@.50IOU'],
    })

    eval_dict.update({
        'groundtruth_classes': object_ids,
        'groundtruth_boxes': inputs['scene_graph/object/box'],
    })
    det_evaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=self.entity_categories)
    det_metrics = det_evaluator.get_estimator_eval_metric_ops(eval_dict)
    metric_dict.update({
        'metrics/object_detection/recall@100':
            det_metrics['DetectionBoxes_Recall/AR@100'],
        'metrics/object_detection/mAP@0.50IOU':
            det_metrics['DetectionBoxes_Precision/mAP@.50IOU'],
    })

    # Compute the scene graph generation metrics.
    eval_dict = {
        'image_id': inputs['id'],
        'groundtruth/n_triple': inputs['scene_graph/n_triple'],
        'groundtruth/subject': inputs['scene_graph/subject'],
        'groundtruth/subject/box': inputs['scene_graph/subject/box'],
        'groundtruth/object': inputs['scene_graph/object'],
        'groundtruth/object/box': inputs['scene_graph/object/box'],
        'groundtruth/predicate': inputs['scene_graph/predicate'],
        'prediction/n_triple': predictions['triple_proposal/n_triple'],
        'prediction/subject/box': predictions['triple_proposal/subject/box'],
        'prediction/object/box': predictions['triple_proposal/object/box'],
        'prediction/subject': predictions['triple_proposal/subject'],
        'prediction/object': predictions['triple_proposal/object'],
        'prediction/predicate': predictions['triple_proposal/predicate'],
    }

    sg_evaluator = SceneGraphEvaluator()
    for k, v in sg_evaluator.get_estimator_eval_metric_ops(eval_dict).items():
      metric_dict['metrics/midn/%s' % k] = v
    metric_dict['metrics/accuracy'] = metric_dict[
        'metrics/midn/scene_graph_recall@100']
    return metric_dict
