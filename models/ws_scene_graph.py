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

from models.graph_mps import GraphMPS
from models.graph_nms import GraphNMS
from modeling.layers import id_to_token
from modeling.layers import token_to_id
from modeling.utils import box_ops
from modeling.utils import hyperparams
from modeling.utils import masked_ops

from models import model_base
from models import utils

from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields

from metrics.scene_graph_evaluation import SceneGraphEvaluator


class WSSceneGraph(model_base.ModelBase):
  """WSSceneGraph model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the WSSceneGraph instance. """
    super(WSSceneGraph, self).__init__(options, is_training)

    if (not isinstance(options, model_pb2.WSSceneGraph) and
        not isinstance(options, model_pb2.WSSceneGraphRnnRefine)):
      raise ValueError(
          'Options has to be an WSSceneGraph/WSSceneGraphRnnRefine proto.')

    # Load token2id mapping, convert 1-based index to 0-based index.
    with tf.io.gfile.GFile(options.token_to_id_meta_file, 'r') as fid:
      meta = json.load(fid)
      (entity2id, predicate2id) = (meta['label_to_idx'],
                                   meta['predicate_to_idx'])
    id2entity = {}
    id2predicate = {}
    for key in entity2id.keys():
      entity2id[key] -= 1
      id2entity[entity2id[key]] = key
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

    # Load pre-trained word embeddings.
    self.entity_emb_weights = np.load(options.entity_emb_npy_file)[1:]
    self.predicate_emb_weights = np.load(options.predicate_emb_npy_file)[1:]

    # Initialize the arg_scope for FC layers.
    self.arg_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                      is_training)

  def _spatial_relation_feature(self, n_proposal, proposals):
    """Extracts spatial relation feature for any two proposal nodes.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      A [batch, max_n_proposal, max_n_proposal, relation_dims] float tensor.
    """
    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposals)[1]

    proposals_broadcast1 = tf.broadcast_to(tf.expand_dims(
        proposals, 2), [batch, max_n_proposal, max_n_proposal, 4])
    proposals_broadcast2 = tf.broadcast_to(tf.expand_dims(
        proposals, 1), [batch, max_n_proposal, max_n_proposal, 4])

    return utils.compute_spatial_relation_feature(proposals_broadcast1,
                                                  proposals_broadcast2)

  def _relation_classification(self, n_proposal, proposals, proposal_features):
    """Classifies relations given the proposal features.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      relation_logits: A [batch, max_n_proposal**2, 1 + n_predicate] float tensor.
    """
    dims = self.predicate_emb_weights.shape[-1]
    predicate_emb_weights = np.concatenate(
        [np.zeros((1, dims)), self.predicate_emb_weights], axis=0)
    weights_initializer = tf.compat.v1.constant_initializer(
        predicate_emb_weights.transpose())

    # Predict the predicate given the subject/object.
    # - logits_predicate_given_subject = [batch, max_n_proposal, 1 + n_predicate].
    # - logits_predicate_given_object = [batch, max_n_proposal, 1 + n_predicate].
    with slim.arg_scope(self.arg_scope_fn()):
      logits_predicate_given_subject = slim.fully_connected(
          proposal_features,
          num_outputs=1 + self.n_predicate,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope='relation/attach_to_subject')
      logits_predicate_given_object = slim.fully_connected(
          proposal_features,
          num_outputs=1 + self.n_predicate,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope='relation/attach_to_object')

    relation_logits = tf.minimum(
        tf.expand_dims(logits_predicate_given_subject, 2),
        tf.expand_dims(logits_predicate_given_object, 1))

    # Reshape.
    batch = proposal_features.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    relation_logits = tf.reshape(relation_logits, [
        batch, max_n_proposal * max_n_proposal, relation_logits.shape[-1].value
    ])
    return relation_logits

  def _multiple_entity_detection(self, n_proposal, proposal_features):
    """Detects multiple entities given the ground-truth.

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      detection_score: Detection score, shape=[batch, max_n_proposal, n_entity]
      logits_entity_given_entity: Entity prediction, shape=[batch, n_entity, n_entity].
    """

    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    # MIDN network: Two-branch approach.
    weights_initializer = tf.compat.v1.constant_initializer(
        self.entity_emb_weights.transpose())
    with slim.arg_scope(self.arg_scope_fn()):
      logits_proposal_given_entity = slim.fully_connected(
          proposal_features,
          num_outputs=self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope='entity/det_branch')
      logits_entity_given_proposal = slim.fully_connected(
          proposal_features,
          num_outputs=self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope='entity/cls_branch')

    logits_proposal_given_entity *= self.options.attn_scale
    attn_proposal_given_entity = masked_ops.masked_softmax(
        logits_proposal_given_entity,
        mask=tf.expand_dims(proposal_masks, -1),
        dim=1)
    attn_proposal_given_entity = slim.dropout(
        attn_proposal_given_entity,
        self.options.attn_dropout_keep_prob,
        is_training=self.is_training)

    logits_entity_given_entity = tf.matmul(attn_proposal_given_entity,
                                           logits_entity_given_proposal,
                                           transpose_a=True)
    detection_score = tf.multiply(attn_proposal_given_entity,
                                  tf.nn.softmax(logits_entity_given_proposal))

    tf.summary.histogram('attn_logits', logits_proposal_given_entity)
    return detection_score, logits_entity_given_entity

  def _refine_entity_scores_once(self, n_proposal, proposal_features, scope):
    """Refines scene graph entity score predictions.

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, dims] float tensor.
      scope: Name of the variable scope.

    Returns:
      logits_entity_given_proposal: A [batch, max_n_proposal, 1 + n_entity] tensor.
    """
    # Add ZEROs to the first row for the OOV embedding.
    dims = self.entity_emb_weights.shape[-1]
    entity_emb_weights = np.concatenate(
        [np.zeros((1, dims)), self.entity_emb_weights], axis=0)

    weights_initializer = tf.compat.v1.constant_initializer(
        entity_emb_weights.transpose())
    with slim.arg_scope(self.arg_scope_fn()):
      logits_entity_given_proposal = slim.fully_connected(
          proposal_features,
          num_outputs=1 + self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope=scope)
    return logits_entity_given_proposal

  def _refine_entity_scores(self, n_proposal, proposals, proposal_features,
                            n_refine_iteration):
    """Refines scene graph entity scores.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, dims] float tensor.
      n_refine_iteration: Refine iterations.

    Returns:
      entity_scores_list: A list of [batch, max_n_proposal, 1 + n_entity] 
        float tensors.
    """
    return [
        self._refine_entity_scores_once(n_proposal,
                                        proposal_features,
                                        scope='refine/iter_%i/entity' % i)
        for i in range(1, 1 + n_refine_iteration)
    ]

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

    # Ground-truth boxes can only be used for evaluation/visualization:
    # - Pop `scene_graph/subject/box`
    # - Pop `scene_graph/object/box`
    if self.is_training:
      inputs.pop('scene_graph/subject/box')
      inputs.pop('scene_graph/object/box')

    # Extract proposal features.
    # - proposal_masks = [batch, max_n_proposal].
    # - proposals = [batch, max_n_proposal, 4].
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    # Checking the refinement score normalizer.
    score_normalizer = (tf.nn.softmax
                        if self.options.refine_use_softmax else tf.nn.sigmoid)

    # One additional hidden layers.
    with slim.arg_scope(self.arg_scope_fn()):
      shared_hiddens = slim.fully_connected(
          proposal_features,
          num_outputs=self.options.entity_hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope="shared_hidden")
    shared_hiddens = slim.dropout(shared_hiddens,
                                  self.options.dropout_keep_prob,
                                  is_training=self.is_training)
    predictions.update({'features/shared_hidden': shared_hiddens})

    # Multiple Entity Detection (MED).
    # - `detection_scores`=[batch, max_n_proposal, n_entity].
    # - `logits_entity_given_entity`=[batch, n_entity, n_entity].

    (detection_scores,
     logits_entity_given_entity) = self._multiple_entity_detection(
         n_proposal, shared_hiddens)
    predictions.update({
        'pseudo/logits_entity_given_entity': logits_entity_given_entity,
        'refinement/iter_0/proposal_probas': detection_scores,
    })

    # Entity Score Refinement (ESR).
    # - `proposal_scores/probas`=[batch, max_n_proposal, 1 + n_entity].
    entity_scores_list = self._refine_entity_scores(
        n_proposal, proposals, shared_hiddens, self.options.n_refine_iteration)

    for i, entity_scores in enumerate(entity_scores_list):
      predictions.update({
          'refinement/iter_%i/proposal_logits' % (1 + i):
              entity_scores,
          'refinement/iter_%i/proposal_probas' % (1 + i):
              score_normalizer(entity_scores),
      })

    # Relation Classification (RC).
    # - `relation_logits`=[batch, max_n_proposal * max_n_proposal, n_predicate].
    relation_logits = self._relation_classification(n_proposal, proposals,
                                                    shared_hiddens)
    predictions.update({
        'refinement/relation_logits': relation_logits,
        'refinement/relation_probas': score_normalizer(relation_logits),
    })

    # Post-process the predictions.
    if not self.is_training:
      graph_proposal_scores = predictions['refinement/iter_%i/proposal_probas' %
                                          self.options.n_refine_iteration]
      graph_relation_scores = predictions['refinement/relation_probas']
      graph_proposal_scores = graph_proposal_scores[:, :, 1:]
      graph_relation_scores = graph_relation_scores[:, :, 1:]

      # Search for the triple proposals.
      search = GraphNMS(
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

      predictions.update({
          'search/n_triple': search.n_triple,
          'search/subject': self.id2entity(search.subject_id),
          'search/subject/score': search.subject_score,
          'search/subject/box': search.get_subject_box(proposals),
          'search/object': self.id2entity(search.object_id),
          'search/object/score': search.object_score,
          'search/object/box': search.get_object_box(proposals),
          'search/predicate': self.id2predicate(search.predicate_id),
          'search/predicate/score': search.predicate_score,
      })

    return predictions

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

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    loss_dict = {}

    # Parse ground-truth annotations.
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']
    batch = n_proposal.shape[0]
    max_n_proposal = tf.shape(proposals)[1]

    subject_ids = self.entity2id(inputs['scene_graph/subject'])
    object_ids = self.entity2id(inputs['scene_graph/object'])
    predicate_ids = self.predicate2id(inputs['scene_graph/predicate'])

    n_triple = inputs['scene_graph/n_triple']
    max_n_triple = tf.shape(subject_ids)[1]

    (subject_index, object_index,
     predicate_index) = self._create_selection_indices(subject_ids, object_ids,
                                                       predicate_ids)

    # Multiple Entity Detection losses.
    subject_logits = tf.gather_nd(
        predictions['pseudo/logits_entity_given_entity'], subject_index)
    object_logits = tf.gather_nd(
        predictions['pseudo/logits_entity_given_entity'], object_index)
    loss_dict.update(
        self._compute_multiple_entity_detection_losses(
            n_triple=n_triple,
            subject_labels=subject_ids,
            subject_logits=subject_logits,
            object_labels=object_ids,
            object_logits=object_logits))

    # Entity Refinement loss.
    # - `proposal_scores_0`=[batch, max_n_proposal, n_entity].
    # - `relation_scores_0`=[batch, max_n_proposal**2, 1 + n_predicate].
    proposal_scores_0 = predictions['refinement/iter_0/proposal_probas']
    relation_scores_0 = predictions['refinement/relation_probas'][:, :, 1:]

    proposal_to_proposal_weight = slim.dropout(
        self.options.joint_inferring_relation_weight,
        self.options.mps_relation_dropout_keep_prob,
        is_training=self.is_training)
    for i in range(1, 1 + self.options.n_refine_iteration):
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
          logits=predictions['refinement/iter_%i/proposal_logits' % i])

      loss_dict.update({
          'losses/sgr_proposal_loss_%i' % i:
              self.options.proposal_refine_loss_weight * sgr_proposal_loss,
      })
      proposal_scores_0 = predictions['refinement/iter_%i/proposal_probas' % i]
      proposal_scores_0 = proposal_scores_0[:, :, 1:]

    # Relation refinement loss, relations are build on refined proposals.
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
        logits=predictions['refinement/relation_logits'])
    loss_dict.update({
        'losses/sgr_relation_loss_%i' % i:
            self.options.relation_refine_loss_weight * sgr_relation_loss
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

    (subject_index, object_index,
     predicate_index) = self._create_selection_indices(subject_ids, object_ids,
                                                       predicate_ids)

    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    gt_subject_box = inputs['scene_graph/subject/box']
    gt_object_box = inputs['scene_graph/object/box']

    # Compute the classification accuracy.
    subject_logits = tf.gather_nd(
        predictions['pseudo/logits_entity_given_entity'], subject_index)
    object_logits = tf.gather_nd(
        predictions['pseudo/logits_entity_given_entity'], object_index)

    for name, labels, logits in [
        ('metrics/predict_subject', subject_ids, subject_logits),
        ('metrics/predict_object', object_ids, object_logits),
    ]:
      bingo = tf.equal(tf.cast(tf.argmax(logits, -1), tf.int32), labels)
      accuracy_metric = tf.keras.metrics.Mean()
      accuracy_metric.update_state(bingo)
      metric_dict[name] = accuracy_metric

    # Compute pseudo box grounding performance.
    iou_threshold = 0.5
    for i in range(0, 1 + self.options.n_refine_iteration):
      graph_proposal_scores = predictions['refinement/iter_%i/proposal_probas' %
                                          i]
      graph_relation_scores = predictions['refinement/relation_probas'][:, :,
                                                                        1:]
      if i > 0:
        graph_proposal_scores = graph_proposal_scores[:, :, 1:]
      mps = GraphMPS(
          n_triple=n_triple,
          n_proposal=n_proposal,
          subject_to_proposal=tf.gather_nd(
              tf.transpose(graph_proposal_scores, [0, 2, 1]), subject_index),
          proposal_to_proposal=tf.reshape(
              tf.gather_nd(tf.transpose(graph_relation_scores, [0, 2, 1]),
                           predicate_index),
              [batch, max_n_triple, max_n_proposal, max_n_proposal]),
          proposal_to_object=tf.gather_nd(
              tf.transpose(graph_proposal_scores, [0, 2, 1]), object_index),
          subject_to_proposal_weight=1.0,
          proposal_to_proposal_weight=1.0,
          proposal_to_object_weight=1.0,
          use_log_prob=self.options.use_log_prob)

      subject_iou = box_ops.iou(gt_subject_box, mps.get_subject_box(proposals))
      object_iou = box_ops.iou(gt_object_box, mps.get_object_box(proposals))

      recalled_subject = tf.greater_equal(subject_iou, iou_threshold)
      recalled_object = tf.greater_equal(object_iou, iou_threshold)
      recalled_relation = tf.logical_and(recalled_subject, recalled_object)

      for name, value in [
          ('metrics@%i/pseudo/iou/subject' % i, subject_iou),
          ('metrics@%i/pseudo/iou/object' % i, object_iou),
          ('metrics@%i/pseudo/recall/subject@%.2lf' % (i, iou_threshold),
           recalled_subject),
          ('metrics@%i/pseudo/recall/object@%.2lf' % (i, iou_threshold),
           recalled_object),
          ('metrics@%i/pseudo/recall/relation@%.2lf' % (i, iou_threshold),
           recalled_relation)
      ]:
        mean_metric = tf.keras.metrics.Mean()
        value = tf.cast(value, tf.float32)
        mean_value = tf.reduce_mean(
            masked_ops.masked_avg(value, mask=triple_mask, dim=1))
        mean_metric.update_state(mean_value)
        metric_dict[name] = mean_metric

    eval_dict = {
        'image_id': inputs['id'],
        'groundtruth/n_triple': inputs['scene_graph/n_triple'],
        'groundtruth/subject': inputs['scene_graph/subject'],
        'groundtruth/subject/box': inputs['scene_graph/subject/box'],
        'groundtruth/object': inputs['scene_graph/object'],
        'groundtruth/object/box': inputs['scene_graph/object/box'],
        'groundtruth/predicate': inputs['scene_graph/predicate'],
        'prediction/n_triple': predictions['search/n_triple'],
        'prediction/subject/box': predictions['search/subject/box'],
        'prediction/object/box': predictions['search/object/box'],
        'prediction/subject': predictions['search/subject'],
        'prediction/object': predictions['search/object'],
        'prediction/predicate': predictions['search/predicate'],
    }

    evaluator = SceneGraphEvaluator()
    for k, v in evaluator.get_estimator_eval_metric_ops(eval_dict).items():
      metric_dict['metrics/midn/%s' % k] = v
    metric_dict['metrics/accuracy'] = metric_dict[
        'metrics/midn/scene_graph_recall@100']
    return metric_dict
