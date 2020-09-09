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

import json

import numpy as np
import tensorflow as tf

import tf_slim as slim

from protos import model_pb2

from modeling.layers import token_to_id
from modeling.layers import id_to_token
from modeling.utils import box_ops
from modeling.utils import hyperparams
from modeling.utils import masked_ops

from models import model_base
from models import utils

from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields


class WSSceneGraph(model_base.ModelBase):
  """WSSceneGraph model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    super(WSSceneGraph, self).__init__(options, is_training)

    if not isinstance(options, model_pb2.WSSceneGraph):
      raise ValueError('Options has to be an WSSceneGraph proto.')

    # Load token2id mapping and pre-trained word embedding weights.
    with tf.io.gfile.GFile(options.token_to_id_meta_file, 'r') as fid:
      meta = json.load(fid)
      (entity2id, predicate2id) = (meta['label_to_idx'],
                                   meta['predicate_to_idx'])
    id2entity = {}
    id2predicate = {}
    self.categories = []

    for key in entity2id.keys():
      self.categories.append({'id': entity2id[key], 'name': key})
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

    # Initialize the arg_scope for FC and CONV layers.
    self.arg_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                      is_training)

    # Load word embeddings.
    self.entity_emb_weights = np.load(options.entity_emb_npy_file)[1:]
    self.predicate_emb_weights = np.load(options.predicate_emb_npy_file)[1:]

  def _semantic_relation_feature(self, n_proposal, proposal_features):
    """Extracts semantic relation feature for pairs of proposal nodes.

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      A [batch, max_n_proposal, max_n_proposal, feature_dims] float tensor.
    """
    batch = proposal_features.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]

    # Compute relation feature.
    # - relation_features = [batch, max_n_proposal, max_n_proposal, dims].
    proposal_broadcast1 = tf.broadcast_to(tf.expand_dims(
        proposal_features, 1), [
            batch, max_n_proposal, max_n_proposal,
            proposal_features.shape[-1].value
        ])
    proposal_broadcast2 = tf.broadcast_to(tf.expand_dims(
        proposal_features, 2), [
            batch, max_n_proposal, max_n_proposal,
            proposal_features.shape[-1].value
        ])

    relation_features = tf.concat([
        proposal_broadcast1,
        proposal_broadcast2,
    ], -1)
    with slim.arg_scope(self.arg_scope_fn()):
      relation_features = slim.fully_connected(
          relation_features,
          num_outputs=self.options.relation_hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope='relation/semantic/hidden')
    return relation_features

  def _spatial_relation_feature(self, n_proposal, proposals):
    """Extracts semantic relation feature for any pairs of proposal nodes.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      A [batch, max_n_proposal, max_n_proposal, relation_dims] float tensor.
    """
    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposals)[1]

    # Compute single features.
    centers = box_ops.center(proposals)
    sizes = box_ops.size(proposals)
    areas = tf.expand_dims(box_ops.area(proposals), -1)

    unary_features = tf.concat([proposals, centers, sizes, areas], -1)
    unary_features_broadcast1 = tf.broadcast_to(
        tf.expand_dims(unary_features, 1),
        [batch, max_n_proposal, max_n_proposal, unary_features.shape[-1]])
    unary_features_broadcast2 = tf.broadcast_to(
        tf.expand_dims(unary_features, 2),
        [batch, max_n_proposal, max_n_proposal, unary_features.shape[-1]])

    # Compute pairwise features.
    proposals_broadcast1 = tf.broadcast_to(tf.expand_dims(
        proposals, 1), [batch, max_n_proposal, max_n_proposal, 4])
    proposals_broadcast2 = tf.broadcast_to(tf.expand_dims(
        proposals, 2), [batch, max_n_proposal, max_n_proposal, 4])

    height1, width1 = tf.unstack(box_ops.size(proposals_broadcast1), axis=-1)
    height2, width2 = tf.unstack(box_ops.size(proposals_broadcast2), axis=-1)
    area1 = box_ops.area(proposals_broadcast1)
    area2 = box_ops.area(proposals_broadcast2)

    x_distance = box_ops.x_distance(proposals_broadcast1, proposals_broadcast2)
    y_distance = box_ops.y_distance(proposals_broadcast1, proposals_broadcast2)
    x_intersect = box_ops.x_intersect_len(proposals_broadcast1,
                                          proposals_broadcast2)
    y_intersect = box_ops.y_intersect_len(proposals_broadcast1,
                                          proposals_broadcast2)
    intersect_area = box_ops.area(
        box_ops.intersect(proposals_broadcast1, proposals_broadcast2))
    iou = box_ops.iou(proposals_broadcast1, proposals_broadcast2)

    pairwise_features_list = [
        x_distance, y_distance, x_distance / width1, x_distance / width2,
        y_distance / height1, y_distance / height2, x_intersect, y_intersect,
        x_intersect / width1, x_intersect / width2, y_intersect / height1,
        y_intersect / height2, intersect_area, intersect_area / area1,
        intersect_area / area2, iou
    ]

    relation_features = tf.concat([
        unary_features_broadcast1, unary_features_broadcast2,
        tf.stack(pairwise_features_list, -1)
    ], -1)

    with slim.arg_scope(self.arg_scope_fn()):
      relation_features = slim.fully_connected(
          relation_features,
          num_outputs=self.options.relation_hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope='relation/spatial/hidden')
      relation_features = slim.dropout(relation_features,
                                       self.options.dropout_keep_prob,
                                       is_training=self.is_training)
    return relation_features

  def _edge_scoring_helper(self, attn, logits):
    """Applies edge scoring.  """
    normalizer_fn = tf.nn.softmax

    if model_pb2.ATTENTION == self.options.edge_scoring:
      return attn

    if model_pb2.CLASSIFICATION == self.options.edge_scoring:
      return normalizer_fn(logits)

    if model_pb2.ATTENTION_x_CLASSIFICATION == self.options.edge_scoring:
      return tf.multiply(attn, normalizer_fn(logits))

    raise ValueError('Invalid edge scoring method %i.',
                     self.options.edge_scoring)

  def _multiple_relation_detection(self, n_proposal, proposals,
                                   proposal_features):
    """Detects multiple relations given the ground-truth.

      Multiple Relation Detection (MRD):
      - attn_relation_given_predicate = [batch, max_n_proposal**2, n_predicate(p-gt)]
          * Given that predicate p-gt exists, find the responsible relation i->j.
          * For each p-gt: sum_{i,j} (attn_relation_given_predicate[i, j, p-gt]) = 1.
      - logits_predicate_given_relation = [batch, max_n_proposal**2,
        n_predicate(p-pred)].
          * Logits to classify relation i->j.
      - logits_predicate_given_predicate = [batch, n_predicate(p-gt), 
        n_predicate(p-pred)].
          * Given the predicate p-gt, predict the logits through optimizing the 
            weighting `attn_relation_given_predicate`.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      score_relation_given_predicate: Attention distribution of relations given
        the predicate. shape=[batch, max_n_proposal**2, n_predicate].
      logits_predicate_given_predicate: Predicate prediction.
        shape=[batch, n_predicate, n_predicate].
    """
    batch = proposal_features.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    # Compute `semantic_relation_features` and `spatial_relation_features`.
    # - `semantic_relation_features`=[batch, max_n_proposal**2, semantic_dims].
    # - `spatial_relation_features`=[batch, max_n_proposal**2, spatial_dims].
    semantic_relation_features = self._semantic_relation_feature(
        n_proposal, proposal_features)
    spatial_relation_features = self._spatial_relation_feature(
        n_proposal, proposals)

    semantic_relation_features = tf.reshape(semantic_relation_features, [
        batch, max_n_proposal * max_n_proposal,
        semantic_relation_features.shape[-1].value
    ])
    spatial_relation_features = tf.reshape(spatial_relation_features, [
        batch, max_n_proposal * max_n_proposal,
        spatial_relation_features.shape[-1].value
    ])

    # Two-branch MRD approach and late fusion (i.e., fusing the logits).
    # - `relation_features` = [batch, max_n_proposal**2, dims].
    # - `relation_masks` = [batch, max_n_proposal**2, 1].
    relation_masks = tf.multiply(tf.expand_dims(proposal_masks, 1),
                                 tf.expand_dims(proposal_masks, 2))
    relation_masks = tf.reshape(relation_masks,
                                [batch, max_n_proposal * max_n_proposal, 1])

    logits_relation_given_predicate = []
    logits_predicate_given_relation = []

    for (scope, weight, relation_features) in [
        ('MRD/semantic', self.options.mrd_semantic_feature_weight,
         semantic_relation_features),
        ('MRD/spatial', self.options.mrd_spatial_feature_weight,
         spatial_relation_features)
    ]:
      with slim.arg_scope(self.arg_scope_fn()):
        logits_relation_given_predicate.append(
            weight * slim.fully_connected(relation_features,
                                          num_outputs=self.n_predicate,
                                          activation_fn=None,
                                          scope='{}/branch1'.format(scope)))
        logits_predicate_given_relation.append(
            weight * slim.fully_connected(relation_features,
                                          num_outputs=self.n_predicate,
                                          activation_fn=None,
                                          scope='{}/branch2'.format(scope)))

    logits_relation_given_predicate = tf.add_n(logits_relation_given_predicate)
    logits_predicate_given_relation = tf.add_n(logits_predicate_given_relation)

    # Compute MRD results.
    # - `attn_relation_given_predicate` = [batch,  max_n_proposal**2, n_predicate].
    # - `logits_predicate_given_predicate` = [batch,  n_predicate, n_predicate].
    attn_relation_given_predicate = masked_ops.masked_softmax(
        logits_relation_given_predicate, relation_masks, dim=1)
    attn_relation_given_predicate = slim.dropout(
        attn_relation_given_predicate,
        self.options.dropout_keep_prob,
        is_training=(self.is_training and self.options.attention_dropout))

    logits_predicate_given_predicate = tf.matmul(
        attn_relation_given_predicate,
        logits_predicate_given_relation,
        transpose_a=True)

    detection_score = self._edge_scoring_helper(
        attn_relation_given_predicate, logits_predicate_given_relation)
    return detection_score, logits_predicate_given_predicate

  def _multiple_entity_detection(self, n_proposal, proposal_features):
    """Detects multiple entities given the ground-truth.

      Multiple Entity Detection (MED):
      - attn_proposal_given_entity = [batch, max_n_proposal, n_entity(e-gt)].
          * Given that entity e-gt exists, find the responsible proposal r.
          * For each e-gt: sum_{r} (attn_proposal_given_entity[r, e-gt]) = 1.
      - logits_entity_given_proposal = [batch, max_n_proposal n_entity(e-pred)],
          * Logits to classify proposal r.
      - logits_entity_given_entity = [batch, n_entity(e-gt), n_entity(e-pred)].
          * Given the entity e-gt, predict the logits through optimizing the
            weighting `attn_proposal_given_entity`.

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      score_proposal_given_entity: Attention distribution of proposals given the
        entities. shape=[batch, max_n_proposal, n_entity]
      logits_entity_given_entity: Entity prediction.
        shape=[batch, n_entity, n_entity]
      proposal_hidden: proposal features for the next relation computing.
    """
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      # Two branches.
      weights_initializer = tf.compat.v1.constant_initializer(
          self.entity_emb_weights.transpose())
      logits_proposal_given_entity = slim.fully_connected(
          proposal_features,
          num_outputs=self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope="MED/branch/detection")
      logits_entity_given_proposal = slim.fully_connected(
          proposal_features,
          num_outputs=self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope="MED/branch/classification")

    attn_proposal_given_entity = masked_ops.masked_softmax(
        logits_proposal_given_entity,
        mask=tf.expand_dims(proposal_masks, -1),
        dim=1)
    attn_proposal_given_entity = slim.dropout(
        attn_proposal_given_entity,
        self.options.dropout_keep_prob,
        is_training=(self.is_training and self.options.attention_dropout))

    logits_entity_given_entity = tf.matmul(attn_proposal_given_entity,
                                           logits_entity_given_proposal,
                                           transpose_a=True)

    detection_score = self._edge_scoring_helper(attn_proposal_given_entity,
                                                logits_entity_given_proposal)
    return detection_score, logits_entity_given_entity

  def _scene_graph_refinement(self, n_proposal, proposal_features, scope):
    """Refines scene graph predictions.

      Scene Graph Refinement (SGR).

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, dims] float tensor.
      scope: Variable scope.

    Returns:
      logits_entity_given_proposal: A [batch, max_n_proposal, 1 + n_entity] tensor.
    """
    # Add 0s for the OOV embedding.
    dims = self.entity_emb_weights.shape[-1]
    entity_emb_weights = np.concatenate(
        [np.zeros((1, dims)), self.entity_emb_weights], axis=0)

    with slim.arg_scope(self.arg_scope_fn()):
      weights_initializer = tf.compat.v1.constant_initializer(
          entity_emb_weights.transpose())
      logits_entity_given_proposal = slim.fully_connected(
          proposal_features,
          num_outputs=1 + self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope=scope)
    return logits_entity_given_proposal

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
    # - `scene_graph/subject/box`
    # - `scene_graph/object/box`

    if self.is_training:
      inputs.pop('scene_graph/subject/box')
      inputs.pop('scene_graph/object/box')

    # Extract proposal features.
    # - proposal_masks = [batch, max_n_proposal].
    # - proposal_features = [batch, max_n_proposal, feature_dims].
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      proposal_features = slim.dropout(proposal_features,
                                       self.options.dropout_keep_prob,
                                       is_training=self.is_training)

      proposal_hiddens = slim.fully_connected(
          proposal_features,
          num_outputs=self.options.entity_hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope="proposal/hidden")
      proposal_hiddens = slim.dropout(proposal_hiddens,
                                      self.options.dropout_keep_prob,
                                      is_training=self.is_training)

    # Multiple Entity Detection (MED).
    # - `score_proposal_given_entity`=[batch, max_n_proposal, n_entity].
    # - `logits_entity_given_entity`=[batch, n_entity, n_entity].
    (score_proposal_given_entity,
     logits_entity_given_entity) = self._multiple_entity_detection(
         n_proposal, proposal_hiddens)

    # Multiple Relation Detection (MRD).
    # - `score_relation_given_predicate`=[batch, max_n_proposal**2, n_predicate].
    # - `logits_predicate_given_predicate`=shape=[batch, n_predicate, n_predicate].
    (score_relation_given_predicate,
     logits_predicate_given_predicate) = self._multiple_relation_detection(
         n_proposal, proposals, proposal_hiddens)

    # Scene Graph Refinement (SGR).
    # - `proposal_scores`=[batch, max_n_proposal, 1 + n_entity].
    # - `proposal_probas`=[batch, max_n_proposal, n_entity].

    proposal_scores = self._scene_graph_refinement(n_proposal,
                                                   proposal_hiddens,
                                                   scope='SGR/logits')
    proposal_probas = tf.nn.sigmoid(proposal_scores)[:, :, 1:]

    (num_detections, detection_boxes, detection_scores,
     detection_class_ids) = utils.nms_post_process(
         n_proposal,
         proposals,
         proposal_probas,
         self.options.post_process.max_output_size_per_class,
         self.options.post_process.max_total_size,
         self.options.post_process.iou_threshold,
         score_threshold=self.options.post_process.score_threshold)
    detection_classes = self.id2entity(detection_class_ids)

    # Parse the image-level scene-graph.
    n_triple = inputs['scene_graph/n_triple']
    subject_ids = self.entity2id(inputs['scene_graph/subject'])
    object_ids = self.entity2id(inputs['scene_graph/object'])
    predicate_ids = self.predicate2id(inputs['scene_graph/predicate'])
    max_n_triple = tf.shape(subject_ids)[1]

    # Get the selection indices. Note: `index = id - 1`.
    # - `batch_index` = [batch, max_n_triple],
    # - `[subject/object/predicate]_index` = [batch, max_n_triple, 2].
    batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_n_triple])
    subject_index = tf.stack([batch_index, subject_ids], -1)
    object_index = tf.stack([batch_index, object_ids], -1)
    predicate_index = tf.stack([batch_index, predicate_ids], -1)

    # Get the image-level prediction.
    # - `[subject/object]_logits` = [batch, max_n_triple, n_entity].
    # - `predicate_logits` = [batch, max_n_triple, n_predicate].

    subject_logits = tf.gather_nd(logits_entity_given_entity, subject_index)
    object_logits = tf.gather_nd(logits_entity_given_entity, object_index)
    predicate_logits = tf.gather_nd(logits_predicate_given_predicate,
                                    predicate_index)

    # Get the pseudo boxes, seek the DP-solution in the graph.
    # - `score_proposal_given_entity` = [batch, max_n_proposal, n_entity].
    # - `score_relation_given_predicate` = [batch, max_n_proposal**2, n_predicate].
    # - `subject_to_proposal` = [batch, max_n_triple, max_n_proposal].
    # - `proposal_to_object` = [batch, max_n_triple, max_n_proposal].
    # - `proposal_to_proposal` = [batch, max_n_triple, max_n_proposal**2].

    subject_to_proposal = tf.gather_nd(
        tf.transpose(score_proposal_given_entity, [0, 2, 1]), subject_index)
    proposal_to_object = tf.gather_nd(
        tf.transpose(score_proposal_given_entity, [0, 2, 1]), object_index)
    proposal_to_proposal = tf.gather_nd(
        tf.transpose(score_relation_given_predicate, [0, 2, 1]),
        predicate_index)
    proposal_to_proposal = tf.reshape(
        proposal_to_proposal,
        [batch, max_n_triple, max_n_proposal, max_n_proposal])

    proposal_to_proposal = tf.multiply(0.0, proposal_to_proposal)
    (max_path_sum, ps_subject_proposal_i,
     ps_object_proposal_j) = utils.compute_max_path_sum(n_proposal, n_triple,
                                                        subject_to_proposal,
                                                        proposal_to_proposal,
                                                        proposal_to_object)
    ps_subject_box = utils.gather_proposal_by_index(proposals,
                                                    ps_subject_proposal_i)
    ps_object_box = utils.gather_proposal_by_index(proposals,
                                                   ps_object_proposal_j)

    # Extract the edge weights in the optimal path.
    # - `edge_weight_subject_to_proposal` = [batch, max_n_triple].
    # - `edge_weight_proposal_to_object` = [batch, max_n_triple].

    edge_weight_subject_to_proposal = utils.gather_proposal_score_by_index(
        subject_to_proposal, ps_subject_proposal_i)
    edge_weight_proposal_to_object = utils.gather_proposal_score_by_index(
        proposal_to_object, ps_object_proposal_j)
    edge_weight_proposal_to_proposal = utils.gather_relation_score_by_index(
        proposal_to_proposal, ps_subject_proposal_i, ps_object_proposal_j)

    predictions = {
        'pseudo/subject/ids':
            subject_ids,
        'pseudo/subject/logits':
            subject_logits,
        'pseudo/subject/box':
            ps_subject_box,
        'pseudo/subject/box/index':
            ps_subject_proposal_i,
        'pseudo/object/ids':
            object_ids,
        'pseudo/object/logits':
            object_logits,
        'pseudo/object/box':
            ps_object_box,
        'pseudo/object/box/index':
            ps_object_proposal_j,
        'pseudo/predicate/ids':
            predicate_ids,
        'pseudo/predicate/logits':
            predicate_logits,
        'pseudo/graph/subject_to_proposal':
            subject_to_proposal,
        'pseudo/graph/proposal_to_proposal':
            proposal_to_proposal,
        'pseudo/graph/proposal_to_object':
            proposal_to_object,
        'pseudo/mps_path/optimal':
            max_path_sum,
        'pseudo/mps_path/subject_to_proposal':
            edge_weight_subject_to_proposal,
        'pseudo/mps_path/proposal_to_proposal':
            edge_weight_proposal_to_proposal,
        'pseudo/mps_path/proposal_to_object':
            edge_weight_proposal_to_object,
        'refinement/proposal_scores':
            proposal_scores,
        'refinement/num_detection':
            num_detections,
        'refinement/detection_boxes':
            detection_boxes,
        'refinement/detection_scores':
            detection_scores,
        'refinement/detection_class_ids':
            detection_class_ids,
        'refinement/detection_classes':
            detection_classes,
    }

    return predictions

  def _compute_multiple_instance_detection_losses(
      self, n_triple, subject_labels, subject_logits, object_labels,
      object_logits, predicate_labels, predicate_logits):
    """Computes MED and MRD losses.

    Args:
      n_triple: A [batch] int tensor.
      subject_labels: A [batch, max_n_triple] int tensor.
      subject_logits: A [batch, max_n_triple, n_entity] float tensor.
      object_labels: A [batch, max_n_triple] int tensor.
      object_logits: A [batch, max_n_triple, n_entity] float tensor.
      predicate_labels: A [batch, max_n_triple] int tensor.
      predicate_logits: A [batch, max_n_triple, n_predicate] float tensor.

    Returns:
      loss_dict involving the following fields:
        - `losses/med_subject_loss`: MED loss for subject.
        - `losses/med_object_loss`: MED loss for object.
        - `losses/mrd_predicate_loss`: MRD loss for predicate.
    """
    max_n_triple = tf.shape(subject_labels)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    loss_dict = {}
    for name, labels, logits in [
        ('losses/med_subject_loss', subject_labels, subject_logits),
        ('losses/med_object_loss', object_labels, object_logits),
        ('losses/mrd_predicate_loss', predicate_labels, predicate_logits)
    ]:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits)
      loss = tf.reduce_mean(masked_ops.masked_avg(loss, triple_mask, dim=1))
      loss_dict[name] = loss
    return loss_dict

  def _compute_scene_graph_refinement_losses(self, n_proposal, labels, logits):
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

    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
    loss = tf.reduce_mean(
        masked_ops.masked_avg(losses, tf.expand_dims(proposal_masks, -1),
                              dim=1))
    return {'losses/sgr_detection_loss': loss}

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
      loss_dict involving the following fields:
    """
    max_n_proposal = tf.shape(proposals)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)
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

    instance_labels = utils.post_process_pseudo_entity_detection_labels(
        tf.add_n(instance_labels), normalize=False)
    return instance_labels

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    loss_dict = {}

    # MED/MRD losses.
    mid_loss_dict = self._compute_multiple_instance_detection_losses(
        n_triple=inputs['scene_graph/n_triple'],
        subject_labels=predictions['pseudo/subject/ids'],
        subject_logits=predictions['pseudo/subject/logits'],
        object_labels=predictions['pseudo/object/ids'],
        object_logits=predictions['pseudo/object/logits'],
        predicate_labels=predictions['pseudo/predicate/ids'],
        predicate_logits=predictions['pseudo/predicate/logits'])
    loss_dict.update(mid_loss_dict)

    # SGR losses.
    # - `pseudo_instance_labels`=[batch, max_n_proposal, 1 + n_entity].
    pseudo_instance_labels = self._compute_pseudo_instance_labels(
        n_entity=self.n_entity,
        n_proposal=inputs['image/n_proposal'],
        proposals=inputs['image/proposal'],
        subject_index=predictions['pseudo/subject/ids'],
        subject_proposal_index=predictions['pseudo/subject/box/index'],
        object_index=predictions['pseudo/object/ids'],
        object_proposal_index=predictions['pseudo/object/box/index'],
        iou_threshold_to_propogate=self.options.iou_threshold_to_propogate)
    sgr_loss_dict = self._compute_scene_graph_refinement_losses(
        n_proposal=inputs['image/n_proposal'],
        labels=pseudo_instance_labels,
        logits=predictions['refinement/proposal_scores'])
    loss_dict.update(sgr_loss_dict)

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
    gt_subject_box = inputs['scene_graph/subject/box']
    gt_object_box = inputs['scene_graph/object/box']

    ps_subject_box = predictions['pseudo/subject/box']
    ps_object_box = predictions['pseudo/object/box']

    n_triple = inputs['scene_graph/n_triple']
    max_n_triple = tf.shape(ps_subject_box)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    # Compute the classification accuracy.
    subject_labels = predictions['pseudo/subject/ids']
    object_labels = predictions['pseudo/object/ids']
    predicate_labels = predictions['pseudo/predicate/ids']

    subject_logits = predictions['pseudo/subject/logits']
    object_logits = predictions['pseudo/object/logits']
    predicate_logits = predictions['pseudo/predicate/logits']

    for name, labels, logits in [
        ('metrics/predict_subject', subject_labels, subject_logits),
        ('metrics/predict_object', object_labels, object_logits),
        ('metrics/predict_predicate', predicate_labels, predicate_logits)
    ]:
      bingo = tf.equal(tf.cast(tf.argmax(logits, -1), tf.int32), labels)
      accuracy_metric = tf.keras.metrics.Mean()
      accuracy_metric.update_state(bingo)
      metric_dict[name] = accuracy_metric

    # Compute pseudo box recall at IoU 0.5.
    subject_iou = box_ops.iou(gt_subject_box, ps_subject_box)
    object_iou = box_ops.iou(gt_object_box, ps_object_box)

    for iou_threshold in [0.5]:
      recalled_subject = tf.greater_equal(subject_iou, iou_threshold)
      recalled_object = tf.greater_equal(object_iou, iou_threshold)
      recalled_relation = tf.logical_and(recalled_subject, recalled_object)

      for name, value in [
          ('metrics/pseudo/iou/subject', subject_iou),
          ('metrics/pseudo/iou/object', object_iou),
          ('metrics/pseudo/recall/subject@%.2lf' % iou_threshold,
           recalled_subject),
          ('metrics/pseudo/recall/object@%.2lf' % iou_threshold,
           recalled_object),
          ('metrics/pseudo/recall/relation@%.2lf' % iou_threshold,
           recalled_relation)
      ]:
        mean_metric = tf.keras.metrics.Mean()
        value = tf.cast(value, tf.float32)
        mean_value = tf.reduce_mean(
            masked_ops.masked_avg(value, mask=triple_mask, dim=1))
        mean_metric.update_state(mean_value)
        metric_dict[name] = mean_metric

    # Compute detection recall.
    num_detection = predictions['refinement/num_detection']
    detection_boxes = predictions['refinement/detection_boxes']
    detection_scores = predictions['refinement/detection_scores']
    detection_classes = predictions['refinement/detection_class_ids']

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    for (recall10_name, recall100_name, gt_box, gt_label) in [
        ('metrics/refinement/recall@10/subject',
         'metrics/refinement/recall@100/subject', gt_subject_box,
         subject_labels),
        ('metrics/refinement/recall@10/object',
         'metrics/refinement/recall@100/object', gt_object_box, object_labels)
    ]:

      evaluator = coco_evaluation.CocoDetectionEvaluator(self.categories)
      eval_dict = {
          input_data_fields.key: inputs['id'],
          'num_groundtruth_boxes_per_image': n_triple,
          input_data_fields.groundtruth_boxes: gt_box,
          input_data_fields.groundtruth_classes: 1 + gt_label,
          'num_det_boxes_per_image': num_detection,
          detection_fields.detection_boxes: detection_boxes,
          detection_fields.detection_scores: detection_scores,
          detection_fields.detection_classes: 1 + detection_classes,
      }
      eval_ops = evaluator.get_estimator_eval_metric_ops(eval_dict)
      metric_dict[recall10_name] = eval_ops['DetectionBoxes_Recall/AR@10']
      metric_dict[recall100_name] = eval_ops['DetectionBoxes_Recall/AR@100']

    return metric_dict
