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
from modeling.utils import box_ops
from modeling.utils import hyperparams
from modeling.utils import masked_ops

from models import model_base
from models import utils

embedding_lookup = tf.nn.embedding_lookup


class WeaklySupervisedSceneGraph(model_base.ModelBase):
  """WeaklySupervisedSceneGraph model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    super(WeaklySupervisedSceneGraph, self).__init__(options, is_training)

    if not isinstance(options, model_pb2.WeaklySupervisedSceneGraph):
      raise ValueError('Options has to be an WeaklySupervisedSceneGraph proto.')

    # Load token2id mapping and pre-trained word embedding weights.
    with tf.io.gfile.GFile(options.token_to_id_meta_file, 'r') as fid:
      meta = json.load(fid)
      (entity2id, predicate2id) = (meta['label_to_idx'],
                                   meta['predicate_to_idx'])
    for key in entity2id.keys():
      entity2id[key] -= 1
    for key in predicate2id.keys():
      predicate2id[key] -= 1

    (self.entity2id,
     self.predicate2id) = (token_to_id.TokenToIdLayer(entity2id, oov_id=0),
                           token_to_id.TokenToIdLayer(predicate2id, oov_id=0))

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
            self.options.relation_hidden_units
        ])
    proposal_broadcast2 = tf.broadcast_to(tf.expand_dims(
        proposal_features, 2), [
            batch, max_n_proposal, max_n_proposal,
            self.options.relation_hidden_units
        ])

    relation_features = tf.concat([
        proposal_broadcast1,
        proposal_broadcast2,
    ], -1)
    return relation_features

  def _spatial_relation_feature(self, n_proposal, proposals):
    """Extracts semantic relation feature for pairs of proposal nodes.

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

    areas_1in2 = tf.expand_dims(
        box_ops.area_box1_in_box2(proposals_broadcast1, proposals_broadcast2),
        -1)
    areas_2in1 = tf.expand_dims(
        box_ops.area_box1_in_box2(proposals_broadcast2, proposals_broadcast1),
        -1)
    iou = tf.expand_dims(
        box_ops.iou(proposals_broadcast1, proposals_broadcast2), -1)

    relation_features = tf.concat([
        unary_features_broadcast1, unary_features_broadcast2, areas_1in2,
        areas_2in1, iou
    ], -1)
    return relation_features

  def _edge_scoring_helper(self, attn, logits):
    """Applies edge scoring.  """
    normalizer_fn = (tf.nn.sigmoid
                     if self.options.sigmoid_cross_entropy else tf.nn.softmax)

    if model_pb2.ATTENTION == self.options.edge_scoring:
      return attn

    if model_pb2.CLASSIFICATION == self.options.edge_scoring:
      return normalizer_fn(logits)

    if model_pb2.ATTENTION_x_CLASSIFICATION == self.options.edge_scoring:
      return tf.multiply(attn, normalizer_fn(logits))

    raise ValueError('Invalid edge scoring method %i.',
                     self.options.edge_scoring)

  def _multiple_relation_detection(self, n_proposal, proposals,
                                   proposal_features, proposal_top_k_mask):
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
    if proposal_top_k_mask is not None:
      proposal_masks = tf.multiply(proposal_masks, proposal_top_k_mask)

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
                                          biases_initializer=None,
                                          scope='{}/branch1'.format(scope)))
        logits_predicate_given_relation.append(
            weight * slim.fully_connected(relation_features,
                                          num_outputs=self.n_predicate,
                                          activation_fn=None,
                                          biases_initializer=None,
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
          scope="MED/proposal_branch1")
      logits_entity_given_proposal = slim.fully_connected(
          proposal_features,
          num_outputs=self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope="MED/proposal_branch2")

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
    # Ground-truth boxes can only be used for evaluation/visualization:
    # - `scene_graph/subject/box`
    # - `scene_graph/object/box`

    if self.is_training:
      inputs.pop('scene_graph/subject/box')
      inputs.pop('scene_graph/object/box')

    # Extract proposal features.
    # - proposals = [batch, max_n_proposal, 4].
    # - proposal_masks = [batch, max_n_proposal].
    # - proposal_features = [batch, max_n_proposal, dims].
    # - proposal_hiddens = [batch, max_n_proposal, feature_dims].
    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      # Add an additional hidden layer with leaky relu activation.
      proposal_hiddens = slim.fully_connected(
          proposal_features,
          num_outputs=self.options.entity_hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope="proposal/hidden")
      proposal_hiddens = slim.dropout(proposal_hiddens,
                                      self.options.dropout_keep_prob,
                                      is_training=self.is_training)

    # Multiple Entity Detection (MED).
    # - `score_proposal_given_entity` denotes the importance of region given
    #    the existence of the ground-truth entity `e`.
    #     * shape=[batch, max_n_proposal, n_entity].
    # - `logits_entity_given_entity` is the prediction of entity given
    #    the ground-truth entity, with the internal attention adjusted.
    #     * shape=[batch, n_entity, n_entity].
    (score_proposal_given_entity,
     logits_entity_given_entity) = self._multiple_entity_detection(
         n_proposal, proposal_hiddens)

    # Get the top-k proposal boxes and build the dense graph.
    # - `top_k_values` = [batch, entity_top_k]
    proposal_top_k_mask = None
    if self.options.HasField('entity_top_k'):
      proposal_scores = tf.reduce_max(score_proposal_given_entity, -1)
      top_k_values, _ = tf.math.top_k(proposal_scores,
                                      self.options.entity_top_k)
      threshold = tf.reduce_min(top_k_values, -1, keep_dims=True)
      proposal_top_k_mask = tf.cast(proposal_scores >= threshold, tf.float32)

    # Multiple Relation Detection (MRD).
    # - `score_relation_given_predicate` denotes the importance of relation
    #    given the existence of the ground-truth predicate `p`.
    #     * shape=[batch, max_n_proposal**2, n_predicate].
    # - `logits_predicate_given_predicate` is the prediction of predicate given
    #    the ground-truth predicate, with the internal attention adjusted.
    #     * shape=[batch, n_predicate, n_predicate].
    (score_relation_given_predicate,
     logits_predicate_given_predicate) = self._multiple_relation_detection(
         n_proposal, proposals, proposal_hiddens, proposal_top_k_mask)

    # Parse the image-level scene-graph.
    n_triple = inputs['scene_graph/n_triple']
    subject_ids = self.entity2id(inputs['scene_graph/subject'])
    object_ids = self.entity2id(inputs['scene_graph/object'])
    predicate_ids = self.predicate2id(inputs['scene_graph/predicate'])
    max_n_triple = tf.shape(subject_ids)[1]

    # Get the selection indices. Note: `index = id - 1`.
    # - `batch_index` = [batch, max_n_triple],
    # - `subject_index` = [batch, max_n_triple, 2].
    # - `object_index` = [batch, max_n_triple, 2].
    # - `predicate_index` = [batch, max_n_triple, 2].
    batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_n_triple])
    subject_index = tf.stack([batch_index, subject_ids], -1)
    object_index = tf.stack([batch_index, object_ids], -1)
    predicate_index = tf.stack([batch_index, predicate_ids], -1)

    # Get the image-level prediction.
    # - `subject_logits` = [batch, max_n_triple, n_entity].
    # - `object_logits` = [batch, max_n_triple, n_entity].
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

    subject_to_proposal = tf.multiply(1.0, subject_to_proposal)
    proposal_to_proposal = tf.multiply(1.0, proposal_to_proposal)
    proposal_to_object = tf.multiply(1.0, proposal_to_object)
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

    tf.summary.histogram('edge_weight/edge_weight_subject_to_proposal',
                         edge_weight_subject_to_proposal)
    tf.summary.histogram('edge_weight/edge_weight_proposal_to_object',
                         edge_weight_proposal_to_object)
    tf.summary.histogram('edge_weight/edge_weight_proposal_to_proposal',
                         edge_weight_proposal_to_proposal)

    predictions = {
        'pseudo/subject/ids':
            subject_ids,
        'pseudo/subject/logits':
            subject_logits,
        'pseudo/subject/box':
            ps_subject_box,
        'pseudo/object/ids':
            object_ids,
        'pseudo/object/logits':
            object_logits,
        'pseudo/object/box':
            ps_object_box,
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
        'pseudo/mps_path/sum':
            max_path_sum,
        'pseudo/mps_path/subject_to_proposal':
            edge_weight_subject_to_proposal,
        'pseudo/mps_path/proposal_to_proposal':
            edge_weight_proposal_to_proposal,
        'pseudo/mps_path/proposal_to_object':
            edge_weight_proposal_to_object,
        'proposal_top_k_mask':
            proposal_top_k_mask,
    }

    return predictions

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    max_path_sum = predictions['pseudo/mps_path/sum']

    subject_labels = predictions['pseudo/subject/ids']
    object_labels = predictions['pseudo/object/ids']
    predicate_labels = predictions['pseudo/predicate/ids']

    subject_logits = predictions['pseudo/subject/logits']
    object_logits = predictions['pseudo/object/logits']
    predicate_logits = predictions['pseudo/predicate/logits']

    n_triple = inputs['scene_graph/n_triple']

    max_n_triple = tf.shape(subject_labels)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    # Compute path-sum loss.
    mean_max_path_sum = tf.reduce_mean(
        masked_ops.masked_avg(max_path_sum, mask=triple_mask, dim=1))
    path_sum_loss = -mean_max_path_sum

    # Compute MED/MRD losses.
    if self.options.sigmoid_cross_entropy:
      triple_mask = tf.expand_dims(triple_mask, -1)
      subject_labels = tf.one_hot(subject_labels, self.n_entity)
      object_labels = tf.one_hot(object_labels, self.n_entity)
      predicate_labels = tf.one_hot(predicate_labels, self.n_predicate)

      subject_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=subject_labels, logits=subject_logits)
      object_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=object_labels, logits=object_logits)
      predicate_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=predicate_labels, logits=predicate_logits)

      subject_loss = tf.reduce_mean(
          masked_ops.masked_avg(subject_losses, triple_mask, dim=1))
      object_loss = tf.reduce_mean(
          masked_ops.masked_avg(object_losses, triple_mask, dim=1))
      predicate_loss = tf.reduce_mean(
          masked_ops.masked_avg(predicate_losses, triple_mask, dim=1))

    else:

      subject_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=subject_labels, logits=subject_logits)
      object_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=object_labels, logits=object_logits)
      predicate_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=predicate_labels, logits=predicate_logits)

      subject_loss = tf.reduce_mean(
          masked_ops.masked_avg(subject_losses, triple_mask, dim=1))
      object_loss = tf.reduce_mean(
          masked_ops.masked_avg(object_losses, triple_mask, dim=1))
      predicate_loss = tf.reduce_mean(
          masked_ops.masked_avg(predicate_losses, triple_mask, dim=1))

    global_step = tf.compat.v1.train.get_or_create_global_step()
    mrd_indicator = tf.cast(
        tf.greater_equal(global_step, self.options.mrd_starting_step),
        tf.float32)

    return {
        'metrics/path_sum_loss':
            self.options.path_sum_loss_weight * path_sum_loss,
        'metrics/med_subject_loss':
            self.options.med_loss_weight * subject_loss,
        'metric/med_object_loss':
            self.options.med_loss_weight * object_loss,
        'metric/mrd_predicate_loss':
            mrd_indicator * self.options.mrd_loss_weight * predicate_loss,
    }

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
    max_path_sum = predictions['pseudo/mps_path/sum']
    gt_subject_box = inputs['scene_graph/subject/box']
    gt_object_box = inputs['scene_graph/object/box']

    ps_subject_box = predictions['pseudo/subject/box']
    ps_object_box = predictions['pseudo/object/box']

    n_triple = inputs['scene_graph/n_triple']

    max_n_triple = tf.shape(ps_subject_box)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    # Compute IoU of the subject and object boxes.
    subject_iou = box_ops.iou(gt_subject_box, ps_subject_box)
    object_iou = box_ops.iou(gt_object_box, ps_object_box)

    _mean_fn = lambda x: tf.reduce_mean(
        masked_ops.masked_avg(tf.cast(x, tf.float32), mask=triple_mask, dim=1))

    mean_subject_iou = tf.keras.metrics.Mean()
    mean_object_iou = tf.keras.metrics.Mean()

    mean_subject_iou.update_state(_mean_fn(subject_iou))
    mean_object_iou.update_state(_mean_fn(object_iou))

    metric_dict = {
        'metrics/pseudo/subject_iou': mean_subject_iou,
        'metrics/pseudo/object_iou': mean_object_iou,
    }

    # Path-sum.
    mean_max_path_sum = tf.keras.metrics.Mean()
    mean_max_path_sum.update_state(_mean_fn(max_path_sum))
    metric_dict.update({
        'metrics/max_path_sum': mean_max_path_sum,
    })

    # Compute the classification.
    subject_labels = predictions['pseudo/subject/ids']
    object_labels = predictions['pseudo/object/ids']
    predicate_labels = predictions['pseudo/predicate/ids']

    subject_logits = predictions['pseudo/subject/logits']
    object_logits = predictions['pseudo/object/logits']
    predicate_logits = predictions['pseudo/predicate/logits']

    correct_subject = tf.equal(tf.cast(tf.argmax(subject_logits, -1), tf.int32),
                               subject_labels)
    correct_object = tf.equal(tf.cast(tf.argmax(object_logits, -1), tf.int32),
                              object_labels)
    correct_relation = tf.equal(
        tf.cast(tf.argmax(predicate_logits, -1), tf.int32), predicate_labels)
    subject_accuracy = tf.keras.metrics.Mean()
    object_accuracy = tf.keras.metrics.Mean()
    predicate_accuracy = tf.keras.metrics.Mean()

    subject_accuracy.update_state(_mean_fn(correct_subject))
    object_accuracy.update_state(_mean_fn(correct_object))
    predicate_accuracy.update_state(_mean_fn(correct_relation))
    metric_dict.update({
        'metrics/predict_subject': subject_accuracy,
        'metrics/predict_object': object_accuracy,
        'metrics/predict_predicate': predicate_accuracy,
    })

    # Compute accuracy at different IoU level.
    for iou_threshold in [0.25, 0.5, 0.75]:
      correct_subject = tf.greater_equal(subject_iou, iou_threshold)
      correct_object = tf.greater_equal(object_iou, iou_threshold)
      correct_relation = tf.logical_and(correct_subject, correct_object)

      mean_subject_recall = tf.keras.metrics.Mean()
      mean_object_recall = tf.keras.metrics.Mean()
      mean_relation_recall = tf.keras.metrics.Mean()

      mean_subject_recall.update_state(_mean_fn(correct_subject))
      mean_object_recall.update_state(_mean_fn(correct_object))
      mean_relation_recall.update_state(_mean_fn(correct_relation))

      metric_dict.update({
          'metrics/pseudo/subject_recall@%.2lf' % (iou_threshold):
              mean_subject_recall,
          'metrics/pseudo/object_recall@%.2lf' % (iou_threshold):
              mean_object_recall,
          'metrics/pseudo/relation_recall@%.2lf' % (iou_threshold):
              mean_relation_recall,
      })

    return metric_dict
