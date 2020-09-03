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


class MaxPathSumV2(model_base.ModelBase):
  """MaxPathSumV2 model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    super(MaxPathSumV2, self).__init__(options, is_training)

    if not isinstance(options, model_pb2.MaxPathSumV2):
      raise ValueError('Options has to be an MaxPathSumV2 proto.')

    # Load token2id mapping and pre-trained word embedding weights.
    with tf.io.gfile.GFile(options.token_to_id_meta_file, 'r') as fid:
      meta = json.load(fid)
      (entity2id, predicate2id) = (meta['label_to_idx'],
                                   meta['predicate_to_idx'])
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

  def _trim_proposals_to_limit(self, inputs, limit):
    """Trims proposals to the `limit` denoted.

    Args:
      inputs: A dictionary of input tensors keyed by names.
        - `image/n_proposal`: A [batch] int32 tensor.
        - `image/proposal`: A [batch, max_n_proposal, 4] float tensor.
        - `image/proposal/feature`: A [batch, max_proposal, feature_dims] float tensor.
      limit:
        Maximum number of proposals to be used.

    Returns:
      Same `inputs` object.
    """
    inputs.update({
        'image/n_proposal':
            tf.minimum(inputs['image/n_proposal'], self.options.max_n_proposal),
        'image/proposal':
            inputs['image/proposal'][:, :self.options.max_n_proposal, :],
        'image/proposal/feature':
            inputs['image/proposal/feature']
            [:, :self.options.max_n_proposal, :],
    })
    return inputs

  def _multiple_relation_detection(self, n_proposal, proposals,
                                   proposal_features):
    """Detects multiple relations given a pair of proposals.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      logits_predicate_given_relation: A batch, max_n_proposal, max_n_proposal,
        n_predicate] tensor.
    """
    batch = proposal_features.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      # Add an additional hidden layer to reduce feature dimensions.
      proposal_hiddens = slim.fully_connected(
          proposal_features,
          num_outputs=self.options.hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope="MRD/proposal_hidden")
      proposal_hiddens = slim.dropout(proposal_hiddens,
                                      self.options.dropout_keep_prob,
                                      is_training=self.is_training)

    # Compute relation feature.
    # - relation_features = [batch, max_n_proposal, max_n_proposal, dims].
    proposal_broadcast1 = tf.broadcast_to(
        tf.expand_dims(proposal_hiddens, 1),
        [batch, max_n_proposal, max_n_proposal, self.options.hidden_units])
    proposal_broadcast2 = tf.broadcast_to(
        tf.expand_dims(proposal_hiddens, 2),
        [batch, max_n_proposal, max_n_proposal, self.options.hidden_units])

    relation_features = proposal_broadcast1 + proposal_broadcast2

    with slim.arg_scope(self.arg_scope_fn()):
      weights_initializer = tf.compat.v1.constant_initializer(
          self.predicate_emb_weights.transpose())
      logits_predicate_given_relation = slim.fully_connected(
          relation_features,
          num_outputs=self.n_predicate,
          activation_fn=None,
          weights_initializer=weights_initializer,
          scope='MRD/relation')

    return logits_predicate_given_relation

  def _multiple_entity_detection(self, n_proposal, proposal_features):
    """Detects multiple entities given the ground-truth.

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      logits_entity_given_proposal: [batch, max_n_proposal, n_entity].
    """
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      # Add an additional hidden layer with leaky relu activation.
      # - Note: the complexity is only applied to localization branch.
      proposal_hiddens = slim.fully_connected(
          proposal_features,
          num_outputs=self.options.hidden_units,
          activation_fn=tf.nn.leaky_relu,
          scope="MED/proposal_hidden")
      proposal_hiddens = slim.dropout(proposal_hiddens,
                                      self.options.dropout_keep_prob,
                                      is_training=self.is_training)

      weights_initializer = tf.compat.v1.constant_initializer(
          self.entity_emb_weights.transpose())
      logits_entity_given_proposal = slim.fully_connected(
          proposal_hiddens,
          num_outputs=self.n_entity,
          activation_fn=None,
          weights_initializer=weights_initializer,
          biases_initializer=None,
          scope="MED/entity")

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
    # Ground-truth boxes can only be used for evaluation/visualization:
    # - `scene_graph/subject/box`
    # - `scene_graph/object/box`

    if self.is_training:
      inputs.pop('scene_graph/subject/box')
      inputs.pop('scene_graph/object/box')

    # Trim proposals to the limit number.
    # - proposals = [batch, max_n_proposal, 4].
    # - proposal_masks = [batch, max_n_proposal].
    # - proposal_features = [batch, max_n_proposal, feature_dims].
    self._trim_proposals_to_limit(inputs, limit=self.options.max_n_proposal)

    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    # Multiple Entity Detection (MED).
    # - `logits_entity_given_proposal` = [batch, max_n_propoosal, n_entity].
    logits_entity_given_proposal = self._multiple_entity_detection(
        n_proposal, proposal_features)

    # Multiple Relation detection network.
    # - `logits_predicate_given_relation` = [batch, max_n_propoosal,
    #    max_n_proposal, n_predicate].
    logits_predicate_given_relation = self._multiple_relation_detection(
        n_proposal, proposals, proposal_features)

    # Parse the image-level scene-graph.
    n_triple = inputs['scene_graph/n_triple']
    subject_ids = self.entity2id(inputs['scene_graph/subject'])
    object_ids = self.entity2id(inputs['scene_graph/object'])
    predicate_ids = self.predicate2id(inputs['scene_graph/predicate'])

    max_n_triple = tf.shape(subject_ids)[1]
    triple_masks = tf.sequence_mask(n_triple,
                                    maxlen=max_n_triple,
                                    dtype=tf.float32)

    # Get the selection indices. Note: `index = id - 1`.
    # - `batch_index` = [batch, max_n_triple],
    # - `subject_index` = [batch, max_n_triple, 2].
    # - `object_index` = [batch, max_n_triple, 2].
    # - `predicate_index` = [batch, max_n_triple, 2].
    batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_n_triple])
    subject_index = tf.stack([batch_index, tf.maximum(0, subject_ids - 1)], -1)
    object_index = tf.stack([batch_index, tf.maximum(0, object_ids - 1)], -1)
    predicate_index = tf.stack(
        [batch_index, tf.maximum(0, predicate_ids - 1)], -1)

    # Create the graph to run max-path-sum.
    # - `subject_to_proposal` = [batch, max_n_triple, max_n_proposal].
    # - `proposal_to_object` = [batch, max_n_triple, max_n_proposal].
    # - `proposal_to_proposal` = [batch, max_n_triple, max_n_proposal**2].

    subject_to_proposal = tf.gather_nd(
        tf.transpose(logits_entity_given_proposal, [0, 2, 1]), subject_index)
    proposal_to_object = tf.gather_nd(
        tf.transpose(logits_entity_given_proposal, [0, 2, 1]), object_index)

    logits_predicate_given_relation_reshaped = tf.reshape(
        logits_predicate_given_relation,
        [batch, max_n_proposal * max_n_proposal, self.n_predicate])
    proposal_to_proposal = tf.gather_nd(
        tf.transpose(logits_predicate_given_relation_reshaped, [0, 2, 1]),
        predicate_index)
    proposal_to_proposal = tf.reshape(
        proposal_to_proposal,
        [batch, max_n_triple, max_n_proposal, max_n_proposal])

    # DP solution of max-path-sum.
    (max_path_sum, ps_subject_proposal_i,
     ps_object_proposal_j) = utils.compute_max_path_sum(
         n_proposal, n_triple, tf.nn.sigmoid(subject_to_proposal),
         tf.nn.sigmoid(proposal_to_proposal), tf.nn.sigmoid(proposal_to_object))

    ps_subject_box = utils.gather_proposal_by_index(proposals,
                                                    ps_subject_proposal_i)
    ps_object_box = utils.gather_proposal_by_index(proposals,
                                                   ps_object_proposal_j)

    ps_subject_logits = utils.gather_proposal_by_index(
        logits_entity_given_proposal, ps_subject_proposal_i)
    ps_object_logits = utils.gather_proposal_by_index(
        logits_entity_given_proposal, ps_object_proposal_j)

    ps_predicate_logits = utils.gather_relation_by_index(
        logits_predicate_given_relation, ps_subject_proposal_i,
        ps_object_proposal_j)

    predictions = {
        'midn/subject/ids': subject_ids,
        'midn/subject/logits': ps_subject_logits,
        'midn/subject/box': ps_subject_box,
        'midn/object/ids': object_ids,
        'midn/object/logits': ps_object_logits,
        'midn/object/box': ps_object_box,
        'midn/predicate/ids': predicate_ids,
        'midn/predicate/logits': ps_predicate_logits,
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
    subject_ids = predictions['midn/subject/ids']
    object_ids = predictions['midn/object/ids']
    predicate_ids = predictions['midn/predicate/ids']

    subject_logits = predictions['midn/subject/logits']
    object_logits = predictions['midn/object/logits']
    predicate_logits = predictions['midn/predicate/logits']

    n_triple = inputs['scene_graph/n_triple']

    max_n_triple = tf.shape(subject_ids)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    # Compute MED/MRD losses.
    subject_labels = tf.maximum(0, subject_ids - 1)
    object_labels = tf.maximum(0, object_ids - 1)
    predicate_labels = tf.maximum(0, predicate_ids - 1)

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
        masked_ops.masked_avg(subject_losses, triple_mask, dim=1))
    predicate_loss = tf.reduce_mean(
        masked_ops.masked_avg(predicate_losses, triple_mask, dim=1))

    return {
        'metrics/med_subject_loss':
            self.options.med_loss_weight * subject_loss,
        'metric/med_object_loss':
            self.options.med_loss_weight * object_loss,
        'metric/mrd_predicate_loss':
            self.options.mrd_loss_weight * predicate_loss,
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
    gt_subject_box = inputs['scene_graph/subject/box']
    gt_object_box = inputs['scene_graph/object/box']

    ps_subject_box = predictions['midn/subject/box']
    ps_object_box = predictions['midn/object/box']

    n_triple = inputs['scene_graph/n_triple']

    max_n_triple = tf.shape(ps_subject_box)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    # Compute IoU of the subject and object boxes.
    subject_iou = box_ops.iou(gt_subject_box, ps_subject_box)
    object_iou = box_ops.iou(gt_object_box, ps_object_box)

    _mean_fn = lambda x: tf.reduce_mean(
        masked_ops.masked_avg(x, mask=triple_mask, dim=1))

    mean_subject_iou = tf.keras.metrics.Mean()
    mean_object_iou = tf.keras.metrics.Mean()

    mean_subject_iou.update_state(_mean_fn(subject_iou))
    mean_object_iou.update_state(_mean_fn(object_iou))

    return {
        'metrics/iou_subject': mean_subject_iou,
        'metrics/iou_object': mean_object_iou,
    }
