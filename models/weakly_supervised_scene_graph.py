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
    (self.entity2id,
     self.predicate2id) = (token_to_id.TokenToIdLayer(entity2id, oov_id=0),
                           token_to_id.TokenToIdLayer(predicate2id, oov_id=0))

    self.n_entity = len(entity2id)
    self.n_predicate = len(predicate2id)

    logging.info('#Entity=%s, #Predicate=%s', self.n_entity, self.n_predicate)

    # Initialize the arg_scope for FC and CONV layers.
    self.arg_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                      is_training)

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

  def _multiple_instance_detection(self, n_proposal, proposal_features):
    """Detects multiple instances given the ground-truth.

      MIDN network:
      - attn_r_given_e = [batch, max_n_proposal(r), n_entity(e-gt)].
          * Given that entity e-gt exists, find the responsible proposal r.
          * For each e-gt: sum_r (proba_r_given_e) = 1.
      - logits_e_given_r = [batch, max_n_proposal(r), n_entity(e-pred)],
          * Logits to classify proposal r.
      - logits_e_given_e = [batch, n_entity(e-gt), n_entity(e-pred)].
          * Given the entity e-gt, predict the logits through optimizing the
            weighting `attn_r_given_e`.

    Args:
      n_proposal: A [batch] int tensor.
      proposal_features: A [batch, max_n_proposal, feature_dims] float tensor.

    Returns:
      attn_r_given_e: Attention distribution of proposals given the entities.
        shape=[batch, max_n_proposal, n_entity]
      logits_e_given_e: Entity prediction.
        shape=[batch, n_entity, n_entity]
    """
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_masks = tf.sequence_mask(n_proposal,
                                      maxlen=max_n_proposal,
                                      dtype=tf.float32)

    with slim.arg_scope(self.arg_scope_fn()):
      # Add an additional hidden layer with leaky relu activation.
      proposal_features = slim.fully_connected(proposal_features,
                                               num_outputs=300,
                                               activation_fn=tf.nn.leaky_relu,
                                               scope="proposal_hidden")
      proposal_features = slim.dropout(proposal_features,
                                       self.options.dropout_keep_prob,
                                       is_training=self.is_training)

      # Two branches.
      logits_r_given_e = slim.fully_connected(proposal_features,
                                              num_outputs=self.n_entity,
                                              activation_fn=None,
                                              scope="proposal_branch1")
      logits_e_given_r = slim.fully_connected(proposal_features,
                                              num_outputs=self.n_entity,
                                              activation_fn=None,
                                              scope="proposal_branch2")
    attn_r_given_e = masked_ops.masked_softmax(logits_r_given_e,
                                               mask=tf.expand_dims(
                                                   proposal_masks, -1),
                                               dim=1)
    logits_e_given_e = tf.matmul(attn_r_given_e,
                                 logits_e_given_r,
                                 transpose_a=True)
    return attn_r_given_e, logits_e_given_e

  def _predict_relation_embeddings(self, proposal_embs, output_dims, scope):
    """Predicts edge embeddings.

    `edge_emb = linear(concat(node1_emb, node2_emb))`

    Args:
      proposal_embs: A [batch, max_n_proposal, dims] float tensor.

    Returns:
      A [batch, max_n_proposal, max_n_proposal, output_dims] float tensor.
    """
    (batch, dims) = (proposal_embs.shape[0].value,
                     proposal_embs.shape[-1].value)
    max_n_proposal = tf.shape(proposal_embs)[1]

    proposal_embs_1 = tf.broadcast_to(tf.expand_dims(
        proposal_embs, 1), [batch, max_n_proposal, max_n_proposal, dims])
    proposal_embs_2 = tf.broadcast_to(tf.expand_dims(
        proposal_embs, 2), [batch, max_n_proposal, max_n_proposal, dims])

    # edge_embs = tf.concat([proposal_embs_1, proposal_embs_2], -1)
    edge_embs = tf.concat(
        [proposal_embs_1, proposal_embs_2, proposal_embs_1 * proposal_embs_2],
        -1)
    with slim.arg_scope(self.arg_scope_fn()):
      edge_embs = slim.fully_connected(edge_embs,
                                       num_outputs=output_dims,
                                       activation_fn=None,
                                       scope=scope)
    return edge_embs

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

    # MIDN network.
    # - `attn_r_given_e` = [batch, max_n_proposal, n_entity].
    # - `logits_e_given_e` = [batch, n_entity, n_entity]
    (attn_r_given_e, logits_e_given_e) = self._multiple_instance_detection(
        n_proposal, proposal_features)

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
    batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_n_triple])
    subject_index = tf.stack([batch_index, tf.maximum(0, subject_ids - 1)], -1)
    object_index = tf.stack([batch_index, tf.maximum(0, object_ids - 1)], -1)

    # Get the MIDN prediction.
    # - `logits_e_given_e` = [batch, n_entity, n_entity]
    # - `subject_logits` = [batch, max_n_triple, n_entity].
    # - `object_logits` = [batch, max_n_triple, n_entity].

    subject_logits = tf.gather_nd(logits_e_given_e, subject_index)
    object_logits = tf.gather_nd(logits_e_given_e, object_index)

    # Get the pseudo boxes.
    # - `ps_subject_box_attn` = [batch, max_n_triple, max_n_proposal].
    # - `ps_object_box_attn` = [batch, max_n_triple, max_n_proposal].

    ps_subject_box_attn = tf.gather_nd(tf.transpose(attn_r_given_e, [0, 2, 1]),
                                       subject_index)
    ps_object_box_attn = tf.gather_nd(tf.transpose(attn_r_given_e, [0, 2, 1]),
                                      object_index)

    proposal_masks_expanded = tf.expand_dims(proposal_masks, 1)
    ps_subject_box_index = tf.cast(
        masked_ops.masked_argmax(ps_subject_box_attn,
                                 proposal_masks_expanded,
                                 dim=2), tf.int32)
    ps_object_box_index = tf.cast(
        masked_ops.masked_argmax(ps_object_box_attn,
                                 proposal_masks_expanded,
                                 dim=2), tf.int32)
    ps_subject_box = utils.gather_proposal_by_index(proposals,
                                                    ps_subject_box_index)
    ps_object_box = utils.gather_proposal_by_index(proposals,
                                                   ps_object_box_index)

    predictions = {
        'midn/subject/ids': subject_ids,
        'midn/subject/logits': subject_logits,
        'midn/subject/box': ps_subject_box,
        'midn/object/ids': object_ids,
        'midn/object/logits': object_logits,
        'midn/object/box': ps_object_box,
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
    (subject_ids, object_ids, subject_logits,
     object_logits) = (predictions['midn/subject/ids'],
                       predictions['midn/object/ids'],
                       predictions['midn/subject/logits'],
                       predictions['midn/subject/logits'])
    n_triple = inputs['scene_graph/n_triple']

    max_n_triple = tf.shape(subject_ids)[1]
    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)

    # Compute MIDN losses.
    subject_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.maximum(0, subject_ids - 1), logits=subject_logits)
    object_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.maximum(0, object_ids - 1), logits=object_logits)

    subject_loss = tf.reduce_mean(
        masked_ops.masked_avg(subject_losses, triple_mask, dim=1))
    object_loss = tf.reduce_mean(
        masked_ops.masked_avg(subject_losses, triple_mask, dim=1))

    return {
        'metrics/midn_subject_loss': subject_loss,
        'metric/midn_object_loss': object_loss
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

    # Compute IoU and max-path-sum metric ops.
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
