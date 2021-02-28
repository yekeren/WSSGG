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
from models import model_base

from modeling.utils import masked_ops
from modeling.utils import box_ops
from models.cap2sg_data import DataTuple
from models.cap2sg_preprocess import initialize
from models.cap2sg_grounding import ground_entities
from models.cap2sg_detection import detect_entities
from models.cap2sg_relation import detect_relations
from models.cap2sg_common_sense import apply_common_sense_refinement
from models.cap2sg_common_sense import train_common_sense_model

from model_utils.scene_graph_evaluation import SceneGraphEvaluator


class Cap2SG(model_base.ModelBase):
  """Cap2SG model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the Cap2SG instance. """
    if not isinstance(options, model_pb2.Cap2SG):
      raise ValueError('Options has to be an Cap2SG proto.')
    super(Cap2SG, self).__init__(options, is_training)

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
    dt = DataTuple()

    parse_attribute = (
        self.is_training and self.options.parse_attribute_in_training or
        not self.is_training and self.options.parse_attribute_in_evaluation)

    dt = initialize(self.options.preprocess_options, dt)
    dt = self.parse_data_fields(dt, inputs, parse_attribute)

    dt = ground_entities(self.options.grounding_options,
                         dt,
                         is_training=self.is_training)
    dt = detect_entities(self.options.detection_options, dt)
    dt = detect_relations(self.options.relation_options, dt)

    if self.options.HasField('common_sense_options'):
      dt = train_common_sense_model(self.options.common_sense_options, dt)
      dt = apply_common_sense_refinement(self.options.common_sense_options,
                                         dt,
                                         reuse=True)

    for k, v in vars(dt).items():
      print(k, v)

    predictions = {
        'grounding/entity/proposal_box':
            dt.grounding.entity_proposal_box,
        'grounding/entity/proposal_score':
            dt.grounding.entity_proposal_score,
        'detection/entity/proposal_box':
            dt.refined_grounding.entity_proposal_box,
        'detection/entity/proposal_score':
            dt.refined_grounding.entity_proposal_score,
        'detection/prediction/num_detections':
            dt.detection.valid_detections,
        'detection/prediction/detection_boxes':
            dt.detection.nmsed_boxes,
        'detection/prediction/detection_scores':
            dt.detection.nmsed_scores,
        'detection/prediction/detection_classes':
            dt.detection.nmsed_classes,
        # 'detection/prediction/detection_attribute_scores':
        #     dt.detection.nmsed_attribute_scores,
        # 'detection/prediction/detection_attribute_classes':
        #     dt.id2token_func(dt.detection.nmsed_attribute_classes),
        'relation/prediction/num_relations':
            dt.relation.num_relations,
        'relation/prediction/log_prob':
            dt.relation.log_prob,
        'relation/prediction/relation_score':
            dt.relation.relation_score,
        'relation/prediction/relation_class':
            dt.id2token_func(dt.relation.relation_class),
        'relation/prediction/subject_box':
            dt.relation.subject_box,
        'relation/prediction/subject_score':
            dt.relation.subject_score,
        'relation/prediction/subject_class':
            dt.id2token_func(dt.relation.subject_class),
        'relation/prediction/object_box':
            dt.relation.object_box,
        'relation/prediction/object_score':
            dt.relation.object_score,
        'relation/prediction/object_class':
            dt.id2token_func(dt.relation.object_class),
        'common_sense/prediction/num_relations':
            dt.refined_relation.num_relations,
        'common_sense/prediction/log_prob':
            dt.refined_relation.log_prob,
        'common_sense/prediction/relation_score':
            dt.refined_relation.relation_score,
        'common_sense/prediction/relation_class':
            dt.id2token_func(dt.refined_relation.relation_class),
        'common_sense/prediction/subject_box':
            dt.refined_relation.subject_box,
        'common_sense/prediction/subject_score':
            dt.refined_relation.subject_score,
        'common_sense/prediction/subject_class':
            dt.id2token_func(dt.refined_relation.subject_class),
        'common_sense/prediction/object_box':
            dt.refined_relation.object_box,
        'common_sense/prediction/object_score':
            dt.refined_relation.object_score,
        'common_sense/prediction/object_class':
            dt.id2token_func(dt.refined_relation.object_class),
    }
    self.data_tuple = dt
    return predictions

  def parse_data_fields(self, dt, inputs, parse_attribute=False):
    """Parses data fields from TF record.

    Args:
      dt: A DataTuple instance.
      inputs: A dictionary of input tensors keyed by names.
    """
    if not isinstance(dt, DataTuple):
      raise ValueError('Invalid DataTuple object.')

    # Visual proposal data fields.
    dt.n_proposal = inputs['image/n_proposal']
    dt.proposals = inputs['image/proposal']
    dt.proposal_features = inputs['image/proposal/feature']
    if self.is_training:
      dt.proposal_features = tf.nn.dropout(
          dt.proposal_features, keep_prob=self.options.dropout_keep_prob)

    dt.batch = dt.proposals.shape[0].value
    dt.max_n_proposal = tf.shape(dt.proposals)[1]
    dt.proposal_masks = tf.sequence_mask(dt.n_proposal,
                                         dt.max_n_proposal,
                                         dtype=tf.float32)

    # Proposal IoU.
    dt.proposal_iou = box_ops.iou(tf.expand_dims(dt.proposals, 2),
                                  tf.expand_dims(dt.proposals, 1))

    # Text graph entity data fields.
    (entities, per_ent_n_att,
     per_ent_atts) = parse_entity_and_attributes(inputs['caption_graph/nodes'],
                                                 parse_attribute)

    dt.n_entity = inputs['caption_graph/n_node']
    dt.entity_ids = dt.token2id_func(entities)
    dt.entity_embs = dt.embedding_func(dt.entity_ids)

    dt.max_n_entity = tf.shape(dt.entity_ids)[1]
    dt.entity_masks = tf.sequence_mask(dt.n_entity,
                                       dt.max_n_entity,
                                       dtype=tf.float32)
    dt.entity_image_labels = tf.one_hot(dt.entity_ids, depth=dt.vocab_size)

    # Text graph attribute data fields.
    dt.per_ent_n_att = per_ent_n_att
    dt.per_ent_att_ids = dt.token2id_func(per_ent_atts)
    dt.per_ent_att_embs = dt.embedding_func(dt.per_ent_att_ids)

    dt.attribute_image_labels = tf.reduce_max(
        tf.one_hot(dt.per_ent_att_ids, depth=dt.vocab_size), 2)

    # Text graph relation data fields.
    relations = inputs['caption_graph/edges']

    dt.n_relation = inputs['caption_graph/n_edge']
    dt.relation_ids = dt.token2id_func(relations)
    dt.relation_embs = dt.embedding_func(dt.relation_ids)

    dt.max_n_relation = tf.shape(dt.relation_ids)[1]
    dt.relation_masks = tf.sequence_mask(dt.n_relation,
                                         dt.max_n_relation,
                                         dtype=tf.float32)
    dt.relation_senders = inputs['caption_graph/senders']
    dt.relation_receivers = inputs['caption_graph/receivers']
    return dt

  def _compute_image_level_classification_loss(self, entity_masks, logits,
                                               labels):
    """Computes cross-entropy loss.

    Args:
      entity_masks: Entity masks, A [batch, max_n_entity] float tensor.
      logits: Entity logits, A [batch, max_n_entity, vocab_size - 1] float tensor.
      labels: Entity labels, A [batch, max_n_entity, vocab_size - 1] float tensor.

    Returns:
      A scalar loss tensor.
    """
    label_masks = tf.cast(tf.greater(tf.reduce_sum(labels, -1), 0), tf.float32)
    label_masks = tf.multiply(label_masks, entity_masks)

    # Normalize label and apply softmax cross-entropy loss.
    labels = tf.div(labels, 1e-6 + tf.reduce_sum(labels, -1, keepdims=True))
    per_entity_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                logits=logits)
    loss = tf.div(tf.reduce_sum(per_entity_losses * label_masks),
                  1e-6 + tf.reduce_sum(label_masks))
    return loss

  def _post_process_detection_labels(self, detection_labels, normalize=True):
    """Postprocesses the detection labels.
  
    Assuming the first token in the vocabulary is the `background`.
  
    Args:
      detection_labels: A [batch, max_n_proposal, vocab_size] tensor.
  
    Returns:
      detection_labels: A [batch, max_n_proposal, vocab_size] tensor.
      foreground_mask: A [batch, max_n_proposal] tensor.
    """
    # Remove the current invalid `background`.
    detection_labels = detection_labels[:, :, 1:]

    # `background` is assigned to proposals having no other labels.
    background = tf.cast(
        tf.greater(1e-6, tf.reduce_sum(detection_labels, -1, keepdims=True)),
        tf.float32)
    detection_labels = tf.concat([background, detection_labels], -1)

    # Normalize labels for training multi-class model.
    if normalize:
      detection_labels = tf.div(
          detection_labels, tf.reduce_sum(detection_labels, -1, keepdims=True))
    return detection_labels, tf.squeeze(1 - background, -1)

  def _compute_instance_level_detection_loss(self,
                                             loss_masks,
                                             logits,
                                             labels,
                                             foreground_mask=False):
    """Computes the softmax crossentropy loss.

    Args:
      loss_masks: A [batch, max_n_proposal] float tensor.
      logits: A [batch, max_n_proposal, vocab_size] float tensor.
      labels: A [batch, max_n_proposal, vocab_size] float tensor.
      foreground_mask: If true, apply foreground mask to the loss.

    Returns:
      loss: A scalar tensor.
    """
    # Normalize detection labels, manipulate `background` class.
    labels, foreground = self._post_process_detection_labels(labels,
                                                             normalize=True)
    if foreground_mask:
      loss_masks = tf.multiply(foreground, loss_masks)

    # Compute the loss.
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
    return tf.div(tf.reduce_sum(losses * loss_masks),
                  1e-6 + tf.reduce_sum(loss_masks))

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    dt = self.data_tuple

    loss_dict = {
        'grounding/entity/loss':
            self.options.grounding_options.loss_weight *
            self._compute_image_level_classification_loss(
                dt.entity_masks, dt.entity_image_logits[:, :, 1:],
                dt.entity_image_labels[:, :, 1:]),
        'grounding/attribute/loss':
            self.options.grounding_options.loss_weight *
            self._compute_image_level_classification_loss(
                dt.entity_masks, dt.attribute_image_logits[:, :, 1:],
                dt.attribute_image_labels[:, :, 1:]),
        # 'detection/entity/loss':
        #     self._compute_instance_level_detection_loss(
        #         dt.proposal_masks, dt.detection_instance_logits,
        #         dt.detection_instance_labels),
        # 'detection/attribute/loss':
        #     self._compute_instance_level_detection_loss(
        #         dt.proposal_masks,
        #         dt.attribute_instance_logits,
        #         dt.attribute_instance_labels,
        #         foreground_mask=True),
        'relation/relation/subject_loss':
            self.options.relation_options.loss_weight *
            self._compute_instance_level_detection_loss(
                dt.proposal_masks,
                dt.relation_subject_instance_logits,
                dt.relation_subject_instance_labels,
                foreground_mask=False),
        'relation/relation/object_loss':
            self.options.relation_options.loss_weight *
            self._compute_instance_level_detection_loss(
                dt.proposal_masks,
                dt.relation_object_instance_logits,
                dt.relation_object_instance_labels,
                foreground_mask=False),
    }

    for itno in range(self.options.detection_options.num_iterations):
      loss_dict.update({
          'detection/entity/loss_%i' % itno:
              self.options.detection_options.loss_weight *
              self._compute_instance_level_detection_loss(
                  dt.proposal_masks,
                  dt.detection_instance_logits_list[itno],
                  dt.detection_instance_labels_list[itno],
                  foreground_mask=False)
      })

    if self.options.HasField('common_sense_options'):
      loss_dict.update({
          'common_sense/relation/subject_loss':
              self.options.common_sense_options.loss_weight *
              self._compute_instance_level_detection_loss(
                  dt.relation_masks,
                  dt.subject_logits,
                  tf.one_hot(dt.subject_labels, dt.vocab_size),
                  foreground_mask=True),
          'common_sense/relation/object_loss':
              self.options.common_sense_options.loss_weight *
              self._compute_instance_level_detection_loss(
                  dt.relation_masks,
                  dt.object_logits,
                  tf.one_hot(dt.object_labels, dt.vocab_size),
                  foreground_mask=True),
          'common_sense/relation/predicate_loss':
              self.options.common_sense_options.loss_weight *
              self._compute_instance_level_detection_loss(
                  dt.relation_masks,
                  dt.predicate_logits,
                  tf.one_hot(dt.predicate_labels, dt.vocab_size),
                  foreground_mask=True),
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

    dt = self.data_tuple
    prefix, relation = 'initial_', dt.relation

    eval_dict = {
        'image_id': inputs['id'],
        'groundtruth/n_triple': inputs['scene_graph/n_relation'],
        'groundtruth/subject': inputs['scene_graph/subject'],
        'groundtruth/subject/box': inputs['scene_graph/subject/box'],
        'groundtruth/object': inputs['scene_graph/object'],
        'groundtruth/object/box': inputs['scene_graph/object/box'],
        'groundtruth/predicate': inputs['scene_graph/predicate'],
        'prediction/n_triple': relation.num_relations,
        'prediction/subject/box': relation.subject_box,
        'prediction/object/box': relation.object_box,
        'prediction/subject': dt.id2token_func(relation.subject_class),
        'prediction/object': dt.id2token_func(relation.object_class),
        'prediction/predicate': dt.id2token_func(relation.relation_class),
    }

    sg_evaluator = SceneGraphEvaluator()
    for k, v in sg_evaluator.get_estimator_eval_metric_ops(eval_dict).items():
      metric_dict['metrics/%s%s' % (prefix, k)] = v

    metric_dict['metrics/accuracy'] = metric_dict[
        'metrics/initial_scene_graph_per_image_recall@100']
    return metric_dict


def parse_entity_and_attributes(entity_and_attributes, parse_attribute=False):
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

  if parse_attribute:
    split_res = tf.sparse_tensor_to_dense(
        tf.strings.split(entity_and_attributes, sep=':', maxsplit=1), '')
    entity, attributes = split_res[:, :, 0], split_res[:, :, 1]
  else:
    entity = entity_and_attributes
    attributes = tf.zeros_like(entity, dtype=tf.string)

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
