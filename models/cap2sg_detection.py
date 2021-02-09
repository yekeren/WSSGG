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
from modeling.utils import box_ops

from models import model_base
from models import cap2sg_grounding

from bert.modeling import transformer_model

sigmoid_focal_crossentropy = cap2sg_grounding.sigmoid_focal_crossentropy


def predict_classes(detection_head, embeddings, bias, normalize_fn):
  """Predicts classes based on the detection head.

  Args:
    detection_head: A [batch, max_n_proposal, dims] float tensor.
    embeddings: Embedding matrix, a [vocab_size, dims] float tensor.
    bias: A [vocab_size] float tensor.
    normalize_fn: Function to normalize scores.

  Returns:
    detection_logits: Logits tensor, a [batch, max_n_proposal, vocab_size] 
      float tensor.
    detection_scores: Normalized scores, a [batch, max_n_proposal, vocab_size] 
      float tensor.
  """
  detection_logits = tf.matmul(detection_head, embeddings, transpose_b=True)
  detection_logits = tf.nn.bias_add(detection_logits, bias)
  detection_scores = normalize_fn(detection_logits)

  # Set background scores to zeros; note the logits still contain background.
  batch = detection_head.shape[0].value
  max_n_proposal = tf.shape(detection_head)[1]
  detection_scores = tf.concat(
      [tf.zeros([batch, max_n_proposal, 1]), detection_scores[:, :, 1:]], -1)

  return detection_logits, detection_scores


def compute_iou(n_box1, box1, n_box2, box2):
  """Computes the IoU between two sets of boxes.

  Args:
    n_box1: A [batch] int tensor.
    box1: A [batch, max_n_box1, 4] float tensor.
    n_box2: A [batch] int tensor.
    box2: A [batch, max_n_box2, 4] float tensor.

  Returns:
    iou: A [batch, max_n_box1, max_n_box2] float tensor.
  """
  mask1 = tf.sequence_mask(n_box1, maxlen=tf.shape(box1)[1], dtype=tf.float32)
  mask2 = tf.sequence_mask(n_box2, maxlen=tf.shape(box2)[1], dtype=tf.float32)
  mask = tf.multiply(tf.expand_dims(mask1, 2), tf.expand_dims(mask2, 1))

  iou = box_ops.iou(tf.expand_dims(box1, 2), tf.expand_dims(box2, 1))
  return tf.multiply(iou, mask)


def scatter_entity_labels(proposal_id, entity_id, max_n_proposal, vocab_size):
  """Creates entity labels from pseudo instances.

  Args:
    proposal_id: A [batch, max_n_node] int tensor, denoting the proposal index.
    entity_id: A [batch, max_n_node] int tensor, values are in [0, vocab_size).
    max_n_proposal: Maximum number of proposals.
    vocab_size: Size of the vocabulary.

  Returns:
    A [batch, max_n_proposal, vocab_size] tensor.
  """
  batch = proposal_id.shape[0].value
  max_n_node = tf.shape(entity_id)[1]

  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_node])
  index_full = tf.stack([index_batch, proposal_id, entity_id], -1)
  return tf.scatter_nd(index_full,
                       updates=tf.fill([batch, max_n_node], 1.0),
                       shape=[batch, max_n_proposal, vocab_size])


def scatter_attribute_labels(proposal_id, attribute_id, max_n_proposal,
                             vocab_size):
  """Create attribute labels from pseudo instances.

  Args:
    proposal_id: A [batch, max_n_node] int tensor, denoting the proposal index.
    attribute_id: A [batch, max_n_node, max_n_attribute] int tensor, values are in [0, vocab_size).
    max_n_proposal: Maximum number of proposals.
    vocab_size: Size of the vocabulary.

  Returns:
    A [batch, max_n_proposal, vocab_size] tensor.
  """
  batch = proposal_id.shape[0].value
  max_n_node = tf.shape(proposal_id)[1]

  attribute_labels = tf.reduce_max(tf.one_hot(attribute_id, depth=vocab_size),
                                   2)
  attribute_labels = tf.concat(
      [tf.zeros([batch, max_n_node, 1]), attribute_labels[:, :, 1:]], -1)
  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_node])
  return tf.scatter_nd(tf.stack([index_batch, proposal_id], -1),
                       updates=attribute_labels,
                       shape=[batch, max_n_proposal, vocab_size])


def post_process_detection_labels(detection_labels, normalize=True):
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


class Cap2SGDetection(model_base.ModelBase):
  """Cap2SGDetection model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    """Constructs the Cap2SGDetection instance. """
    if not isinstance(options, model_pb2.Cap2SGDetection):
      raise ValueError('Options has to be an Cap2SGDetection proto.')
    super(Cap2SGDetection, self).__init__(options, is_training)

    self.grounding_model = cap2sg_grounding.Cap2SGGrounding(
        options.grounding_config, is_training)

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
    predictions = self.grounding_model.predict(inputs)

    n_proposal, proposals, proposal_features = (
        inputs['image/n_proposal'], inputs['image/proposal'],
        inputs['image/proposal/feature'])
    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]
    proposal_mask = tf.sequence_mask(n_proposal,
                                     tf.shape(proposals)[1],
                                     dtype=tf.float32)

    # Predict detection_scores and attribute_scores.
    #   shape = [batch, max_n_proposa, vocab_size].
    normalize_fn = (tf.nn.softmax if model_pb2.LOSS_TYPE_SOFTMAX_CROSSENTROPY ==
                    self.options.loss_type else tf.nn.sigmoid)

    detection_head, attribute_head = [
        tf.layers.Dense(self.grounding_model.options.embedding_dims,
                        activation=tf.nn.leaky_relu,
                        use_bias=True,
                        name=name)(proposal_features)
        for name in ['entity_detection_head', 'attribute_detection_head']
    ]
    detection_logits, detection_scores = predict_classes(
        detection_head, self.grounding_model.embeddings,
        self.grounding_model.entity_bias, normalize_fn)
    attribute_logits, attribute_scores = predict_classes(
        attribute_head, self.grounding_model.embeddings,
        self.grounding_model.attribute_bias, normalize_fn)

    # Postprocess: non-maximum-suppression.
    (nmsed_boxes, nmsed_scores, nmsed_classes,
     valid_detections) = tf.image.combined_non_max_suppression(
         tf.expand_dims(proposals, 2),
         detection_scores[:, :, 1:],
         max_output_size_per_class=self.options.post_process.max_size_per_class,
         max_total_size=self.options.post_process.max_total_size,
         iou_threshold=self.options.post_process.iou_thresh,
         score_threshold=self.options.post_process.score_thresh)

    # Get the proposal id of the detection box, then fetch the attributes.
    #   nmsed_proposal shape = [batch, max_total_size, max_n_proposal].
    #   nmsed_attribute shape [batch, max_total_size, vocab_size]
    iou = compute_iou(valid_detections, nmsed_boxes, n_proposal, proposals)
    nmsed_proposal = tf.math.argmax(iou, axis=2, output_type=tf.int32)
    gather_nd_index = tf.stack([
        tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                        [batch, self.options.post_process.max_total_size]),
        nmsed_proposal
    ], -1)
    nmsed_attribute = tf.gather_nd(attribute_scores, gather_nd_index)
    nmsed_attribute_scores = tf.reduce_max(nmsed_attribute, -1)
    nmsed_attribute_classes = tf.argmax(nmsed_attribute,
                                        axis=2,
                                        output_type=tf.int32)

    # Compute entity_detection_scores (analogous to attention score in grouding).
    #   entity_detection_scores shape = [batch, max_n_node, max_n_proposal]
    index_entity = predictions['grounding/entity/id']
    index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, tf.shape(index_entity)[1]])
    entity_detection_scores = tf.gather_nd(
        tf.transpose(detection_scores, [0, 2, 1]),
        tf.stack([index_batch, index_entity], -1))

    # Update instance entity boxes.
    index_box = tf.math.argmax(entity_detection_scores,
                               axis=2,
                               output_type=tf.int32)

    predictions.update({
        'detection/detection_logits':
            detection_logits,
        'detection/attribute_logits':
            attribute_logits,
        # Detection grounding.
        'detection/entity/proposal_id':
            index_box,
        'detection/entity/proposal_box':
            tf.gather_nd(proposals, tf.stack([index_batch, index_box], -1)),
        'detection/entity/proposal_score':
            tf.reduce_max(entity_detection_scores, 2),
        # Detection results.
        'detection/num_detections':
            valid_detections,
        'detection/detection_boxes':
            nmsed_boxes,
        'detection/detection_scores':
            nmsed_scores,
        'detection/detection_classes':
            self.grounding_model.id2token(1 + tf.cast(nmsed_classes, tf.int32)),
        'detection/detection_attribute_scores':
            nmsed_attribute_scores,
        'detection/detection_attribute_classes':
            self.grounding_model.id2token(nmsed_attribute_classes),
    })
    return predictions

  def _compute_softmax_crossentropy(self,
                                    labels,
                                    logits,
                                    loss_mask,
                                    foreground_mask=False):
    """Computes the softmax crossentropy loss.

    Args:
      label: A [batch, max_n_proposal, vocab_size] float tensor.
      logits: A [batch, max_n_proposal, vocab_size] float tensor.
      loss_mask: A [batch, max_n_proposal] float tensor.
      foreground_mask: If true, apply foreground mask to the loss.

    Returns:
      loss: A scalar tensor.
    """
    # Normalize detection labels, manipulate `background` class.
    labels, foreground = post_process_detection_labels(labels, normalize=True)
    if foreground_mask:
      loss_mask = tf.multiply(foreground, loss_mask)

    # Compute the loss.
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)

    per_example_loss = masked_ops.masked_sum(losses, mask=loss_mask, dim=1)
    return tf.div(tf.reduce_sum(per_example_loss),
                  1e-6 + tf.reduce_sum(loss_mask))

  def _compute_sigmoid_crossentropy(self,
                                    labels,
                                    logits,
                                    loss_mask,
                                    focal_loss=False,
                                    foreground_mask=False):
    """Computes the sigmoid crossentropy loss.

    Args:
      label: A [batch, max_n_proposal, vocab_size] float tensor.
      logits: A [batch, max_n_proposal, vocab_size] float tensor.
      loss_mask: A [batch, max_n_proposal] float tensor.
      focal_loss: If true, use focal sigmoid crossentropy.
      foreground_mask: If true, apply foreground mask to the loss.

    Returns:
      loss: A scalar tensor.
    """
    # Postprocess detection labels, manipulate `background` class.
    labels, foreground = post_process_detection_labels(labels, normalize=False)
    if foreground_mask:
      loss_mask = tf.multiply(foreground, loss_mask)

    # Compute the loss.
    if focal_loss:
      losses = sigmoid_focal_crossentropy(labels, logits)
    else:
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)
      losses = tf.reduce_sum(losses, -1)

    per_example_loss = masked_ops.masked_sum(losses, mask=loss_mask, dim=1)
    return tf.div(tf.reduce_sum(per_example_loss),
                  1e-6 + tf.reduce_sum(loss_mask))

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    loss_dict = self.grounding_model.build_losses(inputs, predictions)

    (n_proposal, proposals, entity_proposal, entity_id, entity_attribute,
     detection_logits,
     attribute_logits) = (inputs['image/n_proposal'], inputs['image/proposal'],
                          predictions['grounding/entity/proposal_id'],
                          predictions['grounding/entity/id'],
                          predictions['grounding/entity/attribute_id'],
                          predictions['detection/detection_logits'],
                          predictions['detection/attribute_logits'])

    # Compute loss mask, i.e., proposal_mask.
    proposal_mask = tf.sequence_mask(n_proposal,
                                     tf.shape(proposals)[1],
                                     dtype=tf.float32)

    # Scatter entity labels.
    #   detection_labels shape = [batch, max_n_proposal, vocab_size].
    detection_labels = scatter_entity_labels(
        proposal_id=entity_proposal,
        entity_id=entity_id,
        max_n_proposal=tf.shape(proposals)[1],
        vocab_size=self.grounding_model.vocab_size)
    attribute_labels = scatter_attribute_labels(
        proposal_id=entity_proposal,
        attribute_id=entity_attribute,
        max_n_proposal=tf.shape(proposals)[1],
        vocab_size=self.grounding_model.vocab_size)

    # Compute the proposal pairwise IoU.
    #   iou shape = [batch, max_n_proposal, max_n_proposal].
    iou = compute_iou(n_proposal, proposals, n_proposal, proposals)

    # Maintain consistency, propogate detection_labels if IoU is greater than threshold.
    propogate_matrix = tf.cast(iou > self.options.grounding_iou_threshold,
                               tf.float32)
    detection_labels = tf.matmul(propogate_matrix, detection_labels)
    attribute_labels = tf.matmul(propogate_matrix, attribute_labels)

    if self.options.loss_type == model_pb2.LOSS_TYPE_SOFTMAX_CROSSENTROPY:
      detection_loss = self._compute_softmax_crossentropy(
          detection_labels, detection_logits, proposal_mask)
      attribute_loss = self._compute_softmax_crossentropy(attribute_labels,
                                                          attribute_logits,
                                                          proposal_mask,
                                                          foreground_mask=True)

    elif self.options.loss_type == model_pb2.LOSS_TYPE_SIGMOID_CROSSENTROPY:
      detection_loss = self._compute_sigmoid_crossentropy(detection_labels,
                                                          detection_logits,
                                                          proposal_mask,
                                                          focal_loss=False)
      attribute_loss = self._compute_sigmoid_crossentropy(attribute_labels,
                                                          attribute_logits,
                                                          proposal_mask,
                                                          focal_loss=False,
                                                          foreground_mask=True)

    elif self.options.loss_type == model_pb2.LOSS_TYPE_FOCAL_SIGMOID_CROSSENTROPY:
      detection_loss = self._compute_sigmoid_crossentropy(detection_labels,
                                                          detection_logits,
                                                          proposal_mask,
                                                          focal_loss=True)
      attribute_loss = self._compute_sigmoid_crossentropy(attribute_labels,
                                                          attribute_logits,
                                                          proposal_mask,
                                                          focal_loss=True,
                                                          foreground_mask=True)

    # Compute entropy loss.
    loss_dict.update({
        'detection/entity/loss':
            tf.multiply(self.options.loss_weight, detection_loss),
        'detection/attribute/loss':
            tf.multiply(self.options.loss_weight, attribute_loss),
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
