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

from protos import model_pb2
from modeling.utils import masked_ops
from modeling.utils import box_ops
from models.cap2sg_data import DataTuple


def detect_entities(options, dt):
  """Grounds entities.

  Args:
    options: A Cap2SGDetection proto.
    dt: A DataTuple object, served as a collection of global variables.

  Returns:
    dt.detection.valid_detections: Number of detections, [batch].
    dt.detection.nmsed_boxes: Detection boxes, [batch, max_n_detection, 4].
    dt.detection.nmsed_scores: Detection scores, [batch, max_n_detection],
    dt.detection.nmsed_classes: Detection classes, [batch, max_n_detection],
    dt.detection.nmsed_attribute_scores: Object attribute scores, [batch, max_n_detection].
    dt.detection.nmsed_attribute_classes: Object attribute class, [batch_max_n_detection].
    dt.refined_grounding.entity_proposal_id: Grounding vector, the ``g'' vector in our paper, [batch, max_n_entity].
    dt.refined_grounding.entity_proposal_box: Grounded boxes, [batch, max_n_entity, 4],
    dt.refined_grounding.entity_proposal_score: Grounding scores, [batch, max_n_entity]
    dt.refined_grounding.entity_proposal_feature: Grounded features, [batch, max_n_entity, vdims]
  """
  if not isinstance(options, model_pb2.Cap2SGDetection):
    raise ValueError('Options has to be a Cap2SGDetection proto.')

  if not isinstance(dt, DataTuple):
    raise ValueError('Invalid DataTuple object.')

  # Compute proposal iou.
  propogation_matrix = tf.cast(
      dt.proposal_iou > options.grounding_iou_threshold, tf.float32)

  # Compute detection labels.
  dt.detection_instance_labels_list = []
  dt.detection_instance_logits_list = []
  dt.detection_instance_scores_list = []

  entity_proposal_id = dt.grounding.entity_proposal_id
  for itno in range(options.num_iterations):
    detection_instance_labels = _scatter_entity_labels(
        proposal_id=entity_proposal_id,
        entity_id=dt.entity_ids,
        max_n_proposal=dt.max_n_proposal,
        vocab_size=dt.vocab_size)
    detection_instance_labels = tf.matmul(propogation_matrix,
                                          detection_instance_labels)
    dt.detection_instance_labels_list.append(detection_instance_labels)

    # Predict detection scores.
    detection_head = tf.layers.Dense(dt.dims,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(
                                         mean=0.0, stddev=0.01),
                                     name='entity_detection_head_%i' % itno)(
                                         dt.proposal_features)
    (detection_instance_logits,
     detection_instance_scores) = _box_classify(detection_head, dt.embeddings,
                                                dt.bias_entity)
    dt.detection_instance_logits_list.append(detection_instance_logits)
    dt.detection_instance_scores_list.append(detection_instance_scores)

    # Update the proposal id associated to the image-level entity label.
    # I.e., update the ``g'' vector in the paper.
    dummy_attention = tf.gather_nd(tf.transpose(detection_instance_scores,
                                                [0, 2, 1]),
                                   indices=_get_full_indices(dt.entity_ids))
    entity_proposal_id = tf.math.argmax(dummy_attention,
                                        axis=2,
                                        output_type=tf.int32)
  # Compute attribute labels.
  if options.predict_attributes:
    dt.attribute_instance_labels = _scatter_attribute_labels(
        dt.grounding.entity_proposal_id, dt.per_ent_att_ids, dt.max_n_proposal,
        dt.vocab_size)
    attribute_head = tf.layers.Dense(dt.dims,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(
                                         mean=0.0, stddev=0.01),
                                     name='attribute_detection_head')(
                                         dt.proposal_features)
    (dt.attribute_instance_logits,
     dt.attribute_instance_scores) = _box_classify(attribute_head,
                                                   dt.embeddings,
                                                   dt.bias_attribute)

  # Save the grounding results.
  dt.refined_grounding.entity_proposal_id = entity_proposal_id
  dt.refined_grounding.entity_proposal_score = tf.reduce_max(
      dummy_attention, 2)

  indices = _get_full_indices(entity_proposal_id)
  dt.refined_grounding.entity_proposal_box = tf.gather_nd(
      dt.proposals, indices)
  dt.refined_grounding.entity_proposal_feature = tf.gather_nd(
      dt.proposal_features, indices)

  # Postprocess: non-maximum-suppression.
  post_process = options.post_process
  (dt.detection.nmsed_boxes, dt.detection.nmsed_scores,
   dt.detection.nmsed_classes,
   dt.detection.valid_detections) = tf.image.combined_non_max_suppression(
       tf.expand_dims(dt.proposals, 2),
       detection_instance_scores[:, :, 1:],
       max_output_size_per_class=post_process.max_size_per_class,
       max_total_size=post_process.max_total_size,
       iou_threshold=post_process.iou_thresh,
       score_threshold=post_process.score_thresh)
  dt.detection.nmsed_classes = tf.cast(
      1 + dt.detection.nmsed_classes, tf.int32)

  # Get the proposal id of the detection box, then fetch the other information.
  iou = _compute_iou(dt.detection.valid_detections, dt.detection.nmsed_boxes,
                     dt.n_proposal, dt.proposals)
  dt.detection.nmsed_proposal_id = tf.math.argmax(iou,
                                                  axis=2,
                                                  output_type=tf.int32)
  indices = _get_full_indices(dt.detection.nmsed_proposal_id)
  dt.detection.nmsed_features = tf.gather_nd(dt.proposal_features, indices)
  # nmsed_attribute = tf.gather_nd(dt.attribute_instance_scores, indices)
  # dt.detection.nmsed_attribute_scores = tf.reduce_max(nmsed_attribute, -1)
  # dt.detection.nmsed_attribute_classes = tf.argmax(nmsed_attribute,
  #                                                  axis=2,
  #                                                  output_type=tf.int32)

  return dt


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


def _box_classify(detection_head, embeddings, bias, normalize_fn=tf.nn.softmax):
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


def _compute_iou(n_box1, box1, n_box2, box2):
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


def _scatter_entity_labels(proposal_id, entity_id, max_n_proposal, vocab_size):
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


def _scatter_attribute_labels(proposal_id, attribute_id, max_n_proposal,
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
