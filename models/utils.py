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

from modeling.utils import box_ops
from modeling.utils import masked_ops

from object_detection.core.post_processing import batch_multiclass_non_max_suppression


def gather_overlapped_box_indicator_by_iou(n_proposal,
                                           proposals,
                                           n_reference,
                                           reference,
                                           threshold=0.5):
  """Gathers overlapped proposal boxes and split into two sets.

    Proposals have IoU >= threshold will be denoted in the `highly_overlapped`.
    Proposals have IoU < threshold will be denoted in the `roughly_overlapped`.

  Args:
    n_proposal: A [batch] int tensor, number of proposals.
    proposals: A [batch, max_n_proposal, 4] float tensor.
    n_reference: A [batch] int tensor, number of reference.
    reference: A [batch, max_n_reference, 4] float tensor.
    threshold: A float, determining the threshold.

  Returns:
    highly_overlapped: A [batch, max_n_reference, max_n_proposal] boolean tensor.
    roughly_overlapped: A [batch, max_n_reference, max_n_proposal] boolean tensor.
  """
  # Compute mask.
  # - proposal_mask = [batch, max_n_proposal]
  # - reference_mask = [batch, max_n_reference]
  (proposal_mask, reference_mask) = (tf.sequence_mask(n_proposal,
                                                      tf.shape(proposals)[1]),
                                     tf.sequence_mask(n_reference,
                                                      tf.shape(reference)[1]))
  mask = tf.logical_and(tf.expand_dims(proposal_mask, 1),
                        tf.expand_dims(reference_mask, 2))
  mask = tf.cast(mask, tf.float32)

  # Compute iou.
  # - iou = [batch, max_n_reference, max_n_proposal]
  iou = box_ops.iou(tf.expand_dims(reference, 2), tf.expand_dims(proposals, 1))
  iou = tf.multiply(iou, mask)
  highly_overlapped = tf.math.greater_equal(iou, threshold)
  roughly_overlapped = tf.logical_and(tf.math.greater(iou, 0.0),
                                      tf.math.less(iou, threshold))
  return highly_overlapped, roughly_overlapped


def sample_one_based_ids_not_equal(ids, max_id=100):
  """Random sample one based ids NOT equal to `ids`.

  Args:
    ids: A [batch, max_n_triple] int tensor. 
    max_id: Max id value.

  Returns:
    sampled_id: A [batch, max_n_triple] int tensor, each entry has value in
      [1...max_id] and should not be equal to the value in `ids`.
  """
  batch = ids.shape[0].value
  max_n_triple = tf.shape(ids)[1]

  # `offset`  values are in [1, max_id - 1].
  offset = tf.random.uniform([batch, max_n_triple],
                             minval=1,
                             maxval=max_id,
                             dtype=tf.int32)
  # Note `ids` starts with 1.
  sampled_ids = tf.mod(ids - 1 + offset, max_id) + 1
  return sampled_ids


def sample_index_not_equal(index, size):
  """Random sample index NOT equal to `index`.

  Args:
    index: A [batch, max_n_triple] int tensor. 
    size: A [batch] int tensor, the size of the indexing array.

  Returns:
    sampled_index: A [batch, max_n_triple] int tensor, each entry has value 
      sampled_index[i,j] in [0...size[i]) and should not equal to index[i,j].
  """
  batch = index.shape[0].value
  max_n_triple = tf.shape(index)[1]

  random_value = tf.random.uniform([batch, max_n_triple],
                                   minval=0,
                                   maxval=9999999999,
                                   dtype=tf.int32)
  size_minus_1 = tf.expand_dims(size, 1) - 1
  offset = tf.mod(random_value, size_minus_1)

  sampled_index = tf.mod((index + offset + 1), tf.expand_dims(size, 1))

  return sampled_index


def scatter_pseudo_entity_detection_labels(n_entity,
                                           n_proposal,
                                           proposals,
                                           entity_index,
                                           proposal_index,
                                           iou_threshold=0.5):
  """Creates pseudo entity detection labels.

    Besides labeling the denoted entries, also propogate pseudo annotations
    to highly overlapped boxes.

  Args:
    n_entity: An integer, total number of entities.
    n_proposal: A [batch] int tensor.
    proposals: A [batch, max_n_proposal, 4] float tensor.
    entity_index: A [batch, max_n_triple] int tensor, denoting the entity index
      in the range [0, n_entity]. The 0-th entry denotes the background.
    proposal_index: A [batch, max_n_triple] in tensor, denoting the proposal
      index in the range [0, n_proposal - 1].
    iou_threshold: Also labels the boxes overlapped >= `iou_threshold`.

  Returns:
    pseudo_labels: A [batch, max_n_proposal, 1 + n_entity] float tensor.
  """
  batch = n_proposal.shape[0].value
  max_n_proposal = tf.shape(proposals)[1]
  max_n_triple = tf.shape(entity_index)[1]

  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])

  # Note: background is NOT properly set, need to call `normalize_pseudo_detection_labels`.
  indices = tf.stack([index_batch, proposal_index, entity_index], axis=-1)
  pseudo_labels = tf.scatter_nd(indices,
                                updates=tf.fill([batch, max_n_triple], 1.0),
                                shape=[batch, max_n_proposal, 1 + n_entity])

  # Propogate box annotations.
  # - `pseudo_labels` = [batch, max_n_proposal, 1 + n_entity].
  # - `iou` = [batch, max_n_proposal, max_n_proposal].
  proposal_broadcast1 = tf.broadcast_to(tf.expand_dims(
      proposals, 1), [batch, max_n_proposal, max_n_proposal, 4])
  proposal_broadcast2 = tf.broadcast_to(tf.expand_dims(
      proposals, 2), [batch, max_n_proposal, max_n_proposal, 4])
  iou = box_ops.iou(proposal_broadcast1, proposal_broadcast2)

  propogate_matrix = tf.cast(iou > iou_threshold, tf.float32)
  pseudo_labels = tf.matmul(propogate_matrix, pseudo_labels)
  return pseudo_labels


def scatter_pseudo_relation_detection_labels(n_predicate,
                                             n_proposal,
                                             proposals,
                                             predicate_index,
                                             subject_proposal_index,
                                             object_proposal_index,
                                             iou_threshold=0.5):
  """Creates pseudo relation detection labels.

    Besides labeling the denoted entries, also propogate pseudo annotations
    to highly overlapped relations.

  Args:
    n_predicate: An integer, total number of predicates.
    n_proposal: A [batch] int tensor.
    proposals: A [batch, max_n_proposal, 4] float tensor.
    predicate_index: A [batch, max_n_triple] int tensor, denoting the predicate
      index in the range [0, n_predicate]. The 0-th entry denotes the background.
    subject_proposal_index: A [batch, max_n_triple] in tensor, denoting the
      proposal index in the range [0, n_proposal - 1].
    object_proposal_index: A [batch, max_n_triple] in tensor, denoting the
      proposal index in the range [0, n_proposal - 1].
    iou_threshold: Also labels the boxes overlapped >= `iou_threshold`.

  Returns:
    Pseudo relation labels of shape [batch, max_n_proposal, max_n_proposal, 1 + n_predicate].
  """
  batch = n_proposal.shape[0].value
  max_n_proposal = tf.shape(proposals)[1]
  max_n_triple = tf.shape(predicate_index)[1]

  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])

  # Note: background is NOT properly set, need to call `normalize_pseudo_detection_labels`.
  indices = tf.stack([
      index_batch, subject_proposal_index, object_proposal_index,
      predicate_index
  ], -1)
  pseudo_labels = tf.scatter_nd(
      indices,
      updates=tf.fill([batch, max_n_triple], 1.0),
      shape=[batch, max_n_proposal, max_n_proposal, 1 + n_predicate])

  # Propogate box annotations.
  # - `pseudo_labels` = [batch, max_n_proposal, max_n_proposal, 1 + n_predicate].
  # - `iou` = [batch, max_n_proposal, max_n_proposal].
  proposal_broadcast1 = tf.broadcast_to(tf.expand_dims(
      proposals, 1), [batch, max_n_proposal, max_n_proposal, 4])
  proposal_broadcast2 = tf.broadcast_to(tf.expand_dims(
      proposals, 2), [batch, max_n_proposal, max_n_proposal, 4])
  iou = box_ops.iou(proposal_broadcast1, proposal_broadcast2)
  propogate_matrix = tf.cast(iou > iou_threshold, tf.float32)

  # Broadcast and matmul.
  # - `propogate_matrix` = [batch, 1 + n_predicate, max_n_proposal, max_n_proposal].
  # - `pseudo_labels` = [batch, 1 + n_predicate, max_n_proposal, max_n_proposal]
  propogate_matrix = tf.broadcast_to(
      tf.expand_dims(propogate_matrix, -1),
      [batch, max_n_proposal, max_n_proposal, 1 + n_predicate])

  propogate_matrix = tf.transpose(propogate_matrix, [0, 3, 1, 2])
  pseudo_labels = tf.transpose(pseudo_labels, [0, 3, 1, 2])
  pseudo_labels = tf.matmul(tf.matmul(propogate_matrix, pseudo_labels),
                            propogate_matrix)

  # Transpose back.
  pseudo_labels = tf.transpose(pseudo_labels, [0, 2, 3, 1])
  return pseudo_labels


def post_process_pseudo_detection_labels(pseudo_labels, normalize=False):
  """Noarmalizes pseudo entity detection labels.

  Args:
    pseudo_labels: A [batch, max_n_proposal, 1 + n_classes] float tensor.

  Returns:
    normalzied_pseudo_labels: A [batch, max_n_proposal, 1 + n_classes] float tensor.
  """
  batch = pseudo_labels.shape[0].value
  max_n_proposal = tf.shape(pseudo_labels)[1]
  n_classes = pseudo_labels.shape[-1].value - 1

  # Add the batckground dimension to the 0-th index.
  background = tf.cast(tf.equal(0.0, tf.reduce_sum(pseudo_labels, -1)),
                       tf.float32)
  background = tf.concat([
      tf.expand_dims(background, -1),
      tf.fill([batch, max_n_proposal, n_classes], 0.0)
  ], -1)
  pseudo_labels = tf.add(pseudo_labels, background)
  if normalize:
    sumv = tf.reduce_sum(pseudo_labels, -1, keep_dims=True)
    pseudo_labels = tf.math.divide(pseudo_labels, sumv)
  else:
    pseudo_labels = tf.cast(pseudo_labels > 0, tf.float32)
  return pseudo_labels


def beam_search_post_process(n_triple,
                             subject_box_index,
                             object_box_index,
                             beam_scores,
                             beam_subject,
                             beam_predicate,
                             beam_object,
                             max_total_size=100):
  """

  Args:
    n_triple: A [batch] int tensor.
    subject_box_index: A [batch, max_n_triple] int tensor.
    object_box_index: A [batch, max_n_triple] int tensor.
    beam_scores: A [batch, max_n_triple, beam_size] float tensor.
    beam_subject_ids: A [batch, max_n_triple, beam_size] int tensor.
    beam_predicate_ids: A [batch, max_n_triple, beam_size] int tensor.
    beam_object_ids: A [batch, max_n_triple, beam_size] int tensor.
    max_total_size: Max total triples to retain.

  Returns:
    pass.
  """
  batch = beam_scores.shape[0].value
  beam_size = beam_scores.shape[-1].value
  max_n_triple = tf.shape(beam_scores)[1]

  triple_mask = tf.sequence_mask(n_triple,
                                 maxlen=max_n_triple,
                                 dtype=tf.float32)
  reshape_fn = lambda x: tf.reshape(x, [batch, max_n_triple * beam_size])

  # Note: consider scores of ZERO.
  top_k = max_n_triple * beam_size  # No difference from sorting.
  beam_scores += tf.multiply(-9999999.0, tf.expand_dims(1 - triple_mask, -1))
  best_scores, indices_1 = tf.nn.top_k(reshape_fn(beam_scores), top_k)

  # Get the indices to gather the top `max_total_size` predictions.
  indices_0 = tf.broadcast_to(tf.expand_dims(tf.range(batch), -1),
                              [batch, top_k])
  indices = tf.stack([indices_0, indices_1], -1)

  # Extract results.
  subject_box_index = tf.broadcast_to(tf.expand_dims(subject_box_index, 2),
                                      [batch, max_n_triple, beam_size])
  object_box_index = tf.broadcast_to(tf.expand_dims(object_box_index, 2),
                                     [batch, max_n_triple, beam_size])

  ret_subject_box_index = tf.gather_nd(reshape_fn(subject_box_index), indices)
  ret_object_box_index = tf.gather_nd(reshape_fn(object_box_index), indices)
  ret_subject = tf.gather_nd(reshape_fn(beam_subject), indices)
  ret_object = tf.gather_nd(reshape_fn(beam_object), indices)
  ret_predicate = tf.gather_nd(reshape_fn(beam_predicate), indices)

  n_valid_example = tf.reduce_sum(tf.cast(best_scores > -9999999.0, tf.int32),
                                  -1)

  def _py_per_image_nms(n_example, scores, sub, sub_box_ind, pred, obj,
                        obj_box_ind):
    total = 0
    dedup = set()

    ret_data = []
    ret_scores = []
    for i in range(n_example):
      key = (sub[i], sub_box_ind[i], pred[i], obj[i], obj_box_ind[i])
      if not key in dedup:
        dedup.add(key)
        ret_data.append(list(key))
        ret_scores.append(scores[i])

    if len(dedup):
      ret_scores = np.array(ret_scores, np.float32)
      ret_data = np.array(ret_data, np.int32)
    else:
      ret_scores = np.zeros((0), dtype=np.float32)
      ret_data = np.zeros((0, 5), dtype=np.int32)

    if len(dedup) < max_total_size:
      pad = max_total_size - len(dedup)
      ret_data = np.concatenate(
          [ret_data, np.zeros((pad, 5), dtype=np.int32)], 0)
      ret_scores = np.concatenate(
          [ret_scores, np.zeros((pad), dtype=np.float32)], 0)

    ret_scores = ret_scores[:max_total_size]
    ret_data = ret_data[:max_total_size, :]

    return [
        np.array(len(dedup), np.int32), ret_scores, ret_data[:, 0],
        ret_data[:, 1], ret_data[:, 2], ret_data[:, 3], ret_data[:, 4]
    ]

  def _per_image_nms(elems):
    return tf.py_func(_py_per_image_nms, elems, [
        tf.int32,
        tf.float32,
        tf.int32,
        tf.int32,
        tf.int32,
        tf.int32,
        tf.int32,
    ])

  batch_outputs = tf.map_fn(_per_image_nms,
                            elems=[
                                n_valid_example, best_scores, ret_subject,
                                ret_subject_box_index, ret_predicate,
                                ret_object, ret_object_box_index
                            ],
                            dtype=[
                                tf.int32,
                                tf.float32,
                                tf.int32,
                                tf.int32,
                                tf.int32,
                                tf.int32,
                                tf.int32,
                            ],
                            parallel_iterations=32,
                            back_prop=False)

  (n_valid_example, best_scores, ret_subject, ret_subject_box_index,
   ret_predicate, ret_object, ret_object_box_index) = batch_outputs

  n_valid_example.set_shape([batch])
  best_scores.set_shape([batch, max_total_size])
  ret_subject.set_shape([batch, max_total_size])
  ret_subject_box_index.set_shape([batch, max_total_size])
  ret_predicate.set_shape([batch, max_total_size])
  ret_object.set_shape([batch, max_total_size])
  ret_object_box_index.set_shape([batch, max_total_size])

  return (n_valid_example, best_scores, ret_subject, ret_subject_box_index,
          ret_predicate, ret_object, ret_object_box_index)


def nms_post_process(n_proposal,
                     proposals,
                     proposal_scores,
                     max_size_per_class=10,
                     max_total_size=300,
                     iou_thresh=0.5,
                     score_thresh=0.001,
                     use_class_agnostic_nms=False):
  """Non-max-suppression process.

  Args:
    n_proposal: A [batch] int tensor.
    proposals: A [batch, max_n_proposal, 4] float tensor.
    proposal_scores: A [batch, max_n_proposal, n_entity] float tensor.

  Returns:
    num_detections: A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top num_detections[i] entries in
      nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.
    detection_boxes: A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    detection_scores: A [batch_size, max_detections] float32 tensor containing
      the scores of the boxes.
    detection_classes: A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    selected_indices: A [batch_size, max_detections] int tensor.
  """
  batch = proposals.shape[0]
  max_n_proposal = tf.shape(proposals)[1]

  proposal_indices = tf.expand_dims(tf.range(max_n_proposal, dtype=tf.int32), 0)
  proposal_indices = tf.broadcast_to(proposal_indices, [batch, max_n_proposal])

  proposals = tf.expand_dims(proposals, axis=2)
  (detection_boxes, detection_scores, detection_classes, _, additional_fields,
   num_detections) = batch_multiclass_non_max_suppression(
       boxes=proposals,
       scores=proposal_scores,
       num_valid_boxes=n_proposal,
       additional_fields={'proposal_indices': proposal_indices},
       max_size_per_class=max_size_per_class,
       max_total_size=max_total_size,
       iou_thresh=iou_thresh,
       score_thresh=score_thresh,
       use_class_agnostic_nms=use_class_agnostic_nms,
       parallel_iterations=256)
  detection_classes = tf.cast(detection_classes, tf.int32)
  selected_indices = additional_fields['proposal_indices']

  return (num_detections, detection_boxes, detection_scores, detection_classes,
          selected_indices)


def compute_sigmoid_focal_loss(labels, logits, gamma=2.0, alpha=0.25):
  """Compute focal loss.

  Args:
    labels: A [batch, max_n_proposal, n_classes] float tensor.
    logits: A [batch, max_n_proposal, n_classes] float tensor.

  Returns:
    loss: A [batch, max_n_proposal, n_classes] float tensor.
  """
  per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                 logits=logits))
  prediction_probabilities = tf.sigmoid(logits)
  p_t = ((labels * prediction_probabilities) + ((1 - labels) *
                                                (1 - prediction_probabilities)))
  modulating_factor = 1.0
  if gamma:
    modulating_factor = tf.pow(1.0 - p_t, gamma)
  alpha_weight_factor = 1.0
  if alpha is not None:
    alpha_weight_factor = (labels * alpha + (1 - labels) * (1 - alpha))
  focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                              per_entry_cross_ent)
  return focal_cross_entropy_loss


def compute_spatial_relation_feature(box1, box2):
  """Computes spatial relation features between two sets of boxes.

  Args:
    box1, A [i1,...iN, 4] float tensor.
    box2, A [i1,...iN, 4] float tensor.

  Returns:
    relation_features: A [i1,...,iN, 34] float tensor.
  """
  features_list = []

  # Unary features.
  for box in [box1, box2]:
    center = box_ops.center(box)
    size = box_ops.size(box)
    area = box_ops.area(box)
    features_list.extend([box, center, size, tf.expand_dims(area, -1)])

  # Pairwise features.
  x_distance = box_ops.x_distance(box1, box2)
  y_distance = box_ops.y_distance(box1, box2)
  x_intersect = box_ops.x_intersect_len(box1, box2)
  y_intersect = box_ops.y_intersect_len(box1, box2)
  intersect_area = box_ops.area(box_ops.intersect(box1, box2))
  iou = box_ops.iou(box1, box2)

  height1, width1 = tf.unstack(box_ops.size(box1), axis=-1)
  height2, width2 = tf.unstack(box_ops.size(box2), axis=-1)
  area1 = box_ops.area(box1)
  area2 = box_ops.area(box2)

  features_list.extend([
      tf.expand_dims(x, -1) for x in [
          x_distance, y_distance, x_distance / width1, x_distance /
          width2, y_distance / height1, y_distance /
          height2, x_intersect, y_intersect, x_intersect / width1, x_intersect /
          width2, y_intersect / height1, y_intersect /
          height2, intersect_area, intersect_area / area1, intersect_area /
          area2, iou
      ]
  ])
  return tf.concat(features_list, -1)
