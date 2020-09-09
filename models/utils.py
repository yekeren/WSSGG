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

import tensorflow as tf

from modeling.utils import box_ops
from modeling.utils import masked_ops

from object_detection.core.post_processing import batch_multiclass_non_max_suppression


def compute_max_path_sum(n_proposal, n_triple, subject_to_proposal,
                         proposal_to_proposal, proposal_to_object):
  """Computes the max path sum solution.

    Given the batch index and the image-level triple (subject, predicate, object),
    we explore the max path sum solution in the following graph:

                   `subject`      
                  /  |   |  \           (subject_to_proposal  - n_proposal edges)
                p1  p2   p3  p4   `dp0`
    `predicate`  | X | X | X |          (proposal_to_proposal - n_proposal x n_proposal edges)
                p1   p2  p3  p4   `dp1`
                  \  |   |  /           (proposal_to_object   - n_proposal edges)
                   `object`       `dp2`

    Dynamic programming solution (batch - b, tuple - t, proposal - p, proposal' - q):
      dp0[b, t, p] = subject_to_proposal[b, t, p]
      dp1[b, t, p] = max_q(dp0[b, t, q] + proposal_to_proposal[b, t, q, p])
      dp2[b, t] = max_q(dp1[b, t, q] + proposal_to_object[b, t, q])

    Backtracking the optimal path:
      bt1[b, t, p] = argmax_q(dp0[b, t, q] + proposal_to_proposal[b, t, q, p])
      bt2[b, t] = max_q(dp1[b, t, q] + proposal_to_object[b, t, q])

  Args:
    n_proposal: A [batch] int tensor.
    n_triple: A [batch] int tensor.
    subject_to_proposal: A [batch, max_n_triple, max_n_proposal] float tensor.
    proposal_to_proposal: A [batch, max_n_triple, max_n_proposal, max_n_proposal] float tensor.
    proposal_to_object: A [batch, max_n_triple, max_n_proposal] float tensor.

  Returns:
    max_path_sum: A [batch, max_n_triple] float tensor, the max-path-sum.
    subject_proposal_index: A [batch, max_n_triple] integer tensor, index of the subject proposal.
    object_proposal_index: A [batch, max_n_triple] integer tensor, index of the object proposal.
  """
  batch = subject_to_proposal.shape[0]
  max_n_triple = tf.shape(subject_to_proposal)[1]
  max_n_proposal = tf.shape(subject_to_proposal)[-1]

  # Computes triple / proposal mask.
  # - triple_mask = [batch, max_n_triple]
  # - proposal_mask = [batch, max_n_proposal]
  triple_mask = tf.sequence_mask(n_triple, max_n_triple, dtype=tf.int32)
  proposal_mask = tf.sequence_mask(n_proposal, max_n_proposal, dtype=tf.float32)

  # DP0.
  dp0 = subject_to_proposal

  # DP1, note the masked version of dp1 = tf.reduce_max(dp1, 2)
  dp1 = tf.expand_dims(dp0, -1) + proposal_to_proposal
  mask = tf.expand_dims(tf.expand_dims(proposal_mask, 1), -1)
  bt1 = masked_ops.masked_argmax(data=dp1, mask=mask, dim=2)
  bt1 = tf.cast(bt1, tf.int32)
  dp1 = tf.squeeze(masked_ops.masked_maximum(data=dp1, mask=mask, dim=2), 2)

  # DP2, note the masked version of dp2 = tf.reduce_max(dp2, 2).
  dp2 = dp1 + proposal_to_object
  mask = tf.expand_dims(proposal_mask, 1)
  bt2 = masked_ops.masked_argmax(data=dp2, mask=mask, dim=2)
  bt2 = tf.cast(bt2, tf.int32)
  dp2 = tf.squeeze(masked_ops.masked_maximum(data=dp2, mask=mask, dim=2), 2)

  # Backtrack the optimal path.
  max_path_sum, object_proposal_index = dp2, bt2
  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])
  triple_index = tf.broadcast_to(tf.expand_dims(tf.range(max_n_triple), 0),
                                 [batch, max_n_triple])
  track_index = tf.stack([batch_index, triple_index, bt2], -1)
  subject_proposal_index = tf.gather_nd(bt1, track_index)

  # Mask on triple dimension.
  max_path_sum = tf.multiply(max_path_sum, tf.cast(triple_mask, tf.float32))
  subject_proposal_index = tf.add(subject_proposal_index * triple_mask,
                                  triple_mask - 1)
  object_proposal_index = tf.add(object_proposal_index * triple_mask,
                                 triple_mask - 1)

  return max_path_sum, subject_proposal_index, object_proposal_index


def gather_proposal_by_index(proposals, proposal_index):
  """Gathers proposal box or proposal features by index..

  This is a helper function to extract max-path-sum solution.

  Args:
    proposals: A [batch, max_n_proposal, dims] float tensor.
      It could be either proposal box (dims=4) or proposal features.
    proposal_index: A [batch, max_n_triple] int tensor.

  Returns:
    A [batch, max_n_triple, dims] float tensor, the gathered proposal info,
      could be proposal boxes or proposal features.
  """
  batch = proposals.shape[0].value
  dims = proposals.shape[-1].value
  max_n_triple = tf.shape(proposal_index)[1]
  max_n_proposal = tf.shape(proposals)[1]

  proposals = tf.broadcast_to(tf.expand_dims(proposals, 1),
                              [batch, max_n_triple, max_n_proposal, dims])

  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])
  triple_index = tf.broadcast_to(tf.expand_dims(tf.range(max_n_triple), 0),
                                 [batch, max_n_triple])
  index = tf.stack([batch_index, triple_index, proposal_index], -1)
  return tf.gather_nd(proposals, index)


def gather_proposal_score_by_index(entity_to_proposal, proposal_index):
  """Gathers graph edge weight from entity node to proposal node.

  This is a helper function to extract max-path-sum solution.

  Args:
    entity_to_proposal: A [batch, max_n_triple, max_n_proposal] float tensor.
      It could be either `subject_to_proposal` or `proposal_to_object`.
    proposal_index: A [batch, max_n_triple] int tensor.

  Returns:
    A [batch, max_n_triple] float tensor, denoting the weight.
  """
  batch = entity_to_proposal.shape[0].value
  max_n_triple = tf.shape(entity_to_proposal)[1]
  max_n_proposal = tf.shape(entity_to_proposal)[2]

  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])
  triple_index = tf.broadcast_to(tf.expand_dims(tf.range(max_n_triple), 0),
                                 [batch, max_n_triple])
  index = tf.stack([batch_index, triple_index, proposal_index], -1)
  return tf.gather_nd(entity_to_proposal, index)


def gather_relation_by_index(relations, subject_index, object_index):
  """Gathers relation by index.

  This is a helper function to extract max-path-sum solution.

  Args:
    relations: A [batch, max_n_proposal, max_n_proposal, dims] float tensor.
    subject_index: A [batch, max_n_triple] int tensor.
    object_index: A [batch, max_n_triple] int tensor.

  Returns:
    A [batch, max_n_triple, dims] float tensor, the relation embedding vector.
  """
  batch = relations.shape[0].value
  dims = relations.shape[-1].value
  max_n_triple = tf.shape(subject_index)[1]
  max_n_proposal = tf.shape(relations)[1]

  relations = tf.broadcast_to(
      tf.expand_dims(relations, 1),
      [batch, max_n_triple, max_n_proposal, max_n_proposal, dims])

  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])
  triple_index = tf.broadcast_to(tf.expand_dims(tf.range(max_n_triple), 0),
                                 [batch, max_n_triple])
  index = tf.stack([batch_index, triple_index, subject_index, object_index], -1)
  return tf.gather_nd(relations, index)


def gather_relation_score_by_index(proposal_to_proposal, subject_index,
                                   object_index):
  """Gathers graph edge weight from subject proposal to object proposal.

  This is a helper function to extract max-path-sum solution.

  Args: 
    proposal_to_proposal: A [batch, max_n_triple, max_n_proposal, 
      max_n_proposal] float tensor.
    subject_index: A [batch, max_n_triple] int tensor.
    object_index: A [batch, max_n_triple] int tensor.

  Returns:
    A [batch, max_n_triple] float tensor, denoting the weight.
  """
  batch = proposal_to_proposal.shape[0].value
  max_n_triple = tf.shape(proposal_to_proposal)[1]
  max_n_proposal = tf.shape(proposal_to_proposal)[2]

  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])
  triple_index = tf.broadcast_to(tf.expand_dims(tf.range(max_n_triple), 0),
                                 [batch, max_n_triple])
  index = tf.stack([batch_index, triple_index, subject_index, object_index], -1)
  return tf.gather_nd(proposal_to_proposal, index)


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
    n_entity: A integer, total number of entities.
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

  # Create a zero tensor to scatter on.
  # Note: background is NOT properly set, need to call `normalize_pseudo_entity_detection_labels`.
  indices = tf.stack([index_batch, proposal_index, entity_index], axis=-1)
  pseudo_labels = tf.scatter_nd(indices,
                                updates=tf.fill([batch, max_n_triple], 1.0),
                                shape=[batch, max_n_proposal, 1 + n_entity])

  # Propogate box annotations.
  # - `iou` = [batch, max_n_proposal, max_n_proposal].
  proposal_broadcast1 = tf.broadcast_to(tf.expand_dims(
      proposals, 1), [batch, max_n_proposal, max_n_proposal, 4])
  proposal_broadcast2 = tf.broadcast_to(tf.expand_dims(
      proposals, 2), [batch, max_n_proposal, max_n_proposal, 4])
  iou = box_ops.iou(proposal_broadcast1, proposal_broadcast2)

  propogate_matrix = tf.cast(iou > iou_threshold, tf.float32)
  pseudo_labels = tf.matmul(propogate_matrix, pseudo_labels)
  return pseudo_labels


def post_process_pseudo_entity_detection_labels(n_entity,
                                                n_proposal,
                                                proposals,
                                                pseudo_labels,
                                                normalize=False):
  """Noarmalizes pseudo entity detection labels.

  Args:
    n_entity: A integer, total number of entities.
    n_proposal: A [batch] int tensor.
    proposals: A [batch, max_n_proposal, 4] float tensor.
    pseudo_labels: A [batch, max_n_proposal, 1 + n_entity] float tensor.

  Returns:
    normalzied_pseudo_labels: A [batch, max_n_proposal, 1 + n_entity] float tensor.
  """
  batch = n_proposal.shape[0].value
  max_n_proposal = tf.shape(proposals)[1]

  # Add the batckground dimension to the 0-th index.
  background = tf.cast(tf.equal(0.0, tf.reduce_sum(pseudo_labels, -1)),
                       tf.float32)
  background = tf.concat([
      tf.expand_dims(background, -1),
      tf.fill([batch, max_n_proposal, n_entity], 0.0)
  ], -1)
  pseudo_labels = tf.add(pseudo_labels, background)

  # Normalize the labels.
  if normalize:
    sumv = tf.reduce_sum(pseudo_labels, -1, keep_dims=True)
    pseudo_labels = tf.math.divide(pseudo_labels, sumv)

  return pseudo_labels


def nms_post_process(n_proposal,
                     proposals,
                     proposal_scores,
                     max_output_size_per_class=10,
                     max_total_size=20,
                     iou_threshold=0.5,
                     score_threshold=0.001):
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
  """
  proposals = tf.expand_dims(proposals, axis=2)

  (detection_boxes, detection_scores, detection_classes,
   num_detections) = tf.image.combined_non_max_suppression(
       boxes=proposals,
       scores=proposal_scores,
       max_output_size_per_class=max_output_size_per_class,
       max_total_size=max_total_size,
       iou_threshold=iou_threshold,
       score_threshold=score_threshold)
  detection_classes = tf.cast(detection_classes, tf.int32)
  return num_detections, detection_boxes, detection_scores, detection_classes
