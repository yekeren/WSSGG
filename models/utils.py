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


def compute_nms_l2_distillation_loss(entity_embs, proposal_embs,
                                     proposal_indicator):
  """Computes non-max-suppresion l2 distillation loss.

  Args:
    entity_embs: Subject or object embeddings of shape [batch, max_n_triple, dims].
    proposal_embs: A [batch, max_n_proposal, dims] float tensor.
    proposal_indicator: A boolean tensor denoting the matching relation, 
      shape=[batch, max_n_triple, max_n_proposal], should be the returned
      `highly_overlapped` of `gather_overlapped_box_indicator_by_iou`.

  Returns:
    A scalar tensor denoting the loss.
  """
  (batch, dims) = (entity_embs.shape[0].value, entity_embs.shape[-1].value)
  max_n_triple = tf.shape(entity_embs)[1]
  max_n_proposal = tf.shape(proposal_embs)[1]

  # Broadcast shape.
  (entity_embs, proposal_embs) = (tf.expand_dims(entity_embs, 2),
                                  tf.expand_dims(proposal_embs, 1))
  entity_embs = tf.broadcast_to(entity_embs,
                                [batch, max_n_triple, max_n_proposal, dims])
  proposal_embs = tf.broadcast_to(proposal_embs,
                                  [batch, max_n_triple, max_n_proposal, dims])

  proposal_indicator = tf.expand_dims(proposal_indicator, -1)
  proposal_indicator = tf.broadcast_to(
      proposal_indicator, [batch, max_n_triple, max_n_proposal, dims])
  proposal_indicator = tf.cast(proposal_indicator, tf.float32)

  # L2 distance.
  l2_sum = tf.reduce_sum(
      tf.square(entity_embs - proposal_embs) * proposal_indicator)
  l2_avg = tf.math.divide(l2_sum,
                          tf.maximum(1e-10, tf.reduce_sum(proposal_indicator)))
  return l2_avg


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
