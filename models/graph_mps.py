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

import tensorflow as tf

from modeling.utils import masked_ops


class GraphMPS(object):
  """Explores the Maximum-Path-Sum solution. """

  def __init__(self,
               n_triple,
               n_proposal,
               subject_to_proposal,
               proposal_to_proposal,
               proposal_to_object,
               subject_to_proposal_weight=1.0,
               proposal_to_proposal_weight=1.0,
               proposal_to_object_weight=1.0):
    """Initializes the object.

    Args:
      n_triple: A [batch] int tensor.
      n_proposal: A [batch] int tensor.
      subject_to_proposal: A [batch, max_n_triple, max_n_proposal] float tensor.
      proposal_to_proposal: A [batch, max_n_triple, max_n_proposal, max_n_proposal] float tensor.
      proposal_to_object: A [batch, max_n_triple, max_n_proposal] float tensor.
      subject_to_proposal_weight: Weight of `subject_to_proposal` edges.
      proposal_to_proposal_weight: Weight factor of `proposal_to_proposal` edges.
      proposal_to_object_weight: Weight factor of `proposal_to_object` edges.
    """
    subject_to_proposal = tf.multiply(subject_to_proposal,
                                      subject_to_proposal_weight)
    proposal_to_proposal = tf.multiply(proposal_to_proposal,
                                       proposal_to_proposal_weight)
    proposal_to_object = tf.multiply(proposal_to_object,
                                     proposal_to_object_weight)

    (self._max_path_sum, self._subject_proposal_index,
     self._object_proposal_index) = self.compute_max_path_sum(
         n_proposal, n_triple, subject_to_proposal, proposal_to_proposal,
         proposal_to_object)

    self._subject_to_proposal_edge_weight = self._gather_proposal_score_by_index(
        subject_to_proposal, self.subject_proposal_index)
    self._proposal_to_object_edge_weight = self._gather_proposal_score_by_index(
        proposal_to_object, self.object_proposal_index)

    self._proposal_to_proposal_edge_weight = self._gather_relation_score_by_index(
        proposal_to_proposal, self.subject_proposal_index,
        self.object_proposal_index)

  @property
  def max_path_sum(self):
    """Returns the maximum path sum.

    Returns:
      A [batch, max_n_triple] float tensor, the maximum path sum.
    """
    return self._max_path_sum

  @property
  def subject_proposal_index(self):
    """Returns subject proposal index.
    
    Returns:
      A [batch, max_n_triple] int tensor, each value denotes a proposal index.
    """
    return self._subject_proposal_index

  @property
  def object_proposal_index(self):
    """Returns subject proposal index.
    
    Returns:
      A [batch, max_n_triple] int tensor, each value denotes a proposal index.
    """
    return self._object_proposal_index

  @property
  def subject_to_proposal_edge_weight(self):
    """Returns subject-proposal edge weight on the optimal path.

    Returns:
      A [batch, max_n_triple] float tensor, each value denotes an edge weight on
        the optimal path.
    """
    return self._subject_to_proposal_edge_weight

  @property
  def proposal_to_proposal_edge_weight(self):
    """Returns proposal-proposal edge weight on the optimal path.

    Returns:
      A [batch, max_n_triple] float tensor, each value denotes an edge weight on
        the optimal path.
    """
    return self._proposal_to_proposal_edge_weight

  @property
  def proposal_to_object_edge_weight(self):
    """Returns proposal-object edge weight on the optimal path.

    Returns:
      A [batch, max_n_triple] float tensor, each value denotes an edge weight on
        the optimal path.
    """
    return self._proposal_to_object_edge_weight

  def get_subject_box(self, proposals):
    """Returns the subject box.

    Args:
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      A [batch, max_n_triple, 4] float tensor.
    """
    return self._gather_proposal_by_index(proposals,
                                          self.subject_proposal_index)

  def get_object_box(self, proposals):
    """Returns the object box.

    Args:
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      A [batch, max_n_triple, 4] float tensor.
    """
    return self._gather_proposal_by_index(proposals, self.object_proposal_index)

  @staticmethod
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
    proposal_mask = tf.sequence_mask(n_proposal,
                                     max_n_proposal,
                                     dtype=tf.float32)

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

  def _gather_proposal_score_by_index(self, entity_to_proposal, proposal_index):
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

  def _gather_relation_score_by_index(self, proposal_to_proposal, subject_index,
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
    index = tf.stack([batch_index, triple_index, subject_index, object_index],
                     -1)
    return tf.gather_nd(proposal_to_proposal, index)

  def _gather_proposal_by_index(self, proposals, proposal_index):
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
