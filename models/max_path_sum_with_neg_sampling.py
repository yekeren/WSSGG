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

from protos import model_pb2

from models import max_path_sum
from models import utils
from modeling.utils import masked_ops


class MaxPathSumWithNegSampling(max_path_sum.MaxPathSum):
  """MaxPathSum model with Negative path sampling."""

  def __init__(self, options, is_training):
    super(MaxPathSumWithNegSampling, self).__init__(options, is_training)

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

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
      predictions: A dictionary of prediction tensors keyed by name.
        - `max_path_sum/dp_solution/max_path_sum`: A [batch, max_n_triple] float tensor.
        - `max_path_sum/dp_solution/pseudo_subject_box`: A [batch, max_n_triple, 4] float tensor.
        - `max_path_sum/dp_solution/pseudo_subject_box_embs`: A [batch, max_n_triple, dims] float tensor.
        - `max_path_sum/dp_solution/pseudo_object_box`: A [batch, max_n_triple, 4] float tensor.
        - `max_path_sum/dp_solution/pseudo_object_box_embs`: A [batch, max_n_triple, dims] float tensor.
        - `max_path_sum/dp_solution/pseudo_relation_embs`: A [batch, max_n_triple, dims] float tensor.
        - `max_path_sum/graph_weights/subject_to_proposal`: A [batch, max_n_triple, max_n_proposal] float tensor.
        - `max_path_sum/graph_weights/proposal_to_proposal`: A [batch, max_n_triple, max_n_proposal, max_n_proposal] float tensor.
        - `max_path_sum/graph_weights/proposal_to_object`: A [batch, max_n_triple, max_n_proposal] float tensor.
        - `max_path_sum/graph_embs/proposal`: A [batch, max_n_proposal, dims] float tensor.
        - `max_path_sum/graph_embs/proposal_relation`: A [batch, max_n_proposal, max_n_proposal, dims] float tensor.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    (n_triple, subject_ids, object_ids,
     predicate_ids) = (inputs['scene_graph/n_triple'],
                       self.entity2id(inputs['scene_graph/subject']),
                       self.entity2id(inputs['scene_graph/object']),
                       self.predicate2id(inputs['scene_graph/predicate']))

    (max_path_sum, entity_emb_weights, predicate_emb_weights, subject_embs,
     object_embs, predicate_embs, ps_subject_box_embs, ps_object_box_embs,
     ps_relation_embs) = (
         predictions['max_path_sum/dp_solution/max_path_sum'],
         predictions['word_embedding/entity'],
         predictions['word_embedding/predicate'],
         predictions['max_path_sum/graph_embs/subject'],
         predictions['max_path_sum/graph_embs/object'],
         predictions['max_path_sum/graph_embs/predicate'],
         predictions['max_path_sum/dp_solution/pseudo_subject_box_embs'],
         predictions['max_path_sum/dp_solution/pseudo_object_box_embs'],
         predictions['max_path_sum/dp_solution/pseudo_relation_embs'],
     )

    # ps - pseudo; sp - sampled.
    def _triplet_loss(sim_ap, sim_an):
      return tf.maximum(0.0, sim_an - sim_ap + self.options.triplet_loss_margin)

    ps_p1 = tf.reduce_sum(subject_embs * ps_subject_box_embs, -1)
    ps_p2 = tf.reduce_sum(predicate_embs * ps_relation_embs, -1)
    ps_p3 = tf.reduce_sum(object_embs * ps_object_box_embs, -1)

    # Sample negativ path.
    losses = []
    for _ in range(self.options.triplet_loss_n_negatives):
      sampled_subject = utils.sample_one_based_ids_not_equal(
          subject_ids, max_id=len(self.entity_emb_weights) - 1)
      sampled_object = utils.sample_one_based_ids_not_equal(
          object_ids, max_id=len(self.entity_emb_weights) - 1)
      sampled_predicate = utils.sample_one_based_ids_not_equal(
          predicate_ids, max_id=len(self.predicate_emb_weights) - 1)

      sampled_subject_embs = tf.nn.embedding_lookup(entity_emb_weights,
                                                    sampled_subject)
      sampled_object_embs = tf.nn.embedding_lookup(entity_emb_weights,
                                                   sampled_object)
      sampled_predicate_embs = tf.nn.embedding_lookup(predicate_emb_weights,
                                                      sampled_predicate)

      sp_p1 = tf.reduce_sum(sampled_subject_embs * ps_subject_box_embs, -1)
      sp_p2 = tf.reduce_sum(sampled_predicate_embs * ps_relation_embs, -1)
      sp_p3 = tf.reduce_sum(sampled_object_embs * ps_object_box_embs, -1)

      t1_loss = _triplet_loss(ps_p1, sp_p1)
      t2_loss = _triplet_loss(ps_p2, sp_p2)
      t3_loss = _triplet_loss(ps_p3, sp_p3)
      mask = tf.sequence_mask(n_triple,
                              tf.shape(max_path_sum)[1],
                              dtype=tf.float32)
      triplet_loss = tf.add_n([t1_loss, t2_loss, t3_loss])
      losses.append(masked_ops.masked_avg(triplet_loss, mask=mask, dim=1))
    losses = tf.stack(losses, -1)
    return {
        'metrics/triplet_loss': tf.reduce_mean(losses),
    }

    # sampled_path_sum = tf.add_n([
    #     tf.reduce_sum(sampled_subject_embs * ps_subject_box_embs, -1),
    #     tf.reduce_sum(sampled_object_embs * ps_object_box_embs, -1),
    #     tf.reduce_sum(sampled_predicate_embs * ps_relation_embs, -1)
    # ])

    ###################################################
    # # Contrasive loss.
    # mask = tf.sequence_mask(n_triple,
    #                         tf.shape(max_path_sum)[1],
    #                         dtype=tf.float32)

    # contrasive_loss = sampled_path_sum - max_path_sum
    # contrasive_loss = tf.reduce_mean(
    #     masked_ops.masked_avg(contrasive_loss, mask=mask, dim=1))

    # return {
    #     'metrics/contrasive_loss': contrasive_loss,
    # }

    ###################################################
    # # Triplet loss.
    # triplet_loss = tf.maximum(
    #     0.0, sampled_path_sum - max_path_sum + self.options.triplet_loss_margin)
    # triplet_loss = tf.reduce_mean(
    #     masked_ops.masked_avg(triplet_loss, mask=mask, dim=1))

    # return {
    #     'metrics/triplet_loss': triplet_loss,
    # }
