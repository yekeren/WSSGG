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

from models import core
from models import model_base
from models import utils


def _l2_norm(x):
  return x / np.maximum(1e-10, np.linalg.norm(x, axis=-1, keepdims=True))


class MaxPathSum(model_base.ModelBase):
  """MaxPathSum model to provide instance-level annotations. """

  def __init__(self, options, is_training):
    super(MaxPathSum, self).__init__(options, is_training)

    if not isinstance(options, model_pb2.MaxPathSum):
      raise ValueError('Options has to be an MaxPathSum proto.')

    # Load token2id mapping and embedding data.
    with tf.io.gfile.GFile(options.token_to_id_meta_file, 'r') as fid:
      meta = json.load(fid)
      (entity2id, predicate2id) = (meta['label_to_idx'],
                                   meta['predicate_to_idx'])
    (self.entity2id,
     self.predicate2id) = (token_to_id.TokenToIdLayer(entity2id, oov_id=0),
                           token_to_id.TokenToIdLayer(predicate2id, oov_id=0))
    (self.entity_emb_weights,
     self.predicate_emb_weights) = (np.load(options.entity_emb_npy_file),
                                    np.load(options.predicate_emb_npy_file))

    (self.entity_emb_weights,
     self.predicate_emb_weights) = (_l2_norm(self.entity_emb_weights),
                                    _l2_norm(self.predicate_emb_weights))
    logging.info('Load embedding matrix: entity shape=%s, predicate shape=%s',
                 self.entity_emb_weights.shape,
                 self.predicate_emb_weights.shape)

    # Initialize the arg_scope for FC and CONV layers.
    self.arg_scope_fn = hyperparams.build_hyperparams(options.fc_hyperparams,
                                                      is_training)

  def _predict_relation_embeddings(self, proposal_embs, output_dims):
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

    proposal_embs_1 = tf.expand_dims(proposal_embs, 1)
    proposal_embs_2 = tf.expand_dims(proposal_embs, 2)

    proposal_embs_1 = tf.broadcast_to(
        proposal_embs_1, [batch, max_n_proposal, max_n_proposal, dims])
    proposal_embs_2 = tf.broadcast_to(
        proposal_embs_2, [batch, max_n_proposal, max_n_proposal, dims])

    edge_embs = tf.concat([proposal_embs_1, proposal_embs_2], -1)
    with slim.arg_scope(self.arg_scope_fn()):
      edge_embs = slim.fully_connected(edge_embs,
                                       num_outputs=output_dims,
                                       activation_fn=None)
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
    # The following tensors can only be used for evaluation/visualization:
    # - `scene_graph/subject/box`
    # - `scene_graph/object/box`

    if self.is_training:
      inputs.pop('scene_graph/subject/box')
      inputs.pop('scene_graph/object/box')

    # Convert subject, object, predicate into embeddings.
    # - subject_embs = [batch, max_n_triple, dims]
    # - object_embs = [batch, max_n_triple, dims]
    # - predicate_embs = [batch, max_n_triple, dims]

    (n_triple, subject_ids, object_ids,
     predicate_ids) = (inputs['scene_graph/n_triple'],
                       self.entity2id(inputs['scene_graph/subject']),
                       self.entity2id(inputs['scene_graph/object']),
                       self.predicate2id(inputs['scene_graph/predicate']))
    (entity_emb_weights,
     predicate_emb_weights) = (tf.Variable(self.entity_emb_weights, False),
                               tf.Variable(self.predicate_emb_weights, False))
    (subject_embs, object_embs,
     predicate_embs) = (tf.nn.embedding_lookup(entity_emb_weights, subject_ids),
                        tf.nn.embedding_lookup(entity_emb_weights, object_ids),
                        tf.nn.embedding_lookup(predicate_emb_weights,
                                               predicate_ids))

    # Get proposals_embs.
    # - proposal_embs = [batch, max_n_proposal, dims]
    (n_proposal, proposals,
     proposal_features) = (inputs['image/n_proposal'], inputs['image/proposal'],
                           inputs['image/proposal/feature'])

    dims = subject_embs.shape[-1].value
    with slim.arg_scope(self.arg_scope_fn()):
      proposal_features = slim.fully_connected(proposal_features,
                                               num_outputs=dims,
                                               activation_fn=tf.nn.relu6)
      proposal_features = slim.dropout(proposal_features,
                                       self.options.dropout_keep_prob,
                                       is_training=self.is_training)
      proposal_embs = slim.fully_connected(proposal_features,
                                           num_outputs=dims,
                                           activation_fn=None)

    proposal_embs = tf.nn.l2_normalize(proposal_embs, axis=-1)

    # Create matching graph.
    # - subject_to_proposal = [batch, max_n_triple, max_n_proposal]
    # - proposal_to_object = [batch, max_n_triple, max_n_proposal]
    # - proposal_to_proposal = [batch, max_n_triple, max_n_proposal, max_n_proposal]
    # - proposal_to_proposal_embs = [batch, max_n_triple, max_n_proposal, dims]

    subject_to_proposal = tf.linalg.matmul(subject_embs,
                                           proposal_embs,
                                           transpose_b=True)
    proposal_to_object = tf.linalg.matmul(object_embs,
                                          proposal_embs,
                                          transpose_b=True)

    proposal_semantic_relation_embs = self._predict_relation_embeddings(
        proposal_features, output_dims=dims)
    proposal_spatial_relation_embs = self._predict_relation_embeddings(
        proposals, output_dims=dims)

    proposal_relation_embs = tf.add(proposal_semantic_relation_embs,
                                    proposal_spatial_relation_embs)
    proposal_relation_embs = tf.nn.l2_normalize(proposal_relation_embs, axis=-1)
    proposal_to_proposal = tf.einsum('bpqd,btd->btpq', proposal_relation_embs,
                                     predicate_embs)

    # Compute maximum path sum and gather proposals using the index.
    (max_path_sum, subject_proposal_index,
     object_proposal_index) = core.compute_max_path_sum(n_proposal, n_triple,
                                                        subject_to_proposal,
                                                        proposal_to_proposal,
                                                        proposal_to_object)
    pseudo_subject_box = utils.gather_grounded_proposal_box(
        proposals, subject_proposal_index)
    pseudo_object_box = utils.gather_grounded_proposal_box(
        proposals, object_proposal_index)

    return {
        'max_path_sum/max_path_sum': max_path_sum,
        'max_path_sum/pseudo_subject_box': pseudo_subject_box,
        'max_path_sum/pseudo_object_box': pseudo_object_box,
        'max_path_sum/graph_weights/subject_to_proposal': subject_to_proposal,
        'max_path_sum/graph_weights/proposal_to_proposal': proposal_to_proposal,
        'max_path_sum/graph_weights/proposal_to_object': proposal_to_object,
        'max_path_sum/graph_embs/proposal': proposal_embs,
        'max_path_sum/graph_embs/proposal_relation': proposal_relation_embs,
    }

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
        - `max_path_sum/max_path_sum`: A [batch, max_n_triple] float tensor.
        - `max_path_sum/pseudo_subject_box`: A [batch, max_n_triple, 4] float tensor.
        - `max_path_sum/pseudo_object_box`: A [batch, max_n_triple, 4] float tensor.
        - `max_path_sum/graph_weights/subject_to_proposal`: A [batch, max_n_triple, max_n_proposal] float tensor.
        - `max_path_sum/graph_weights/proposal_to_proposal`: A [batch, max_n_triple, max_n_proposal, max_n_proposal] float tensor.
        - `max_path_sum/graph_weights/proposal_to_object`: A [batch, max_n_triple, max_n_proposal] float tensor.
        - `max_path_sum/graph_embs/proposal`: A [batch, max_n_proposal, dims] float tensor.
        - `max_path_sum/graph_embs/proposal_relation`: A [batch, max_n_proposal, max_n_proposal, dims] float tensor.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    (n_triple) = (inputs['scene_graph/n_triple'])
    (max_path_sum, ps_subject_box,
     ps_object_box) = (predictions['max_path_sum/max_path_sum'],
                       predictions['max_path_sum/pseudo_subject_box'],
                       predictions['max_path_sum/pseudo_object_box'])

    # Get the triple mask.
    triple_mask = tf.sequence_mask(n_triple,
                                   tf.shape(max_path_sum)[1],
                                   dtype=tf.float32)

    # Max-path-sum loss.
    mean_max_path_sum = tf.reduce_mean(
        masked_ops.masked_avg(max_path_sum, mask=triple_mask, dim=1))
    max_path_sum_loss = -mean_max_path_sum

    return {'metrics/path_sum_loss': max_path_sum_loss}

  def build_metrics(self, inputs, predictions, **kwargs):
    """Computes evaluation metrics.

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
        - `max_path_sum/max_path_sum`: A [batch, max_n_triple] float tensor.
        - `max_path_sum/pseudo_subject_box`: A [batch, max_n_triple, 4] float tensor.
        - `max_path_sum/pseudo_object_box`: A [batch, max_n_triple, 4] float tensor.
        - `max_path_sum/graph_weights/subject_to_proposal`: A [batch, max_n_triple, max_n_proposal] float tensor.
        - `max_path_sum/graph_weights/proposal_to_proposal`: A [batch, max_n_triple, max_n_proposal, max_n_proposal] float tensor.
        - `max_path_sum/graph_weights/proposal_to_object`: A [batch, max_n_triple, max_n_proposal] float tensor.
        - `max_path_sum/graph_embs/proposal`: A [batch, max_n_proposal, dims] float tensor.
        - `max_path_sum/graph_embs/proposal_relation`: A [batch, max_n_proposal, max_n_proposal, dims] float tensor.

    Returns:
      eval_metric_ops: dict of metric results keyed by name. The values are the
        results of calling a metric function, namely a (metric_tensor, 
        update_op) tuple. see tf.metrics for details.
    """
    (n_triple, gt_subject_box,
     gt_object_box) = (inputs['scene_graph/n_triple'],
                       inputs['scene_graph/subject/box'],
                       inputs['scene_graph/object/box'])

    (max_path_sum, ps_subject_box,
     ps_object_box) = (predictions['max_path_sum/max_path_sum'],
                       predictions['max_path_sum/pseudo_subject_box'],
                       predictions['max_path_sum/pseudo_object_box'])

    # Get the triple mask.
    triple_mask = tf.sequence_mask(n_triple,
                                   tf.shape(max_path_sum)[1],
                                   dtype=tf.float32)

    # IoU.
    (subject_iou, object_iou) = (box_ops.iou(gt_subject_box, ps_subject_box),
                                 box_ops.iou(gt_object_box, ps_object_box))

    # Compute metric ops.
    _mean_fn = lambda x: tf.reduce_mean(
        masked_ops.masked_avg(x, mask=triple_mask, dim=1))

    (mean_max_path_sum, mean_subject_iou,
     mean_object_iou) = (tf.keras.metrics.Mean(), tf.keras.metrics.Mean(),
                         tf.keras.metrics.Mean())

    mean_max_path_sum.update_state(_mean_fn(max_path_sum))
    mean_subject_iou.update_state(_mean_fn(subject_iou))
    mean_object_iou.update_state(_mean_fn(object_iou))

    return {
        'metrics/max_path_sum': mean_max_path_sum,
        'metrics/iou_subject': mean_subject_iou,
        'metrics/iou_object': mean_object_iou
    }
