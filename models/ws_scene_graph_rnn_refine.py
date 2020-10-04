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
  - Our previous work of Ye et al. 2019 (Cap2det),
  - Online instance classifier refinement, proposed by Tang et al. 2017 (OICR),
Data are preprocessed by:
  - Zareian et al. 2020 (VSPNet)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import json

import numpy as np
import tensorflow as tf

import tf_slim as slim
from models import graph_nms

from protos import model_pb2

from models.graph_mps import GraphMPS
from models.graph_nms import GraphNMS
from modeling.layers import id_to_token
from modeling.layers import token_to_id
from modeling.utils import box_ops
from modeling.utils import hyperparams
from modeling.utils import masked_ops

from models import ws_scene_graph
from models import utils

from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields

from bert.modeling import transformer_model

from metrics.scene_graph_evaluation import SceneGraphEvaluator


class WSSceneGraphRnnRefine(ws_scene_graph.WSSceneGraph):
  """WSSceneGraphRnnRefine model to provide instance-level annotations. """

  def _transformer_contextualize(self, subject_feature, object_feature,
                                 relation_feature):
    """Uses Transformer to contextualize the subject/object/relation features.

    Args:
      subject_feature: A [batch, max_n_triple, dims] float tensor.
      object_feature: A [batch, max_n_triple, dims] float tensor.
      relation_feature: A [batch, max_n_triple, dims] float tensor.

    Args:
      subject_feature: A [batch, max_n_triple, output_dims] float tensor.
      object_feature: A [batch, max_n_triple, output_dims] float tensor.
      relation_feature: A [batch, max_n_triple, output_dims] float tensor.
    """
    input_tensor = tf.stack([subject_feature, object_feature, relation_feature],
                            axis=1)
    hidden_size = input_tensor.shape[-1].value
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      output_tensor = transformer_model(
          input_tensor,
          hidden_size=hidden_size,
          num_hidden_layers=self.options.transformer_layers,
          intermediate_size=hidden_size * 2,
          attention_probs_dropout_prob=self.options.transformer_dropout_prob)
    (subject_feature, object_feature,
     relation_feature) = tf.unstack(output_tensor, axis=1)
    return subject_feature, object_feature, relation_feature

  def _contextualize(self, subject_feature, object_feature, relation_feature):
    """Contextualize the subject/object/relation features.

    Args:
      subject_feature: A [batch, max_n_triple, dims] float tensor.
      object_feature: A [batch, max_n_triple, dims] float tensor.
      relation_feature: A [batch, max_n_triple, dims] float tensor.

    Args:
      subject_feature: A [batch, max_n_triple, output_dims] float tensor.
      object_feature: A [batch, max_n_triple, output_dims] float tensor.
      relation_feature: A [batch, max_n_triple, output_dims] float tensor.
    """
    if self.options.use_transformer:
      batch = subject_feature.shape[0].value
      max_n_triple = tf.shape(subject_feature)[1]

      # Reshape features to [batch * max_n_triple, dims].
      reshape_fn = lambda x: tf.reshape(x, [-1, x.shape[-1].value])
      subject_feature = reshape_fn(subject_feature)
      object_feature = reshape_fn(object_feature)
      relation_feature = reshape_fn(relation_feature)

      (subject_feature, object_feature,
       relation_feature) = self._transformer_contextualize(
           subject_feature, object_feature, relation_feature)

      # Reshape output features to [batch, max_n_triple, dims].
      reshape_fn = lambda x: tf.reshape(
          x, [batch, max_n_triple, x.shape[-1].value])
      subject_feature = reshape_fn(subject_feature)
      object_feature = reshape_fn(object_feature)
      relation_feature = reshape_fn(relation_feature)

    return subject_feature, object_feature, relation_feature

  def _beam_search_refine_triple(self,
                                 n_triple,
                                 subject_feature,
                                 object_feature,
                                 rnn_layers=1,
                                 num_units=50,
                                 beam_size=5):
    """Refines triple prediction.

    Args:
      n_triple: A [batch] int tensor.
      subject_features: A [batch, max_n_triple, dims] float tensor.
      object_features: A [batch, max_n_triple, dims] float tensor.
    """

    def cell_fn():
      rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
      return tf.nn.rnn_cell.MultiRNNCell([rnn_cell for _ in range(rnn_layers)])

    batch = subject_feature.shape[0].value
    max_n_triple = tf.shape(subject_feature)[1]

    # Create visual sequence features.
    # - visual_seq_feature: a list including:
    #    * subject_feature = [batch * max_n_triple, dims] tensors.
    #    * object_feature = [batch * max_n_triple * beam_size, dims] tensors.
    #    * predicate_feature = [batch * max_n_triple * beam_size, dims] tensors.
    relation_feature = tf.add(subject_feature, object_feature)
    (subject_feature, object_feature,
     relation_feature) = self._contextualize(subject_feature, object_feature,
                                             relation_feature)
    visual_feature_list = []
    for i, visual_feature in enumerate(
        [subject_feature, object_feature, relation_feature]):
      if i > 0:
        visual_feature = tf.reshape(
            visual_feature, [batch * max_n_triple, 1, visual_feature.shape[-1]])
        visual_feature = tf.broadcast_to(
            visual_feature,
            [batch * max_n_triple, beam_size, visual_feature.shape[-1]])
        visual_feature_list.append(
            tf.reshape(visual_feature, [-1, visual_feature.shape[-1]]))
      else:
        visual_feature = tf.reshape(
            visual_feature, [batch * max_n_triple, visual_feature.shape[-1]])
        visual_feature_list.append(visual_feature)

    # Create text embedding weights.
    # - current_token_embs = [batch * max_n_triple * beam_size, dims].
    with tf.variable_scope('embedding_weights'):
      entity_emb_weights = tf.get_variable('entity',
                                           initializer=self.entity_emb_weights)
      predicate_emb_weights = tf.get_variable(
          'predicate', initializer=self.predicate_emb_weights)
    current_token_embs = tf.fill(
        [batch * max_n_triple,
         entity_emb_weights.get_shape()[-1]], 0.0)

    # Beam search using RNN.
    cell = cell_fn()
    current_states = cell.zero_state(batch_size=(batch * max_n_triple),
                                     dtype=tf.float32)

    beam_path = []
    beam_token_ids = []
    accum_probs = None

    for i in range(3):
      # Create inputs.
      # - inputs[0] = [batch * max_n_triple, visual_dims + text_dims].
      # - inputs[1,2] = [batch * max_n_triple * beam_size, visual_dims + text_dims].

      inputs = tf.concat([visual_feature_list[i], current_token_embs], -1)

      with tf.variable_scope('triple_refine'):
        with slim.arg_scope(self.arg_scope_fn()):
          # Get outputs at time `i`.
          # - outputs = [batch * max_n_triple * beam_size, output_dims].
          # - entity_logits = [batch * max_n_triple * beam_size, n_entity].
          # - predicate_logits = [batch * max_n_triple * beam_size, n_predicate].
          outputs, current_states = cell(
              inputs, current_states,
              scope='rnn/multi_rnn_cell')  # Scope is tricky!

          if i < 2:  # Predict an entity.
            scope = 'entity'
            num_outputs = self.n_entity
            embedding_weights = entity_emb_weights
          else:  # Predict a predicate.
            scope = 'predicate'
            num_outputs = self.n_predicate
            embedding_weights = predicate_emb_weights

          logits = slim.fully_connected(
              outputs,
              num_outputs=num_outputs,
              activation_fn=None,
              reuse=(True if i == 1 else
                     False),  # Object reuse variables from subject.
              scope=scope)
          probs = tf.log(tf.maximum(1e-10, tf.nn.softmax(logits)))
          if i == 0:
            probs = tf.reshape(probs, [batch * max_n_triple, num_outputs])
            # Tile cell state.
            new_states = []
            for state in current_states:
              h = tf.concat([state.h] * beam_size, 0)
              c = tf.concat([state.c] * beam_size, 0)
              new_states.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
            current_states = tuple(new_states)
          else:
            probs = tf.reshape(accum_probs + probs,
                               [batch * max_n_triple, beam_size * num_outputs])

          # Keep top-kpredictions.
          # - best_probs = [batch * max_n_triple * beam_size, 1].
          # - indices = [batch * max_n_triple * beam_size].
          best_probs, indices = tf.nn.top_k(probs, beam_size)

          indices = tf.reshape(indices, [-1])
          best_probs = tf.reshape(best_probs, [-1, 1])
          accum_probs = best_probs

          token_id = indices % num_outputs
          beam_parent = indices // num_outputs
          beam_path.append(beam_parent)
          beam_token_ids.append(token_id)

          # Update the `current_token_embs`.
          # - current_token_embs = [batch * max_n_triple * beam_size, text_dims].
          current_token_embs = tf.nn.embedding_lookup(embedding_weights,
                                                      token_id)
    # for i in range(3)

    # Reshape to [batch, max_n_triple, beam_size].
    (subject_ids, object_ids, predicate_ids) = [
        tf.reshape(x, [batch, max_n_triple, beam_size]) for x in beam_token_ids
    ]
    (_, subject_index, object_index) = [
        tf.reshape(x, [batch, max_n_triple, beam_size]) for x in beam_path
    ]

    # Backtrack the paths.
    index_batch = tf.range(batch)
    index_batch = tf.broadcast_to(tf.reshape(index_batch, [batch, 1, 1]),
                                  [batch, max_n_triple, beam_size])
    index_triple = tf.range(max_n_triple)
    index_triple = tf.broadcast_to(
        tf.reshape(index_triple, [1, max_n_triple, 1]),
        [batch, max_n_triple, beam_size])
    index_beam = tf.range(beam_size)
    index_beam = tf.broadcast_to(tf.reshape(index_beam, [1, 1, beam_size]),
                                 [batch, max_n_triple, beam_size])

    indices = tf.stack([index_batch, index_triple, index_beam], -1)
    predicate_ids = tf.gather_nd(predicate_ids, indices)

    index_beam = tf.gather_nd(object_index, indices)
    indices = tf.stack([index_batch, index_triple, index_beam], -1)
    object_ids = tf.gather_nd(object_ids, indices)

    index_beam = tf.gather_nd(subject_index, indices)
    indices = tf.stack([index_batch, index_triple, index_beam], -1)
    subject_ids = tf.gather_nd(subject_ids, indices)

    accum_probs = tf.reshape(accum_probs, [batch, max_n_triple, beam_size])
    return accum_probs, subject_ids, object_ids, predicate_ids

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
    predictions = super(WSSceneGraphRnnRefine, self).predict(inputs)

    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']

    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposal_features)[1]

    shared_hiddens = predictions['features/shared_hidden']

    # Post-process the predictions.
    if not self.is_training:
      graph_proposal_scores = predictions['refinement/iter_%i/proposal_probas' %
                                          self.options.n_refine_iteration]
      graph_relation_scores = predictions['refinement/relation_probas']
      graph_proposal_scores = graph_proposal_scores[:, :, 1:]
      graph_relation_scores = graph_relation_scores[:, :, 1:]

      # Search for the triple proposals.
      search_n_triple = predictions['search/n_triple']
      search_subject_proposal_index = predictions['search/subject/box_index']
      search_object_proposal_index = predictions['search/object/box_index']

      # Beam search.
      # - accum_probs = [batch, max_n_triple, beam_size].
      # - sub_ids, obj_ids, pred_ids = [batch, max_n_triple, beam_size].
      (accum_probs, sub_ids, obj_ids,
       pred_ids) = self._beam_search_refine_triple(
           search_n_triple,
           graph_nms.GraphNMS._gather_proposal_by_index(
               shared_hiddens, search_subject_proposal_index),
           graph_nms.GraphNMS._gather_proposal_by_index(
               shared_hiddens, search_object_proposal_index),
           rnn_layers=self.options.rnn_layers,
           num_units=self.options.rnn_hidden_units,
           beam_size=self.options.beam_size)

      predictions.update({
          'beam_refine/accum_probs': accum_probs,
          'beam_refine/subject': self.id2entity(sub_ids),
          'beam_refine/object': self.id2entity(obj_ids),
          'beam_refine/predicate': self.id2predicate(pred_ids),
      })

      (n_valid_example, scores, sub, sub_box_index, pred, obj,
       obj_box_index) = utils.beam_search_post_process(
           search_n_triple,
           search_subject_proposal_index,
           search_object_proposal_index,
           accum_probs,
           sub_ids,
           pred_ids,
           obj_ids,
           max_total_size=self.options.max_total_size,
       )

      predictions.update({
          'beam/n_triple':
              n_valid_example,
          'beam/accum_probs':
              scores,
          'beam/subject/box':
              graph_nms.GraphNMS._gather_proposal_by_index(
                  proposals, sub_box_index),
          'beam/object/box':
              graph_nms.GraphNMS._gather_proposal_by_index(
                  proposals, obj_box_index),
          'beam/subject':
              self.id2entity(sub),
          'beam/object':
              self.id2entity(obj),
          'beam/predicate':
              self.id2predicate(pred),
      })
    return predictions

  def _compute_triple_refine_losses(
      self,
      n_triple,
      subject_feature,
      object_feature,
      subject_ids,
      object_ids,
      predicate_ids,
      rnn_layers=1,
      num_units=50,
      input_keep_prob=1.0,
      output_keep_prob=1.0,
      state_keep_prob=1.0,
  ):
    """Refines triple prediction using RNN.

    Args:
      subject_feature: A [batch, max_n_triple, entity_dims] float tensor.
      object_feature: A [batch, max_n_triple, entity_dims] float tensor.

    Returns:
      subject_ids: A [batch, max_n_triple, n_entity] float tensor.
      object_ids: A [batch, max_n_triple, n_entity] float tensor.
      predicate_ids: A [batch, max_n_triple, n_predicate] float tensor.
    """

    def cell_fn():
      rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
      rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
          rnn_cell,
          input_keep_prob=input_keep_prob,
          output_keep_prob=output_keep_prob,
          state_keep_prob=state_keep_prob)
      return tf.nn.rnn_cell.MultiRNNCell([rnn_cell for _ in range(rnn_layers)])

    # Create visual sequence features.
    # - subject/object/relation_feature = [batch, max_n_triple, visual_dims].
    relation_feature = tf.add(subject_feature, object_feature)
    (subject_feature, object_feature,
     relation_feature) = self._contextualize(subject_feature, object_feature,
                                             relation_feature)

    visual_seq_feature = tf.stack(
        [subject_feature, object_feature, relation_feature], 2)

    # Create text sequence features.
    # - text_seq_feature = [batch, max_n_triple, 3, text_dims].
    with tf.variable_scope('embedding_weights',
                           reuse=(False if self.is_training else True)):
      entity_emb_weights = tf.get_variable('entity',
                                           initializer=self.entity_emb_weights)
      predicate_emb_weights = tf.get_variable(
          'predicate', initializer=self.predicate_emb_weights)
    subject_embs = tf.nn.embedding_lookup(entity_emb_weights, subject_ids)
    object_embs = tf.nn.embedding_lookup(entity_emb_weights, object_ids)
    text_seq_feature = tf.stack(
        [tf.zeros_like(subject_embs), subject_embs, object_embs], 2)

    # Create input features.
    # - seq_feature = [batch*max_n_triple, 3, visual_dims + text_dims].
    batch = subject_feature.shape[0].value
    max_n_triple = tf.shape(subject_feature)[1]
    seq_feature = tf.concat([visual_seq_feature, text_seq_feature], -1)
    seq_feature = tf.reshape(
        seq_feature, [batch * max_n_triple, 3, seq_feature.shape[-1].value])

    # RNN decoder.
    with tf.variable_scope('triple_refine',
                           reuse=(False if self.is_training else True)):
      with slim.arg_scope(self.arg_scope_fn()):
        outputs, _ = tf.nn.dynamic_rnn(cell=cell_fn(),
                                       inputs=seq_feature,
                                       dtype=tf.float32,
                                       scope='rnn')
        entity_logits = slim.fully_connected(outputs[:, :2, :],
                                             num_outputs=self.n_entity,
                                             activation_fn=None,
                                             scope='entity')
        predicate_logits = slim.fully_connected(outputs[:, -1, :],
                                                num_outputs=self.n_predicate,
                                                activation_fn=None,
                                                scope='predicate')
    # Compute losses.
    entity_logits = tf.reshape(
        entity_logits, [batch, max_n_triple, 2, entity_logits.shape[-1]])
    subject_logits, object_logits = tf.unstack(entity_logits, axis=2)
    predicate_logits = tf.reshape(
        predicate_logits, [batch, max_n_triple, predicate_logits.shape[-1]])

    triple_mask = tf.sequence_mask(n_triple,
                                   maxlen=max_n_triple,
                                   dtype=tf.float32)
    loss_dict = {}
    for name, logits, labels in [
        ('losses/refine_subject', subject_logits, subject_ids),
        ('losses/refine_object', object_logits, object_ids),
        ('losses/refine_predicate', predicate_logits, predicate_ids)
    ]:
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=logits)
      loss_dict[name] = tf.reduce_mean(
          masked_ops.masked_avg(losses, triple_mask, dim=1))
    return loss_dict

  def build_losses(self, inputs, predictions, **kwargs):
    """Computes loss tensors.

    Args:
      inputs: A dictionary of input tensors keyed by names.
      predictions: A dictionary of prediction tensors keyed by name.

    Returns:
      loss_dict: A dictionary of loss tensors keyed by names.
    """
    loss_dict = super(WSSceneGraphRnnRefine,
                      self).build_losses(inputs, predictions)

    subject_ids = self.entity2id(inputs['scene_graph/subject'])
    object_ids = self.entity2id(inputs['scene_graph/object'])
    predicate_ids = self.predicate2id(inputs['scene_graph/predicate'])

    n_triple = inputs['scene_graph/n_triple']
    max_n_triple = tf.shape(subject_ids)[1]
    shared_hidden = predictions['features/shared_hidden']

    n_proposal = inputs['image/n_proposal']
    proposals = inputs['image/proposal']
    proposal_features = inputs['image/proposal/feature']
    batch = n_proposal.shape[0]
    max_n_proposal = tf.shape(proposals)[1]

    # Build RNN to contextualizing label generation process.
    (subject_index, object_index,
     predicate_index) = self._create_selection_indices(subject_ids, object_ids,
                                                       predicate_ids)

    proposal_scores_0 = predictions['refinement/iter_%i/proposal_probas' %
                                    self.options.n_refine_iteration][:, :, 1:]
    relation_scores_0 = predictions['refinement/relation_probas'][:, :, 1:]

    proposal_to_proposal_weight = slim.dropout(
        self.options.joint_inferring_relation_weight,
        self.options.mps_relation_dropout_keep_prob,
        is_training=self.is_training)
    mps = GraphMPS(n_triple=n_triple,
                   n_proposal=n_proposal,
                   subject_to_proposal=tf.gather_nd(
                       tf.transpose(proposal_scores_0, [0, 2, 1]),
                       subject_index),
                   proposal_to_proposal=tf.reshape(
                       tf.gather_nd(tf.transpose(relation_scores_0, [0, 2, 1]),
                                    predicate_index),
                       [batch, max_n_triple, max_n_proposal, max_n_proposal]),
                   proposal_to_object=tf.gather_nd(
                       tf.transpose(proposal_scores_0, [0, 2, 1]),
                       object_index),
                   proposal_to_proposal_weight=proposal_to_proposal_weight,
                   use_log_prob=self.options.use_log_prob)

    loss_dict.update(
        self._compute_triple_refine_losses(
            n_triple,
            mps.get_subject_feature(shared_hidden),
            mps.get_object_feature(shared_hidden),
            subject_ids,
            object_ids,
            predicate_ids,
            rnn_layers=self.options.rnn_layers,
            num_units=self.options.rnn_hidden_units,
            input_keep_prob=self.options.rnn_input_keep_prob,
            output_keep_prob=self.options.rnn_output_keep_prob,
            state_keep_prob=self.options.rnn_state_keep_prob))
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
    metric_dict = super(WSSceneGraphRnnRefine,
                        self).build_metrics(inputs, predictions)

    eval_dict = {
        'image_id': inputs['id'],
        'groundtruth/n_triple': inputs['scene_graph/n_triple'],
        'groundtruth/subject': inputs['scene_graph/subject'],
        'groundtruth/subject/box': inputs['scene_graph/subject/box'],
        'groundtruth/object': inputs['scene_graph/object'],
        'groundtruth/object/box': inputs['scene_graph/object/box'],
        'groundtruth/predicate': inputs['scene_graph/predicate'],
        'prediction/n_triple': predictions['beam/n_triple'],
        'prediction/subject/box': predictions['beam/subject/box'],
        'prediction/object/box': predictions['beam/object/box'],
        'prediction/subject': predictions['beam/subject'],
        'prediction/object': predictions['beam/object'],
        'prediction/predicate': predictions['beam/predicate'],
    }

    evaluator = SceneGraphEvaluator()
    for k, v in evaluator.get_estimator_eval_metric_ops(eval_dict).items():
      metric_dict['metrics/beam/%s' % k] = v

    metric_dict['metrics/accuracy'] = metric_dict[
        'metrics/beam/scene_graph_recall@100']
    return metric_dict
