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

import heapq
from protos import model_pb2
from modeling.utils import masked_ops
from modeling.utils import box_ops
from models.cap2sg_data import DataTuple


def detect_relations(options, dt, reuse=False):
  """Detects relations.

  Args:
    options: A Cap2SGRelation proto.
    dt: A DataTuple objects.

  Returns:
  """
  if not isinstance(options, model_pb2.Cap2SGRelation):
    raise ValueError('Options has to be a Cap2SGDetection proto.')

  if not isinstance(dt, DataTuple):
    raise ValueError('Invalid DataTuple object.')

  batch = dt.detection.nmsed_features.shape[0].value
  beam_size = options.beam_size

  feature_dims = dt.detection.nmsed_features.shape[2].value
  max_n_detection = dt.detection.nmsed_features.shape[1].value
  max_n_relation = max_n_detection * max_n_detection

  detection_masks = tf.sequence_mask(dt.detection.valid_detections,
                                     maxlen=max_n_detection,
                                     dtype=tf.float32)
  relation_masks = tf.reshape(
      tf.multiply(tf.expand_dims(detection_masks, 2),
                  tf.expand_dims(detection_masks, 1)), [batch, max_n_relation])

  # Enumerate pairwise detections, max_n_relation == max_n_detection ** 2.
  #   subject/object/relation_vis_feature shape = [batch, max_n_relation, feature_dims].
  (subject_vis_feature, object_vis_feature, relation_vis_feature, subject_box,
   object_box) = _enumerate_relations(dt.detection.nmsed_features,
                                      dt.detection.nmsed_boxes)
  # Initialize rnn_vis_input.
  #   subject_vis_feature shape = [batch * max_n_relation, feature_dims].
  #   object_/relation_vis_feature shape = [batch * max_n_relation * beam_size, feature_dims].
  (subject_vis_feature, object_vis_feature, relation_vis_feature) = [
      tf.reshape(x, [-1, feature_dims])
      for x in [subject_vis_feature, object_vis_feature, relation_vis_feature]
  ]
  (object_vis_feature, relation_vis_feature) = [
      tf.reshape(
          tf.broadcast_to(tf.expand_dims(x, 1),
                          [batch * max_n_relation, beam_size, feature_dims]),
          [-1, feature_dims])
      for x in [object_vis_feature, relation_vis_feature]
  ]

  # Initialize RNN cell.
  cell = _create_rnn_cell(1, dt.dims)
  current_states = cell.zero_state(batch_size=(batch * max_n_relation),
                                   dtype=tf.float32)

  # Execute RNN to infer subject, object, predicate.
  search_path, search_tokens, search_log_probs = [], [], []

  rnn_text_feature = tf.fill([batch * max_n_relation, dt.dims], 0.0)
  for i, rnn_vis_feature in enumerate(
      [subject_vis_feature, object_vis_feature, relation_vis_feature]):
    rnn_logits, current_states = _execute_rnn_once(
        cell,
        rnn_vis_input=rnn_vis_feature,
        rnn_txt_input=rnn_text_feature,
        current_states=current_states,
        embeddings=dt.embeddings,
        bias=dt.bias_entity,
        reuse=reuse)

    if i == 0:
      current_states = _lift_rnn_state(current_states, beam_size)
      accum_log_probs = tf.log(tf.maximum(1e-6, tf.nn.softmax(rnn_logits)))
    else:
      log_probs = tf.log(tf.maximum(1e-6, tf.nn.softmax(rnn_logits)))
      accum_log_probs = tf.reshape(
          search_log_probs[-1] + log_probs,
          [batch * max_n_relation, beam_size * dt.vocab_size])

    _save_beam_search_path(accum_log_probs, beam_size, dt.vocab_size,
                           search_path, search_tokens, search_log_probs)

    # Update input text feature.
    rnn_text_feature = dt.embedding_func(search_tokens[-1])

  # Backtrack the solutions.
  _reshape_list = lambda x: [
      tf.reshape(e, [batch, max_n_relation, beam_size]) for e in x
  ]
  search_path = _reshape_list(search_path)
  search_tokens = _reshape_list(search_tokens)
  search_log_probs = _reshape_list(search_log_probs)

  # Backtrack the paths.
  selected_tokens, selected_accum_log_probs = _backtrack_solutions(
      search_path, search_tokens, search_log_probs)
  accum_log_probs = selected_accum_log_probs[-1]

  selected_log_probs = [selected_accum_log_probs[0]]
  for i in range(1, len(selected_accum_log_probs)):
    selected_log_probs.append(
        tf.subtract(selected_accum_log_probs[i],
                    selected_accum_log_probs[i - 1]))

  relation_masks = tf.broadcast_to(tf.expand_dims(relation_masks, -1),
                                   [batch, max_n_relation, beam_size])
  accum_log_probs = tf.where(relation_masks > 0,
                             x=accum_log_probs,
                             y=tf.multiply(-100.0, (1.0 - relation_masks)))

  # Postprocess the beam search results.
  num_relations = tf.constant([max_n_relation * beam_size] * batch)

  (dt.relation.num_relations, dt.relation.log_prob, dt.relation.subject_class,
   dt.relation.subject_score, dt.relation.subject_box, dt.relation.object_class,
   dt.relation.object_score, dt.relation.object_box, dt.relation.relation_class,
   dt.relation.relation_score) = _beam_search_post_process(
       num_relations,
       subject_box,
       object_box,
       accum_log_probs,
       selected_tokens[0],
       selected_log_probs[0],
       selected_tokens[1],
       selected_log_probs[1] - selected_log_probs[0],
       selected_tokens[2],
       selected_log_probs[2] - selected_log_probs[1],
       iou_thresh=0.5,
       max_total_size=options.relation_max_total_size)

  return dt


def _enumerate_relations(detection_features, detection_boxes):
  """Enumerates pair-wise relations among the detection results.

  Args:
    detection_features: Detection features, a [batch, max_n_detection, feature_dims] float tensor.
    detection_boxes: Detection boxes, a [batch, max_n_detection, 4] float tensor.

  Returns:
    subject_feature: Enumerated subject features, a [batch, max_n_relation, feature_dims] float tensor.
    object_feature: Enumerated object features, a [batch, max_n_relation, feature_dims] float tensor.
    predicate_feature: Enumerated predicate features, a [batch, max_n_relation, feature_dims] float tensor.
  """
  batch = detection_features.shape[0].value
  feature_dims = detection_features.shape[2].value
  max_n_detection = detection_features.shape[1].value
  max_n_relation = max_n_detection * max_n_detection

  subject_feature = tf.broadcast_to(
      tf.expand_dims(detection_features, 2),
      [batch, max_n_detection, max_n_detection, feature_dims])
  object_feature = tf.broadcast_to(
      tf.expand_dims(detection_features, 1),
      [batch, max_n_detection, max_n_detection, feature_dims])

  subject_box = tf.broadcast_to(tf.expand_dims(detection_boxes, 2),
                                [batch, max_n_detection, max_n_detection, 4])
  object_box = tf.broadcast_to(tf.expand_dims(detection_boxes, 1),
                               [batch, max_n_detection, max_n_detection, 4])
  predicate_feature = _compute_relation_feature(subject_feature, subject_box,
                                                object_feature, object_box)
  return (tf.reshape(subject_feature, [batch, -1, feature_dims]),
          tf.reshape(object_feature, [batch, -1, feature_dims]),
          tf.reshape(predicate_feature, [batch, -1, feature_dims]),
          tf.reshape(subject_box,
                     [batch, -1, 4]), tf.reshape(object_box, [batch, -1, 4]))


def _execute_rnn_once(cell,
                      rnn_vis_input,
                      rnn_txt_input,
                      current_states,
                      embeddings,
                      bias,
                      reuse=True):
  """Executes the RNN once.

  Args:
    cell: RNN cell.
    rnn_vis_input: RNN visual input, a [batch * max_n_relation * beam_size, dims] tensor.
    rnn_txt_input: RNN text input, a [batch * max_n_relation * beam_size, dims] tensor.
    current_states: A tuple storing the `c` and `h` states of the LSTM.
    embeddings: Word embeddings, a [vocab_size, dims] float tensor.
    bias: Class bias, a [vocab_size] float tensor.
    reuse: If true, reuse rnn variable.

  Returns:
    rnn_logits: A [batch * max_n_relation * beam_size, vocab_size] float tensor.
    current_states: Updated RNN states.
  """
  rnn_input = tf.concat([rnn_vis_input, rnn_txt_input], -1)
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    rnn_output, current_states = cell(rnn_input,
                                      current_states,
                                      scope='rnn/multi_rnn_cell')
  rnn_logits = tf.nn.bias_add(
      tf.matmul(rnn_output, embeddings, transpose_b=True), bias)
  return rnn_logits, current_states


def _lift_rnn_state(current_states, beam_size):
  """Lifts the beam_size dimension in the RNN states.

  Args:
    current_states: A tuple storing the `c` and `h` states of the LSTM.
    beam_size: Beam size.

  Returns
    current_states: Lifted states.
  """
  new_states = []
  for state in current_states:
    h = tf.concat([state.h] * beam_size, 0)
    c = tf.concat([state.c] * beam_size, 0)
    new_states.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
  return tuple(new_states)


def _save_beam_search_path(accum_log_probs, beam_size, vocab_size, search_path,
                           search_tokens, search_log_probs):
  """Adds information to the `search_` arrays for backtracking the solutions.

  Args:
    accum_log_probs: Accumulated log probabilities, a [batch * max_n_relation, beam_size * vocab_size] float tensor.
    beam_size: Beam size.
    search_path: A list of [batch * max_n_relation * beam_size] tensors, each is an index tensor pointing to the previous (parent) solution.
    search_tokens: A list of [batch * max_n_relation * beam_size] tensors, token id selected.
    search_log_probs: A list of [batch * max_n_relation * beam_size, 1] tensors, saving the optimal log probabilities.
  """
  # The indices returned by top_k involve both the token id and the previous solution info.
  #   Note: accum_log_probs shape = [batch * max_n_relation, beam_size * vocab].
  best_log_probs, indices = tf.nn.top_k(accum_log_probs, beam_size)

  indices = tf.reshape(indices, [-1])  # [batch * max_n_relation * beam_size]
  accum_log_probs = tf.reshape(
      best_log_probs, [-1, 1])  # [batch * max_n_relation * beam_size, 1]

  token_id = indices % vocab_size
  parent = indices // vocab_size

  search_path.append(parent)
  search_tokens.append(token_id)
  search_log_probs.append(accum_log_probs)


def _backtrack_solutions(search_path, search_tokens, search_log_probs):
  """Backtracks the beam search solution.

  Args:
    search_path: A list of [batch, max_n_relation, beam_size] tensors, each is an index tensor pointing to the previous (parent) solution.
    search_tokens: A list of [batch, max_n_relation, beam_size] tensors, token id selected.
    search_log_probs: A list of [batch, max_n_relation, beam_size] tensors, saving the optimal log probabilities.

  Args:
    selected_tokens: A list of [batch, max_n_relation, beam_size] int tensors.
    selected_accum_log_probs: A list of [batch, max_n_relation, beam_size] float tensors.
  """
  elem = search_path[-1]
  batch, max_n_relation, beam_size = [x.value for x in elem.shape]

  index_batch = tf.reshape(tf.range(batch), [batch, 1, 1])
  index_relation = tf.reshape(tf.range(max_n_relation), [1, max_n_relation, 1])
  index_beam = tf.reshape(tf.range(beam_size), [1, 1, beam_size])

  index_batch, index_relation, index_beam = [
      tf.broadcast_to(x, [batch, max_n_relation, beam_size])
      for x in [index_batch, index_relation, index_beam]
  ]

  # Backtrck.
  selected_tokens, selected_accum_log_probs = [], []
  indices = tf.stack([index_batch, index_relation, index_beam], -1)
  while search_path:
    parent = search_path.pop()
    tokens = search_tokens.pop()
    log_probs = search_log_probs.pop()

    selected_tokens.append(tf.gather_nd(tokens, indices))
    selected_accum_log_probs.append(tf.gather_nd(log_probs, indices))
    index_beam = tf.gather_nd(parent, indices)
  return selected_tokens[::-1], selected_accum_log_probs[::-1]


def train_relations(options, dt):
  """Detects relations.

  Args:
    options: A Cap2SGRelation proto.
    dt: A DataTuple objects.

  Returns:
  """
  if not isinstance(options, model_pb2.Cap2SGRelation):
    raise ValueError('Options has to be a Cap2SGDetection proto.')

  if not isinstance(dt, DataTuple):
    raise ValueError('Invalid DataTuple object.')

  # Get subject/object/predicate feature/box.
  #   feature shape = [batch, max_n_edge, feature_dims].
  #   box shape = [batch, max_n_edge, 4].
  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(dt.batch), 1),
                                [dt.batch, dt.max_n_relation])
  index_subject = tf.stack([index_batch, dt.relation_senders], -1)
  index_object = tf.stack([index_batch, dt.relation_receivers], -1)

  (subject_txt_feature, subject_vis_feature, dt.subject_boxes,
   dt.subject_labels) = (tf.gather_nd(dt.entity_embs, index_subject),
                         tf.gather_nd(
                             dt.refined_grounding.entity_proposal_feature,
                             index_subject),
                         tf.gather_nd(dt.refined_grounding.entity_proposal_box,
                                      index_subject),
                         tf.gather_nd(dt.entity_ids, index_subject))

  (object_txt_feature, object_vis_feature, dt.object_boxes,
   dt.object_labels) = (tf.gather_nd(dt.entity_embs, index_object),
                        tf.gather_nd(
                            dt.refined_grounding.entity_proposal_feature,
                            index_object),
                        tf.gather_nd(dt.refined_grounding.entity_proposal_box,
                                     index_object),
                        tf.gather_nd(dt.entity_ids, index_object))

  dt.predicate_labels = dt.relation_ids
  relation_vis_feature = _compute_relation_feature(subject_vis_feature,
                                                   dt.subject_boxes,
                                                   object_vis_feature,
                                                   dt.object_boxes)

  # Create sequence model.
  cell_fn = lambda: _create_rnn_cell(1, dt.dims)
  (subject_output, object_output, predicate_output) = _sequence_modeling(
      subject_vis_feature, object_vis_feature, relation_vis_feature,
      subject_txt_feature, object_txt_feature, cell_fn)

  dt.subject_logits = tf.nn.bias_add(
      tf.matmul(subject_output, dt.embeddings, transpose_b=True),
      dt.bias_entity)
  dt.object_logits = tf.nn.bias_add(
      tf.matmul(object_output, dt.embeddings, transpose_b=True), dt.bias_entity)
  dt.predicate_logits = tf.nn.bias_add(
      tf.matmul(predicate_output, dt.embeddings, transpose_b=True),
      dt.bias_relation)
  return dt


def _create_rnn_cell(rnn_layers,
                     num_units,
                     input_keep_prob=1.0,
                     output_keep_prob=1.0,
                     state_keep_prob=1.0,
                     is_training=True):
  """Creates RNN cell.

  Args:
    rnn_layers: Number of RNN layers.
    num_units: Hidden units of the RNN model.
    input_keep_prob: Keep probability of the input dropout.
    output_keep_prob: Keep probability of the output dropout.
    state_keep_prob: Keep probability of the state dropout.
  """
  rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
  rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,
                                           input_keep_prob=input_keep_prob,
                                           output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob)
  return tf.nn.rnn_cell.MultiRNNCell([rnn_cell for _ in range(rnn_layers)])


def _sequence_modeling(subject_vis_feature, object_vis_feature,
                       relation_vis_feature, subject_txt_feature,
                       object_txt_feature, cell_fn):
  """Builds sequence model to predict subject, object, and relation.

  Args:
    subject_vis_features: A [batch, max_n_edge, feature_dims] float tensor.
    object_vis_features: A [batch, max_n_edge, feature_dims] float tensor.
    relation_vis_feature: A [batch, max_n_edge, feature_dims] float tensor.
    subject_txt_features: A [batch, max_n_edge, dims] float tensor.
    object_txt_features: A [batch, max_n_edge, dims] float tensor.
    cell_fn: A callable to create the RNN cell.

  Returns:
    subject_output: A [batch, max_n_edge, dims] float tensor.
    object_output: A [batch, max_n_edge, dims] float tensor.
    predicate_output: A [batch, max_n_edge, dims] float tensor.
  """
  # Fuse the visual and text features.
  start_txt_feature = tf.zeros_like(subject_txt_feature)
  text_seq_feature = tf.stack(
      [start_txt_feature, subject_txt_feature, object_txt_feature], 2)
  visual_seq_feature = tf.stack(
      [subject_vis_feature, object_vis_feature, relation_vis_feature], 2)
  seq_feature = tf.concat([visual_seq_feature, text_seq_feature], -1)

  # Create RNN model.
  #   seq_feature shape = [batch * max_n_relation, 3, feature_dims + dims].
  #   seq_outputs shape = [batch * max_n_relation, 3, dims].
  batch = seq_feature.shape[0].value
  seq_feature = tf.reshape(seq_feature, [-1, 3, seq_feature.shape[-1].value])

  seq_outputs, _ = tf.nn.dynamic_rnn(cell=cell_fn(),
                                     inputs=seq_feature,
                                     dtype=tf.float32,
                                     scope='rnn')
  return tuple([
      tf.reshape(x, [batch, -1, x.shape[-1].value])
      for x in tf.unstack(seq_outputs, axis=1)
  ])


def _compute_relation_feature(subject_feature, subject_box, object_feature,
                              object_box):
  """Computes the relation feature.

  Args:
    subject_features: A [batch, max_n_edge, feature_dims] float tensor.
    subject_box: A [batch, max_n_edge, 4] float tensor.
    object_features: A [batch, max_n_edge, feature_dims] float tensor.
    object_box: A [batch, max_n_edge, 4] float tensor.

  Returns:
    relation_feature: A [batch, max_n_edge, feature_dims] float tensor.
  """
  return tf.multiply(subject_feature, object_feature)


def _beam_search_post_process(n_triple,
                              subject_box,
                              object_box,
                              beam_scores,
                              beam_subject,
                              beam_subject_score,
                              beam_object,
                              beam_object_score,
                              beam_predicate,
                              beam_predicate_score,
                              iou_thresh=0.5,
                              max_total_size=100):
  """

  Args:
    n_triple: A [batch] int tensor.
    subject_box: A [batch, max_n_triple, 4] int tensor.
    object_box: A [batch, max_n_triple, 4] int tensor.
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
  reshape_box_fn = lambda x: tf.reshape(x, [batch, max_n_triple * beam_size, 4])

  # Note: consider scores of ZERO.
  top_k = max_n_triple * beam_size  # No difference from sorting.
  beam_scores += tf.multiply(-9999999.0, tf.expand_dims(1 - triple_mask, -1))
  best_scores, indices_1 = tf.nn.top_k(reshape_fn(beam_scores), top_k)

  # Get the indices to gather the top `max_total_size` predictions.
  indices_0 = tf.broadcast_to(tf.expand_dims(tf.range(batch), -1),
                              [batch, top_k])
  indices = tf.stack([indices_0, indices_1], -1)

  # Extract top results.
  subject_box = tf.broadcast_to(tf.expand_dims(subject_box, 2),
                                [batch, max_n_triple, beam_size, 4])
  object_box = tf.broadcast_to(tf.expand_dims(object_box, 2),
                               [batch, max_n_triple, beam_size, 4])

  ret_subject_box = tf.gather_nd(reshape_box_fn(subject_box), indices)
  ret_object_box = tf.gather_nd(reshape_box_fn(object_box), indices)
  ret_subject = tf.gather_nd(reshape_fn(beam_subject), indices)
  ret_subject_score = tf.gather_nd(reshape_fn(beam_subject_score), indices)
  ret_object = tf.gather_nd(reshape_fn(beam_object), indices)
  ret_object_score = tf.gather_nd(reshape_fn(beam_object_score), indices)
  ret_predicate = tf.gather_nd(reshape_fn(beam_predicate), indices)
  ret_predicate_score = tf.gather_nd(reshape_fn(beam_predicate_score), indices)

  n_valid_example = tf.reduce_sum(tf.cast(best_scores > -9999999.0, tf.int32),
                                  -1)

  def _py_per_image_nms(n_example, scores, sub, sub_score, sub_box, pred,
                        pred_score, obj, obj_score, obj_box):
    dedup_index = []
    dedup_score = []
    dedup_subject = []
    dedup_subject_score = []
    dedup_subject_box = []
    dedup_predicate = []
    dedup_predicate_score = []
    dedup_object = []
    dedup_object_score = []
    dedup_object_box = []

    subject_iou = box_ops.py_iou(np.expand_dims(sub_box, 1),
                                 np.expand_dims(sub_box, 0))
    object_iou = box_ops.py_iou(np.expand_dims(obj_box, 1),
                                np.expand_dims(obj_box, 0))

    for i in range(n_example):
      j = 0
      while j < len(dedup_score):
        if (sub[i] == dedup_subject[j] and pred[i] == dedup_predicate[j] and
            obj[i] == dedup_object[j] and
            subject_iou[i, dedup_index[j]] > iou_thresh and
            object_iou[i, dedup_index[j]] > iou_thresh):
          break
        j += 1

      if j == len(dedup_score):
        dedup_index.append(i)
        dedup_score.append(scores[i])
        dedup_subject.append(sub[i])
        dedup_subject_score.append(sub_score[i])
        dedup_subject_box.append(sub_box[i])
        dedup_predicate.append(pred[i])
        dedup_predicate_score.append(pred_score[i])
        dedup_object.append(obj[i])
        dedup_object_score.append(obj_score[i])
        dedup_object_box.append(obj_box[i])
        if len(dedup_score) >= max_total_size:
          break

    def _pad_fn(x, dtype=np.float32):
      if isinstance(x, list):
        x = np.array(x, dtype=dtype)
      if len(x) < max_total_size:
        pad = max_total_size - len(x)
        if len(x.shape) == 1:
          x = np.concatenate([x, np.zeros((pad), dtype=dtype)], 0)
        elif len(x.shape) == 2 and x.shape[-1] == 4:
          x = np.concatenate([x, np.zeros((pad, 4), dtype=dtype)], 0)
        else:
          raise ValueError('Not supported')

      return x[:max_total_size]

    if len(dedup_score):
      return [
          np.array(len(dedup_score), np.int32),
          _pad_fn(dedup_score),
          _pad_fn(dedup_subject, np.int32),
          _pad_fn(dedup_subject_score),
          _pad_fn(dedup_subject_box),
          _pad_fn(dedup_predicate, np.int32),
          _pad_fn(dedup_predicate_score),
          _pad_fn(dedup_object, np.int32),
          _pad_fn(dedup_object_score),
          _pad_fn(dedup_object_box),
      ]
    else:
      return [
          np.array(0, np.int32),
          np.zeros((max_total_size), np.float32),
          np.zeros((max_total_size), np.int32),
          np.zeros((max_total_size), np.float32),
          np.zeros((max_total_size, 4), np.float32),
          np.zeros((max_total_size), np.int32),
          np.zeros((max_total_size), np.float32),
          np.zeros((max_total_size), np.int32),
          np.zeros((max_total_size), np.float32),
          np.zeros((max_total_size, 4), np.float32),
      ]

  def _per_image_nms(elems):
    return tf.py_func(_py_per_image_nms, elems, [
        tf.int32,
        tf.float32,
        tf.int32,
        tf.float32,
        tf.float32,
        tf.int32,
        tf.float32,
        tf.int32,
        tf.float32,
        tf.float32,
    ])

  batch_outputs = tf.map_fn(_per_image_nms,
                            elems=[
                                n_valid_example, best_scores, ret_subject,
                                ret_subject_score, ret_subject_box,
                                ret_predicate, ret_predicate_score, ret_object,
                                ret_object_score, ret_object_box
                            ],
                            dtype=[
                                tf.int32,
                                tf.float32,
                                tf.int32,
                                tf.float32,
                                tf.float32,
                                tf.int32,
                                tf.float32,
                                tf.int32,
                                tf.float32,
                                tf.float32,
                            ],
                            parallel_iterations=32,
                            back_prop=False)

  (n_valid_example, best_scores, ret_subject, ret_subject_score,
   ret_subject_box, ret_predicate, ret_predicate_score, ret_object,
   ret_object_score, ret_object_box) = batch_outputs

  n_valid_example.set_shape([batch])
  best_scores.set_shape([batch, max_total_size])
  ret_subject.set_shape([batch, max_total_size])
  ret_subject_score.set_shape([batch, max_total_size])
  ret_subject_box.set_shape([batch, max_total_size, 4])
  ret_predicate.set_shape([batch, max_total_size])
  ret_predicate_score.set_shape([batch, max_total_size])
  ret_object.set_shape([batch, max_total_size])
  ret_object_score.set_shape([batch, max_total_size])
  ret_object_box.set_shape([batch, max_total_size, 4])

  return (n_valid_example, best_scores, ret_subject, ret_subject_score,
          ret_subject_box, ret_object, ret_object_score, ret_object_box,
          ret_predicate, ret_predicate_score)
