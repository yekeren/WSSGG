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


def detect_relations(options, dt):
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

  # Compute relation labels.
  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(dt.batch), 1),
                                [dt.batch, dt.max_n_relation])
  subject_proposal_id = tf.gather_nd(
      dt.refined_grounding.entity_proposal_id,
      tf.stack([index_batch, dt.relation_senders], -1))
  object_proposal_id = tf.gather_nd(
      dt.refined_grounding.entity_proposal_id,
      tf.stack([index_batch, dt.relation_receivers], -1))

  dt.relation_subject_instance_labels, dt.relation_object_instance_labels = [
      _scatter_instance_labels(x,
                               dt.relation_ids,
                               max_n_proposal=dt.max_n_proposal,
                               vocab_size=dt.vocab_size)
      for x in [subject_proposal_id, object_proposal_id]
  ]

  propogation_matrix = tf.cast(
      dt.proposal_iou > options.grounding_iou_threshold, tf.float32)
  dt.relation_subject_instance_labels = tf.matmul(
      propogation_matrix, dt.relation_subject_instance_labels)
  dt.relation_object_instance_labels = tf.matmul(
      propogation_matrix, dt.relation_object_instance_labels)

  # Per-proposal relation prediction.
  (dt.relation_subject_instance_logits, dt.relation_object_instance_logits) = [
      _relation_classify(dt.proposal_features,
                         dt.embeddings,
                         dt.bias_relation,
                         name=name)
      for name in ['relation_subject', 'relation_object']
  ]

  # Relation scores.
  dt.relation_instance_scores = tf.minimum(
      tf.expand_dims(tf.nn.softmax(dt.relation_subject_instance_logits), 2),
      tf.expand_dims(tf.nn.softmax(dt.relation_object_instance_logits), 1))
  dt.relation_instance_scores = tf.concat([
      tf.zeros([dt.batch, dt.max_n_proposal, dt.max_n_proposal, 1]),
      dt.relation_instance_scores[:, :, :, 1:]
  ], -1)

  # Postprocess to provide relations at test time.
  (dt.relation.num_relations, dt.relation.log_prob, dt.relation.relation_score,
   dt.relation.relation_class, dt.relation.subject_proposal,
   dt.relation.subject_score, dt.relation.subject_class,
   dt.relation.object_proposal, dt.relation.object_score,
   dt.relation.object_class) = _postprocess_relations(
       dt.detection.valid_detections,
       dt.detection.nmsed_proposal_id,
       dt.detection.nmsed_scores,
       dt.detection.nmsed_classes,
       dt.relation_instance_scores,
       relation_max_total_size=options.relation_max_total_size,
       relation_max_size_per_class=options.relation_max_size_per_class,
       relation_threshold=options.relation_threshold)

  # Select subject and object boxes.
  proposals = tf.broadcast_to(
      tf.expand_dims(dt.proposals, 1),
      [dt.batch, options.relation_max_total_size, dt.max_n_proposal, 4])
  proposal_features = tf.broadcast_to(tf.expand_dims(dt.proposal_features, 1), [
      dt.batch, options.relation_max_total_size, dt.max_n_proposal,
      dt.proposal_features.shape[-1].value
  ])

  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(dt.batch), 1),
                                [dt.batch, options.relation_max_total_size])
  index_beam = tf.broadcast_to(
      tf.expand_dims(tf.range(options.relation_max_total_size), 0),
      [dt.batch, options.relation_max_total_size])

  index_subject = tf.stack(
      [index_batch, index_beam, dt.relation.subject_proposal], -1)
  dt.relation.subject_box = tf.gather_nd(proposals, index_subject)
  dt.relation.subject_feature = tf.gather_nd(proposal_features, index_subject)

  index_object = tf.stack(
      [index_batch, index_beam, dt.relation.object_proposal], -1)
  dt.relation.object_box = tf.gather_nd(proposals, index_object)
  dt.relation.object_feature = tf.gather_nd(proposal_features, index_object)
  return dt


def _relation_classify(proposal_features, embeddings, bias, name=None):
  """Predicts classes based on proposal features.

  Args:
    proposal_feature: A [batch, max_n_proposal, feature_dims] float tensor.
    embeddings: Embedding matrix, a [vocab_size, dims] float tensor.
    bias: A [vocab_size] float tensor.

  Returns:
    relation_logits: Logits tensor, a [batch, max_n_proposal, vocab_size] float tensor.
    relation_scores: Normalized scores, a [batch, max_n_proposal, vocab_size] float tensor.
  """
  relation_head = tf.layers.Dense(
      embeddings.shape[-1].value,
      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
      name=name)(proposal_features)
  relation_logits = tf.matmul(relation_head, embeddings, transpose_b=True)
  return tf.nn.bias_add(relation_logits, bias)


def _scatter_relation_labels(subject_proposal_id, object_proposal_id,
                             relation_id, max_n_proposal, vocab_size):
  """Scatters relation labels.

  Args:
    subject_proposal_id: A [batch, max_n_edge] int tensor.
    object_proposal_id: A [batch, max_n_edge] int tensor.
    relation_id: A [batch, max_n_edge] int tensor.
    max_n_proposal: Maximum number of proposals.
    vocab_size: Size of the vocabulary.

  Returns:
    A [batch, max_n_proposal, max_n_proposal, vocab_size] tensor.
  """
  batch = relation_id.shape[0].value
  max_n_edge = tf.shape(relation_id)[1]

  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_edge])
  index_full = tf.stack(
      [index_batch, subject_proposal_id, object_proposal_id, relation_id], -1)
  return tf.scatter_nd(
      index_full,
      updates=tf.fill([batch, max_n_edge], 1.0),
      shape=[batch, max_n_proposal, max_n_proposal, vocab_size])


def _postprocess_relations(num_detections,
                           detection_proposal,
                           detection_scores,
                           detection_classes,
                           relation_scores,
                           relation_max_total_size=100,
                           relation_max_size_per_class=2,
                           relation_threshold=0.0):
  """Postprocesses the relation detection.

  Args:
    num_detections: Number of entity detections, a [batch] int tensor.
    detection_proposal: Detected proposal index, a [batch, max_n_detection]
      int tensor, each value is in the range [0, max_n_proposal).
    detection_scores: Detection scores, a [batch, max_n_detection] float tensor.
    detection_classes: Detection classes, a [batch, max_n_detection] int tensor.
    relation_scores: Per-proposal-pair relation scores, a [batch, max_n_proposal,
      max_n_proposal, vocab_size] float tensor.

  Returns:
    num_relations: Number of detected relations, a [batch] int tensor.
    log_prob: Log probability of the (subject, relation, object) tuple, a [batch, relation_max_total_size] float tensor.
    relation_score: Detected relation score, a [batch, relation_max_total_size] float tensor.
    relation_class: Detected relation class, a [batch, relation_max_total_size] int tensor.
    subject_proposal: Index of the detected subject proposal, a [batch, relation_max_total_size] int tensor.
    subject_score: Subject score, a [batch, relation_max_total_size] float tensor.
    subject_class: Subject class, a [batch, relation_max_total_size] int tensor.
    object_proposal: Index of the detected object proposal, a [batch, relation_max_total_size] int tensor.
    object_score: Object score, a [batch, relation_max_total_size] float tensor.
    object_class: Object class, a [batch, relation_max_total_size] int tensor.
  """

  def _py_per_image_relation_search(num_detections, detection_proposal,
                                    detection_scores, detection_classes,
                                    relation_scores):
    """
    Args:
      num_detections: An integer.
      detection_proposal: A [max_n_detection] int array.
      detection_scores: A [max_n_detection] float array.
      detection_classes : A [max_n_detection] int array.
      relation_scores: A [max_n_proposal, max_n_proposal, vocab_size] float array.
    """
    # relation_topk_indices shape
    # = [max_n_proposal, max_n_proposal, relation_max_size_per_class].
    relation_topk_indices = np.argpartition(
        -relation_scores, relation_max_size_per_class,
        axis=-1)[:, :, :relation_max_size_per_class]

    heap = []
    for i in range(num_detections):
      for j in range(num_detections):
        if detection_proposal[i] == detection_proposal[j]:
          # We care relations between different boxes.
          continue
        for relation_id in relation_topk_indices[detection_proposal[i],
                                                 detection_proposal[j]]:
          relation_score = relation_scores[detection_proposal[i],
                                           detection_proposal[j], relation_id]
          # Relation score is not strong.
          if relation_score <= relation_threshold:
            continue
          log_prob = (np.log(max(1e-6, detection_scores[i])) +
                      np.log(max(1e-6, detection_scores[j])) +
                      np.log(max(1e-6, relation_score)))
          if len(heap) < relation_max_total_size or log_prob > heap[0][0]:
            heapq.heappush(heap, [
                log_prob,
                relation_score,
                relation_id,
                detection_proposal[i],
                detection_scores[i],
                detection_classes[i],
                detection_proposal[j],
                detection_scores[j],
                detection_classes[j],
            ])
            if len(heap) > relation_max_total_size:
              heapq.heappop(heap)

    # Stack results.
    values = [heapq.heappop(heap) for i in range(len(heap))][::-1]
    return _pad_values(values)

  def _pad_values(values):

    n_relations = len(values)
    values = list(zip(*values))

    def _assign_value(array, py_list):
      for i, val in enumerate(py_list):
        array[i] = val

    def _init_zero_int(): return np.zeros((relation_max_total_size), np.int32)

    def _init_zero_float(): return np.zeros((relation_max_total_size), np.float32)

    num_relations = np.array(n_relations, np.int32)
    log_prob = _init_zero_float()
    relation_score = _init_zero_float()
    relation_class = _init_zero_int()
    subject_proposal = _init_zero_int()
    subject_score = _init_zero_float()
    subject_class = _init_zero_int()
    object_proposal = _init_zero_int()
    object_score = _init_zero_float()
    object_class = _init_zero_int()

    if n_relations:
      _assign_value(log_prob, values[0])
      _assign_value(relation_score, values[1])
      _assign_value(relation_class, values[2])
      _assign_value(subject_proposal, values[3])
      _assign_value(subject_score, values[4])
      _assign_value(subject_class, values[5])
      _assign_value(object_proposal, values[6])
      _assign_value(object_score, values[7])
      _assign_value(object_class, values[8])

    return [
        num_relations, log_prob, relation_score, relation_class,
        subject_proposal, subject_score, subject_class, object_proposal,
        object_score, object_class
    ]

  output_types = [
      tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32,
      tf.int32, tf.int32, tf.float32, tf.int32
  ]

  def _per_image_relation_search(elems):
    return tf.py_func(_py_per_image_relation_search, elems, output_types)

  batch_outputs = tf.map_fn(_per_image_relation_search,
                            elems=[
                                num_detections, detection_proposal,
                                detection_scores, detection_classes,
                                relation_scores
                            ],
                            dtype=output_types,
                            parallel_iterations=1,
                            back_prop=False)
  batch = num_detections.shape[0].value
  for output in batch_outputs[1:]:
    output.set_shape([batch, relation_max_total_size])
  return batch_outputs


def _scatter_instance_labels(proposal_id, instance_id, max_n_proposal,
                             vocab_size):
  """Creates instance-level entity/relation labels.

  Args:
    proposal_id: A [batch, max_n_inst] int tensor, denoting the proposal id of each instance, values are in the range [0, max_n_proposal).
    instance_id: Either entity or relation ids. A [batch, max_n_inst] int tensor, values are in [0, vocab_size).
    max_n_proposal: Maximum number of proposals.
    vocab_size: Size of the vocabulary.

  Returns:
    A [batch, max_n_proposal, vocab_size] tensor.
  """
  batch = proposal_id.shape[0].value
  max_n_inst = tf.shape(instance_id)[1]

  index_batch = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_inst])
  index_full = tf.stack([index_batch, proposal_id, instance_id], -1)
  return tf.scatter_nd(index_full,
                       updates=tf.fill([batch, max_n_inst], 1.0),
                       shape=[batch, max_n_proposal, vocab_size])
