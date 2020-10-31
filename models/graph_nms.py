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

import heapq
import numpy as np
from modeling.utils import masked_ops
from modeling.utils import box_ops

from models import utils


class GraphNMS(object):
  """Looks for the NMSed solution. """

  def __init__(self,
               n_proposal,
               proposals,
               proposal_scores,
               relation_scores,
               max_size_per_class=2,
               max_total_size=100,
               iou_thresh=0.5,
               score_thresh=0.01,
               use_class_agnostic_nms=False,
               use_log_prob=False):
    """Initializes the object.

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.
      proposal_scores: A [batch, max_n_proposal, n_entity] float tensor.
      relation_scores: A [batch, max_n_proposal, max_n_proposal, n_predicate] float tensor.
      max_total_size: Beam size for the searching.
      parallel_iterations: Number of batch items to process in parallel.
    """
    (num_detections, detection_boxes, detection_scores, detection_classes,
     detection_indices) = utils.nms_post_process(
         n_proposal,
         proposals,
         proposal_scores,
         max_size_per_class=max_size_per_class,
         max_total_size=max_total_size,
         iou_thresh=iou_thresh,
         use_class_agnostic_nms=use_class_agnostic_nms,
         score_thresh=score_thresh)

    self.num_detections = num_detections
    self.detection_boxes = detection_boxes
    self.detection_scores = detection_scores
    self.detection_classes = detection_classes
    self.detection_indices = detection_indices

    def _py_per_image_relation_search(num_detections, detection_indices,
                                      detection_classes, detection_scores,
                                      relation_scores):
      """Computes the score.

      Args:
        num_detections: An integer.
        detection_indices: A [max_detections] float array.
        detection_classes: A [max_detections] int array.
        detection_scores: A [max_detections] float array.
        relation_scores: A [max_n_proposal, max_n_proposal, n_predicate] float tensor.
      """
      detection_indices = detection_indices[:num_detections]
      detection_classes = detection_classes[:num_detections]
      detection_scores = detection_scores[:num_detections]
      n_predicate = relation_scores.shape[-1]

      relation_kth_indices = np.argpartition(-relation_scores,
                                             max_size_per_class,
                                             axis=-1)
      relation_kth_indices = relation_kth_indices[:, :, :max_size_per_class]

      h = []
      for subject_proposal_index, subject_entity_index, subject_score in zip(
          detection_indices, detection_classes, detection_scores):
        for object_proposal_index, object_entity_index, object_score in zip(
            detection_indices, detection_classes, detection_scores):

          if subject_proposal_index == object_proposal_index:
            continue

          #for predicate_index in range(n_predicate):
          for predicate_index in relation_kth_indices[subject_proposal_index,
                                                      object_proposal_index]:

            predicate_score = relation_scores[subject_proposal_index,
                                              object_proposal_index,
                                              predicate_index]
            if predicate_score < score_thresh:
              # Relation score is not strong.
              continue

            if use_log_prob:
              hscore = (np.log(max(1e-10, subject_score)) +
                        np.log(max(1e-10, object_score)) +
                        np.log(max(1e-10, predicate_score)))
            else:
              hscore = subject_score + object_score + predicate_score
            if 0 <= len(h) < max_total_size or hscore > h[0][0]:
              if len(h) == max_total_size:
                heapq.heappop(h)
              heapq.heappush(h, [
                  hscore, subject_score, subject_proposal_index,
                  subject_entity_index, predicate_score, predicate_index,
                  object_score, object_proposal_index, object_entity_index
              ])

      # Stack results.
      values = [heapq.heappop(h) for i in range(len(h))][::-1]

      if values:
        (hscore, subject_score, subject_proposal_index, subject_entity_index,
         predicate_score, predicate_index, object_score, object_proposal_index,
         object_entity_index) = zip(*values)

        # Padding.
        if len(values) < max_total_size:
          n_pad = max_total_size - len(values)
          hscore = list(hscore) + [0] * n_pad
          subject_score = list(subject_score) + [0] * n_pad
          subject_proposal_index = list(subject_proposal_index) + [0] * n_pad
          subject_entity_index = list(subject_entity_index) + [0] * n_pad
          predicate_score = list(predicate_score) + [0] * n_pad
          predicate_index = list(predicate_index) + [0] * n_pad
          object_score = list(object_score) + [0] * n_pad
          object_proposal_index = list(object_proposal_index) + [0] * n_pad
          object_entity_index = list(object_entity_index) + [0] * n_pad

      else:
        hscore = np.zeros((max_total_size), np.float32)
        subject_score = np.zeros((max_total_size), np.float32)
        subject_proposal_index = np.zeros((max_total_size), np.int32)
        subject_entity_index = np.zeros((max_total_size), np.int32)
        predicate_score = np.zeros((max_total_size), np.float32)
        predicate_index = np.zeros((max_total_size), np.int32)
        object_score = np.zeros((max_total_size), np.float32)
        object_proposal_index = np.zeros((max_total_size), np.int32)
        object_entity_index = np.zeros((max_total_size), np.int32)

      return [
          np.array(len(values), np.int32),
          np.array(hscore, np.float32),
          np.array(subject_score, np.float32),
          np.array(subject_proposal_index, np.int32),
          np.array(subject_entity_index, np.int32),
          np.array(predicate_score, np.float32),
          np.array(predicate_index, np.int32),
          np.array(object_score, np.float32),
          np.array(object_proposal_index, np.int32),
          np.array(object_entity_index, np.int32)
      ]

    def _per_image_relation_search(elems):
      return tf.py_func(_py_per_image_relation_search, elems, [
          tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32,
          tf.int32, tf.float32, tf.int32, tf.int32
      ])

    batch_outputs = tf.map_fn(_per_image_relation_search,
                              elems=[
                                  num_detections, detection_indices,
                                  detection_classes, detection_scores,
                                  relation_scores
                              ],
                              dtype=[
                                  tf.int32, tf.float32, tf.float32, tf.int32,
                                  tf.int32, tf.float32, tf.int32, tf.float32,
                                  tf.int32, tf.int32
                              ],
                              parallel_iterations=32,
                              back_prop=False)
    batch = n_proposal.shape[0].value
    batch_outputs[0].set_shape([batch])
    for i in range(1, len(batch_outputs)):
      batch_outputs[i].set_shape([batch, max_total_size])

    (self._n_triple, self._triple_score, self._subject_score,
     self._subject_proposal_index, self._subject_entity_index,
     self._predicate_score, self._predicate_index, self._object_score,
     self._object_proposal_index, self._object_entity_index) = batch_outputs

  @property
  def n_triple(self):
    """Returns number of triples.

    Returns:
      A [batch] int tensor, the number of detected triples.
    """
    return self._n_triple

  @property
  def triple_score(self):
    """Returns triplet score.

    Returns:
      A [batch, max_total_size] float tensor, each value denotes a score.
    """
    return self._triple_score

  @property
  def subject_score(self):
    """Returns triplet score.

    Returns:
      A [batch, max_total_size] float tensor, each value denotes a score.
    """
    return self._subject_score

  @property
  def object_score(self):
    """Returns triplet score.

    Returns:
      A [batch, max_total_size] float tensor, each value denotes a score.
    """
    return self._object_score

  @property
  def predicate_score(self):
    """Returns triplet score.

    Returns:
      A [batch, max_total_size] float tensor, each value denotes a score.
    """
    return self._predicate_score

  @property
  def subject_proposal_index(self):
    """Returns subject proposal index.
    
    Returns:
      A [batch, max_total_size] int tensor, each value denotes a proposal index.
    """
    return self._subject_proposal_index

  @property
  def object_proposal_index(self):
    """Returns object proposal index.
    
    Returns:
      A [batch, max_total_size] int tensor, each value denotes a proposal index.
    """
    return self._object_proposal_index

  @property
  def subject_id(self):
    """Returns subject id.
    
    Returns:
      A [batch, max_total_size] int tensor, each value denotes a entity index.
    """
    return self._subject_entity_index

  @property
  def object_id(self):
    """Returns object id.
    
    Returns:
      A [batch, max_total_size] int tensor, each value denotes a entity index.
    """
    return self._object_entity_index

  @property
  def predicate_id(self):
    """Returns predicate id.
    
    Returns:
      A [batch, max_total_size] int tensor, each value denotes a predicate index.
    """
    return self._predicate_index

  def get_subject_box(self, proposals):
    """Returns the subject box.

    Args:
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      A [batch, max_total_size, 4] float tensor.
    """
    return self._gather_proposal_by_index(proposals,
                                          self.subject_proposal_index)

  def get_object_box(self, proposals):
    """Returns the object box.

    Args:
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      A [batch, max_total_size, 4] float tensor.
    """
    return self._gather_proposal_by_index(proposals, self.object_proposal_index)

  def get_subject_feature(self, proposal_features):
    """Returns the subject feature.

    Args:
      proposal_features: A [batch, max_n_proposal, dims] float tensor.

    Returns:
      A [batch, max_n_triple, dims] float tensor.
    """
    return self._gather_proposal_by_index(proposal_features,
                                          self.subject_proposal_index)

  def get_object_feature(self, proposal_features):
    """Returns the object feature.

    Args:
      proposal_features: A [batch, max_n_proposal, dims] float tensor.

    Returns:
      A [batch, max_n_triple, dims] float tensor.
    """
    return self._gather_proposal_by_index(proposal_features,
                                          self.object_proposal_index)

  @staticmethod
  def _gather_proposal_by_index(proposals, proposal_index):
    """Gathers proposal box or proposal features by index..
  
    This is a helper function to extract beam-search solution.
  
    Args:
      proposals: A [batch, max_n_proposal, dims] float tensor.
        It could be either proposal box (dims=4) or proposal features.
      proposal_index: A [batch, max_total_size] int tensor.
  
    Returns:
      A [batch, max_total_size, dims] float tensor, the gathered proposal info,
        could be proposal boxes or proposal features.
    """
    batch = proposals.shape[0].value
    dims = proposals.shape[-1].value
    max_total_size = proposal_index.shape[1].value
    max_n_proposal = tf.shape(proposals)[1]

    proposals = tf.broadcast_to(tf.expand_dims(proposals, 1),
                                [batch, max_total_size, max_n_proposal, dims])

    batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                  [batch, max_total_size])
    beam_index = tf.broadcast_to(tf.expand_dims(tf.range(max_total_size), 0),
                                 [batch, max_total_size])
    index = tf.stack([batch_index, beam_index, proposal_index], -1)
    return tf.gather_nd(proposals, index)

  @staticmethod
  def _compute_iou(n_proposal, proposals):
    """Computes IoU. 

    Args:
      n_proposal: A [batch] int tensor.
      proposals: A [batch, max_n_proposal, 4] float tensor.

    Returns:
      iou: A [batch, max_n_proposal, max_n_proposal] float tensor.
    """
    batch = proposals.shape[0].value
    max_n_proposal = tf.shape(proposals)[1]

    proposal1 = tf.broadcast_to(tf.expand_dims(proposals, 2),
                                [batch, max_n_proposal, max_n_proposal, 4])
    proposal2 = tf.broadcast_to(tf.expand_dims(proposals, 1),
                                [batch, max_n_proposal, max_n_proposal, 4])
    return box_ops.iou(proposal1, proposal2)
