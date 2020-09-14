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

_EPSILON = 1e-8


class SceneGraphEvaluator(object):
  """Class to evaluate scene graph generation metrics. """

  def __init__(self, iou_threshold=0.5):
    """Constructor. """
    self._image_annotations = {}
    self._metrics = None

    self._iou_threshold = iou_threshold

  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    self._image_annotations.clear()

  def _evaluate(self, image_id, groundtruth, prediction):
    """Evaluates an single image example.

    Args:
      image_id: A unique identifier for the image.
      groundtruth: A dictionary containing the following fields.
        - `subject`, `subject/box`, `predicate`, `object`, `object/box`.
      prediction: A dictionary containing the following fields.
        - `subject`, `subject/box`, `predicate`, `object`, `object/box`.

    Returns:
      n_triples: Number of annotated triples in the groundtruth.
      recall50: Number of recalled groundtruth in the top-50 prediction.
      recall100: Number of recalled groundtruth in the top-100 prediction.
    """
    gt_subject = groundtruth['subject']
    gt_object = groundtruth['object']
    gt_predicate = groundtruth['predicate']
    n_gt = len(gt_subject)

    pred_subject = prediction['subject']
    pred_object = prediction['object']
    pred_predicate = prediction['predicate']
    n_pred = len(pred_subject)

    # Compute the iou between prediction and groundtruth.
    # - `subject_iou` shape = [n_pred, n_gt].
    # - `object_iou` shape = [n_pred, n_gt].
    subject_iou = box_ops.py_iou(np.expand_dims(prediction['subject/box'], 1),
                                 np.expand_dims(groundtruth['subject/box'], 0))
    object_iou = box_ops.py_iou(np.expand_dims(prediction['object/box'], 1),
                                np.expand_dims(groundtruth['object/box'], 0))

    recall50 = 0
    recall100 = 0
    recalled = set()

    for i in range(min(n_pred, 100)):
      for j in range(n_gt):
        if (not j in recalled and pred_subject[i] == gt_subject[j] and
            pred_object[i] == gt_object[j] and
            pred_predicate[i] == gt_predicate[j] and
            subject_iou[i, j] >= self._iou_threshold and
            object_iou[i, j] >= self._iou_threshold):

          recalled.add(j)
          if i < 50:
            recall50 += 1
          if i < 100:
            recall100 += 1
    return n_gt, recall50, recall100

  def evaluate(self):
    """Evaluates the generated scene graphs, returns recall metrics.

    Returns:
      A dictionary holding
      - `SceneGraphs/Recall@50`: Recall with 50 detections.
      - `SceneGraphs/Recall@100`: Recall with 100 detections.
    """
    logging.info('Performing evaluation on %d images.',
                 len(self._image_annotations))

    total_n_triple = 0
    total_recall50 = 0
    total_recall100 = 0

    for image_index, (image_id,
                      image_info) in enumerate(self._image_annotations.items()):
      n_triples, recall50, recall100 = self._evaluate(image_id,
                                                      image_info['groundtruth'],
                                                      image_info['prediction'])
      total_n_triple += n_triples
      total_recall50 += recall50
      total_recall100 += recall100

      if (image_index + 1) % 100 == 0:
        logging.info('Evaluate on %i/%i.', image_index + 1,
                     len(self._image_annotations))

    return {
        'metrics/scene_graph_triplets/recall@50':
            total_recall50 / max(total_n_triple, _EPSILON),
        'metrics/scene_graph_triplets/recall@100':
            total_recall100 / max(total_n_triple, _EPSILON),
        'metrics/scene_graph_triplets/n_example':
            len(self._image_annotations),
    }

  def add_single_ground_truth_image_info(self, image_id, image_info):
    """Adds groundtruth for a single image.

    Args:
      image_id: A unique identifier for the image.
      image_info: A dictionary containing the following fields.
    """
    if image_id in self._image_annotations:
      logging.warning('Ignoring groudtruth for %s.', image_id)
      return
    self._image_annotations[image_id] = {'groundtruth': image_info}

  def add_single_detected_image_info(self, image_id, image_info):
    """Adds detections for a single image.

    Args:
      image_id: A unique identifier for the image.
      image_info: A dictionary containing the following fields.
    """
    if image_id not in self._image_annotations:
      raise ValueError('Missing groundtruth for %s.', image_id)

    if 'prediction' in self._image_annotations[image_id]:
      logging.warning('Ignoring detection for %s.', image_id)
      return

    self._image_annotations[image_id]['prediction'] = image_info

  def add_eval_dict(self, eval_dict):
    """Observes an evaluation result dict for a single example.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating scene graph
        generation performance.

    Returns:
      An update_op that can be used to update the eval metrics in
        tf.estimator.EstimatorSpce.
    """

    def update_op(image_id_batched, gt_n_triple_batched, gt_subject_batched,
                  gt_subject_box_batched, gt_object_batched,
                  gt_object_box_batched, gt_predicate_batched,
                  pred_n_triple_batched, pred_subject_batched,
                  pred_subject_box_batched, pred_object_batched,
                  pred_object_box_batched, pred_predicate_batched):

      for (image_id, gt_n_triple, gt_subject, gt_subject_box, gt_object,
           gt_object_box, gt_predicate, pred_n_triple, pred_subject,
           pred_subject_box,
           pred_object, pred_object_box, pred_predicate) in zip(
               image_id_batched, gt_n_triple_batched, gt_subject_batched,
               gt_subject_box_batched, gt_object_batched, gt_object_box_batched,
               gt_predicate_batched, pred_n_triple_batched,
               pred_subject_batched, pred_subject_box_batched,
               pred_object_batched, pred_object_box_batched,
               pred_predicate_batched):

        self.add_single_ground_truth_image_info(
            image_id, {
                'subject': gt_subject[:gt_n_triple],
                'subject/box': gt_subject_box[:gt_n_triple, :],
                'object': gt_object[:gt_n_triple],
                'object/box': gt_object_box[:gt_n_triple, :],
                'predicate': gt_predicate[:gt_n_triple],
            })
        self.add_single_detected_image_info(
            image_id, {
                'subject': pred_subject[:pred_n_triple],
                'subject/box': pred_subject_box[:pred_n_triple, :],
                'object': pred_object[:pred_n_triple],
                'object/box': pred_object_box[:pred_n_triple, :],
                'predicate': pred_predicate[:pred_n_triple],
            })

    image_id = eval_dict['image_id']

    gt_n_triple = eval_dict['groundtruth/n_triple']
    gt_subject = eval_dict['groundtruth/subject']
    gt_subject_box = eval_dict['groundtruth/subject/box']
    gt_object = eval_dict['groundtruth/object']
    gt_object_box = eval_dict['groundtruth/object/box']
    gt_predicate = eval_dict['groundtruth/predicate']

    pred_n_triple = eval_dict['prediction/n_triple']
    pred_subject = eval_dict['prediction/subject']
    pred_subject_box = eval_dict['prediction/subject/box']
    pred_object = eval_dict['prediction/object']
    pred_object_box = eval_dict['prediction/object/box']
    pred_predicate = eval_dict['prediction/predicate']

    if not image_id.shape.as_list():
      # Apply a batch dimension to all tensors.
      image_id = tf.expand_dims(image_id, 0)

      gt_n_triple = tf.expand_dims(gt_n_triple, 0)
      gt_subject = tf.expand_dims(gt_subject, 0)
      gt_subject_box = tf.expand_dims(gt_subject_box, 0)
      gt_object = tf.expand_dims(gt_object, 0)
      gt_object_box = tf.expand_dims(gt_object_box, 0)
      gt_predicate = tf.expand_dims(gt_predicate, 0)

      pred_n_triple = tf.expand_dims(pred_n_triple, 0)
      pred_subject = tf.expand_dims(pred_subject, 0)
      pred_subject_box = tf.expand_dims(pred_subject_box, 0)
      pred_object = tf.expand_dims(pred_object, 0)
      pred_object_box = tf.expand_dims(pred_object_box, 0)
      pred_predicate = tf.expand_dims(pred_predicate, 0)

    return tf.py_func(update_op, [
        image_id, gt_n_triple, gt_subject, gt_subject_box, gt_object,
        gt_object_box, gt_predicate, pred_n_triple, pred_subject,
        pred_subject_box, pred_object, pred_object_box, pred_predicate
    ], [])

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns a dictionary of eval metric ops.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating scene graph
        generation performance.

    Returns:
      A dictionary of metric names to tule of value_op and update_op that can
        be used as eval metric ops in tf.estimator.EstimatorSpec.
    """
    update_op = self.add_eval_dict(eval_dict)
    metric_names = [
        'metrics/scene_graph_triplets/n_example',
        'metrics/scene_graph_triplets/recall@50',
        'metrics/scene_graph_triplets/recall@100',
    ]

    def first_value_func():
      self._metrics = self.evaluate()
      self.clear()
      return np.float32(self._metrics[metric_names[0]])

    def value_func_factory(metric_name):

      def value_func():
        return np.float32(self._metrics[metric_name])

      return value_func

    first_value_op = tf.py_func(first_value_func, [], tf.float32)
    eval_metric_ops = {metric_names[0]: (first_value_op, update_op)}

    with tf.control_dependencies([first_value_op]):
      for metric_name in metric_names[1:]:
        eval_metric_ops[metric_name] = (tf.py_func(
            value_func_factory(metric_name), [], np.float32), update_op)

    return eval_metric_ops
