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

import numpy as np
import tensorflow as tf
from models.scene_graph_evaluation import SceneGraphEvaluator


class SceneGraphEvaluatorTest(tf.test.TestCase):

  def test_evaluate_single_example1(self):
    evaluator = SceneGraphEvaluator()
    evaluator.add_single_ground_truth_image_info(
        0, {
            'subject': ['person', 'person'],
            'subject/box':
                np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on', 'wear'],
            'object': ['skateboard', 'shirt'],
            'object/box':
                np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]], dtype=np.float32),
        })
    evaluator.add_single_detected_image_info(
        0, {
            'subject': ['person'],
            'subject/box': np.array([[0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on'],
            'object': ['skateboard'],
            'object/box': np.array([[1, 1, 2, 2]], dtype=np.float32),
        })

    metrics = evaluator.evaluate()
    self.assertEqual(metrics['scene_graph_n_example'], 1)
    self.assertAlmostEqual(metrics['scene_graph_recall@50'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_recall@100'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@50'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@100'], 0.5)

  def test_evaluate_single_example2(self):
    evaluator = SceneGraphEvaluator()
    evaluator.add_single_ground_truth_image_info(
        0, {
            'subject': ['person', 'person'],
            'subject/box':
                np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on', 'wear'],
            'object': ['skateboard', 'shirt'],
            'object/box':
                np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]], dtype=np.float32),
        })
    evaluator.add_single_detected_image_info(
        0, {
            'subject': ['person'],
            'subject/box': np.array([[0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on'],
            'object': ['skateboard'],
            'object/box': np.array([[1, 1, 1.5, 1.5]], dtype=np.float32),
        })

    metrics = evaluator.evaluate()
    self.assertEqual(metrics['scene_graph_n_example'], 1)
    self.assertAlmostEqual(metrics['scene_graph_recall@50'], 0.0)
    self.assertAlmostEqual(metrics['scene_graph_recall@100'], 0.0)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@50'], 0.0)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@100'], 0.0)

  def test_evaluate_single_example2_threshold(self):
    evaluator = SceneGraphEvaluator(iou_threshold=0.25)
    evaluator.add_single_ground_truth_image_info(
        0, {
            'subject': ['person', 'person'],
            'subject/box':
                np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on', 'wear'],
            'object': ['skateboard', 'shirt'],
            'object/box':
                np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]], dtype=np.float32),
        })
    evaluator.add_single_detected_image_info(
        0, {
            'subject': ['person'],
            'subject/box': np.array([[0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on'],
            'object': ['skateboard'],
            'object/box': np.array([[1, 1, 1.5, 1.5]], dtype=np.float32),
        })

    metrics = evaluator.evaluate()
    self.assertEqual(metrics['scene_graph_n_example'], 1)
    self.assertAlmostEqual(metrics['scene_graph_recall@50'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_recall@100'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@50'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@100'], 0.5)

  def test_evaluate_single_example3(self):
    evaluator = SceneGraphEvaluator()
    evaluator.add_single_ground_truth_image_info(
        0, {
            'subject': ['person', 'person'],
            'subject/box':
                np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on', 'wear'],
            'object': ['skateboard', 'shirt'],
            'object/box':
                np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]], dtype=np.float32),
        })
    evaluator.add_single_detected_image_info(
        0, {
            'subject': ['person'],
            'subject/box': np.array([[0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['wear'],
            'object': ['shirt'],
            'object/box': np.array([[0, 0, 0.5, 0.5]], dtype=np.float32),
        })

    metrics = evaluator.evaluate()
    self.assertEqual(metrics['scene_graph_n_example'], 1)
    self.assertAlmostEqual(metrics['scene_graph_recall@50'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_recall@100'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@50'], 0.5)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@100'], 0.5)

  def test_evaluate_single_example4(self):
    evaluator = SceneGraphEvaluator()
    evaluator.add_single_ground_truth_image_info(
        0, {
            'subject': ['person', 'person'],
            'subject/box':
                np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['on', 'wear'],
            'object': ['skateboard', 'shirt'],
            'object/box':
                np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]], dtype=np.float32),
        })
    evaluator.add_single_detected_image_info(
        0, {
            'subject': ['person', 'person'],
            'subject/box':
                np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
            'predicate': ['wear', 'on'],
            'object': ['shirt', 'skateboard'],
            'object/box':
                np.array([[0, 0, 0.5, 0.5], [1, 1, 1.5, 2]], dtype=np.float32),
        })

    metrics = evaluator.evaluate()
    self.assertEqual(metrics['scene_graph_n_example'], 1)
    self.assertAlmostEqual(metrics['scene_graph_recall@50'], 1.0)
    self.assertAlmostEqual(metrics['scene_graph_recall@100'], 1.0)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@50'], 1.0)
    self.assertAlmostEqual(metrics['scene_graph_per_image_recall@100'], 1.0)

  def test_get_estimator_eval_metric_ops(self):
    image_id = tf.placeholder(tf.int64, shape=[])

    gt_n_triple = tf.placeholder(tf.int32, shape=[])
    gt_subject = tf.placeholder(tf.string, shape=[None])
    gt_subject_box = tf.placeholder(tf.float32, shape=[None, 4])
    gt_object = tf.placeholder(tf.string, shape=[None])
    gt_object_box = tf.placeholder(tf.float32, shape=[None, 4])
    gt_predicate = tf.placeholder(tf.string, shape=[None])

    pred_n_triple = tf.placeholder(tf.int32, shape=[])
    pred_subject = tf.placeholder(tf.string, shape=[None])
    pred_subject_box = tf.placeholder(tf.float32, shape=[None, 4])
    pred_object = tf.placeholder(tf.string, shape=[None])
    pred_object_box = tf.placeholder(tf.float32, shape=[None, 4])
    pred_predicate = tf.placeholder(tf.string, shape=[None])

    eval_dict = {
        'image_id': image_id,
        'groundtruth/n_triple': gt_n_triple,
        'groundtruth/subject': gt_subject,
        'groundtruth/subject/box': gt_subject_box,
        'groundtruth/object': gt_object,
        'groundtruth/object/box': gt_object_box,
        'groundtruth/predicate': gt_predicate,
        'prediction/n_triple': pred_n_triple,
        'prediction/subject': pred_subject,
        'prediction/subject/box': pred_subject_box,
        'prediction/object': pred_object,
        'prediction/object/box': pred_object_box,
        'prediction/predicate': pred_predicate,
    }

    evaluator = SceneGraphEvaluator()
    eval_metric_ops = evaluator.get_estimator_eval_metric_ops(eval_dict)
    _, update_op = eval_metric_ops['scene_graph_recall@50']

    with self.test_session() as sess:
      sess.run(update_op,
               feed_dict={
                   image_id:
                       0,
                   gt_n_triple:
                       2,
                   gt_subject: ['person', 'person'],
                   gt_subject_box:
                       np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
                   gt_predicate: ['on', 'wear'],
                   gt_object: ['skateboard', 'shirt'],
                   gt_object_box:
                       np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]],
                                dtype=np.float32),
                   pred_n_triple:
                       1,
                   pred_subject: ['person'],
                   pred_subject_box:
                       np.array([[0, 0, 1, 1]], dtype=np.float32),
                   pred_predicate: ['on'],
                   pred_object: ['skateboard'],
                   pred_object_box:
                       np.array([[1, 1, 2, 1.5]], dtype=np.float32),
               })

      recall50 = sess.run(eval_metric_ops['scene_graph_recall@50'][0])
      self.assertAlmostEqual(0.5, recall50)

  def test_get_estimator_eval_metric_ops_gt_padding(self):
    image_id = tf.placeholder(tf.int64, shape=[])

    gt_n_triple = tf.placeholder(tf.int32, shape=[])
    gt_subject = tf.placeholder(tf.string, shape=[None])
    gt_subject_box = tf.placeholder(tf.float32, shape=[None, 4])
    gt_object = tf.placeholder(tf.string, shape=[None])
    gt_object_box = tf.placeholder(tf.float32, shape=[None, 4])
    gt_predicate = tf.placeholder(tf.string, shape=[None])

    pred_n_triple = tf.placeholder(tf.int32, shape=[])
    pred_subject = tf.placeholder(tf.string, shape=[None])
    pred_subject_box = tf.placeholder(tf.float32, shape=[None, 4])
    pred_object = tf.placeholder(tf.string, shape=[None])
    pred_object_box = tf.placeholder(tf.float32, shape=[None, 4])
    pred_predicate = tf.placeholder(tf.string, shape=[None])

    eval_dict = {
        'image_id': image_id,
        'groundtruth/n_triple': gt_n_triple,
        'groundtruth/subject': gt_subject,
        'groundtruth/subject/box': gt_subject_box,
        'groundtruth/object': gt_object,
        'groundtruth/object/box': gt_object_box,
        'groundtruth/predicate': gt_predicate,
        'prediction/n_triple': pred_n_triple,
        'prediction/subject': pred_subject,
        'prediction/subject/box': pred_subject_box,
        'prediction/object': pred_object,
        'prediction/object/box': pred_object_box,
        'prediction/predicate': pred_predicate,
    }

    evaluator = SceneGraphEvaluator()
    eval_metric_ops = evaluator.get_estimator_eval_metric_ops(eval_dict)
    _, update_op = eval_metric_ops['scene_graph_recall@50']

    with self.test_session() as sess:
      sess.run(update_op,
               feed_dict={
                   image_id:
                       0,
                   gt_n_triple:
                       1,
                   gt_subject: ['person', 'person'],
                   gt_subject_box:
                       np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32),
                   gt_predicate: ['on', 'wear'],
                   gt_object: ['skateboard', 'shirt'],
                   gt_object_box:
                       np.array([[1, 1, 2, 2], [0, 0, 0.5, 0.5]],
                                dtype=np.float32),
                   pred_n_triple:
                       1,
                   pred_subject: ['person'],
                   pred_subject_box:
                       np.array([[0, 0, 1, 1]], dtype=np.float32),
                   pred_predicate: ['on'],
                   pred_object: ['skateboard'],
                   pred_object_box:
                       np.array([[1, 1, 2, 1.5]], dtype=np.float32),
               })

      recall50 = sess.run(eval_metric_ops['scene_graph_recall@50'][0])
      self.assertAlmostEqual(1.0, recall50)


if __name__ == '__main__':
  tf.test.main()
