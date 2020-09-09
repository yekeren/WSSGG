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

from models import utils

tf.compat.v1.enable_eager_execution()


class UtilsTest(tf.test.TestCase):

  def test_compute_max_path_sum_batch1_triple1(self):
    """ Basic test case.
           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=5, p2-p3=4
      p0   p1     p2    p3
      \1    |1    |1   /2
            `object`
      solution: 8, subject -> p2 -> p3 -> object
    """

    n_triple, n_proposal, max_n_triple, max_n_proposal = 1, 4, 1, 4
    subject_to_proposal = np.array([4, 3, 2, 1], dtype=np.float32)
    proposal_to_object = np.array([1, 1, 1, 2], dtype=np.float32)
    proposal_to_proposal = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 5, 1, 1]],
        dtype=np.float32)

    (max_path_sum, subject_proposal_index,
     object_proposal_index) = utils.compute_max_path_sum(
         n_proposal=np.array([n_proposal]),
         n_triple=np.array([n_triple]),
         subject_to_proposal=subject_to_proposal.reshape(
             (1, max_n_triple, max_n_proposal)),
         proposal_to_proposal=proposal_to_proposal.reshape(
             (1, max_n_triple, max_n_proposal, max_n_proposal)),
         proposal_to_object=proposal_to_object.reshape(1, max_n_triple,
                                                       max_n_proposal))
    self.assertAllClose(max_path_sum, [[8]])
    self.assertAllEqual(subject_proposal_index, [[2]])
    self.assertAllEqual(object_proposal_index, [[3]])

  def test_compute_max_path_sum_batch1_triple1_mask_proposal_test1(self):
    """ Same input as the previous one but n_proposal(3) < max_n_proposal(4).
           `subject`
       /4   |3    |2
      p0   p1     p2
       | X  |  X  |   all 1
      p0   p1     p2
      \1    |1    |1
            `object`
      solution: 6, subject -> p0 -> p0 -> object
    """

    n_triple, n_proposal, max_n_triple, max_n_proposal = 1, 3, 1, 4
    subject_to_proposal = np.array([4, 3, 2, 1], dtype=np.float32)
    proposal_to_object = np.array([1, 1, 1, 2], dtype=np.float32)
    proposal_to_proposal = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 5, 1, 1]],
        dtype=np.float32)

    (max_path_sum, subject_proposal_index,
     object_proposal_index) = utils.compute_max_path_sum(
         n_proposal=np.array([n_proposal]),
         n_triple=np.array([n_triple]),
         subject_to_proposal=subject_to_proposal.reshape(
             (1, max_n_triple, max_n_proposal)),
         proposal_to_proposal=proposal_to_proposal.reshape(
             (1, max_n_triple, max_n_proposal, max_n_proposal)),
         proposal_to_object=proposal_to_object.reshape(1, max_n_triple,
                                                       max_n_proposal))
    self.assertAllClose(max_path_sum, [[6]])
    self.assertAllEqual(subject_proposal_index, [[0]])
    self.assertAllEqual(object_proposal_index, [[0]])

  def test_compute_max_path_sum_batch1_triple1_mask_proposal_test2(self):
    """ Same input as the previous one but n_proposal(3) < max_n_proposal(4).
           `subject`
       /4   |3    |2
      p0   p1     p2
       | X  |  X  |   all 1
      p0   p1     p2
      \1    |1    |2
            `object`
      solution: 7, subject -> p0 -> p2 -> object
    """

    n_triple, n_proposal, max_n_triple, max_n_proposal = 1, 3, 1, 4
    subject_to_proposal = np.array([4, 3, 2, 1], dtype=np.float32)
    proposal_to_object = np.array([1, 1, 2, 2], dtype=np.float32)
    proposal_to_proposal = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 5, 1, 1]],
        dtype=np.float32)

    (max_path_sum, subject_proposal_index,
     object_proposal_index) = utils.compute_max_path_sum(
         n_proposal=np.array([n_proposal]),
         n_triple=np.array([n_triple]),
         subject_to_proposal=subject_to_proposal.reshape(
             (1, max_n_triple, max_n_proposal)),
         proposal_to_proposal=proposal_to_proposal.reshape(
             (1, max_n_triple, max_n_proposal, max_n_proposal)),
         proposal_to_object=proposal_to_object.reshape(1, max_n_triple,
                                                       max_n_proposal))
    self.assertAllClose(max_path_sum, [[7]])
    self.assertAllEqual(subject_proposal_index, [[0]])
    self.assertAllEqual(object_proposal_index, [[2]])

  def test_compute_max_path_sum_batch1_triple2(self):
    """ Deals with two triples.
           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=5, p2-p3=4
      p0   p1     p2    p3
      \1    |1    |1   /2
            `object`
      solution: 8, subject -> p2 -> p3 -> object

           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=4, p2-p3=4
      p0   p1     p2    p3
      \1    |2    |1   /1
            `object`
      solution: 7, subject -> p0 -> p1 -> object
    """

    n_triple, n_proposal, max_n_triple, max_n_proposal = 2, 4, 2, 4
    subject_to_proposal_1 = np.array([4, 3, 2, 1], dtype=np.float32)
    subject_to_proposal_2 = np.array([4, 3, 2, 1], dtype=np.float32)
    proposal_to_object_1 = np.array([1, 1, 1, 2], dtype=np.float32)
    proposal_to_object_2 = np.array([1, 2, 1, 1], dtype=np.float32)
    proposal_to_proposal_1 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 5, 1, 1]],
        dtype=np.float32)
    proposal_to_proposal_2 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 4, 1, 1]],
        dtype=np.float32)

    (max_path_sum, subject_proposal_index,
     object_proposal_index) = utils.compute_max_path_sum(
         n_proposal=np.array([n_proposal]),
         n_triple=np.array([n_triple]),
         subject_to_proposal=np.expand_dims(
             np.stack([subject_to_proposal_1, subject_to_proposal_2], 0), 0),
         proposal_to_proposal=np.expand_dims(
             np.stack([proposal_to_proposal_1, proposal_to_proposal_2], 0), 0),
         proposal_to_object=np.expand_dims(
             np.stack([proposal_to_object_1, proposal_to_object_2], 0), 0))
    self.assertAllClose(max_path_sum, [[8, 7]])
    self.assertAllEqual(subject_proposal_index, [[2, 0]])
    self.assertAllEqual(object_proposal_index, [[3, 1]])

  def test_compute_max_path_sum_batch1_triple3(self):
    """ Deals with three triples.
           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=5, p2-p3=4
      p0   p1     p2    p3
      \1    |1    |1   /2
            `object`
      solution: 8, subject -> p2 -> p3 -> object

           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=4, p2-p3=4
      p0   p1     p2    p3
      \1    |2    |1   /1
            `object`
      solution: 7, subject -> p0 -> p1 -> object

           `subject`
       /3   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=4, p2-p3=4
      p0   p1     p2    p3
      \1    |2    |1   /1
            `object`
      solution: 7, subject -> p3 -> p1 -> object
    """

    n_triple, n_proposal, max_n_triple, max_n_proposal = 3, 4, 3, 4
    subject_to_proposal_1 = np.array([4, 3, 2, 1], dtype=np.float32)
    subject_to_proposal_2 = np.array([4, 3, 2, 1], dtype=np.float32)
    subject_to_proposal_3 = np.array([3, 3, 2, 1], dtype=np.float32)
    proposal_to_object_1 = np.array([1, 1, 1, 2], dtype=np.float32)
    proposal_to_object_2 = np.array([1, 2, 1, 1], dtype=np.float32)
    proposal_to_object_3 = np.array([1, 2, 1, 1], dtype=np.float32)
    proposal_to_proposal_1 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 5, 1, 1]],
        dtype=np.float32)
    proposal_to_proposal_2 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 4, 1, 1]],
        dtype=np.float32)
    proposal_to_proposal_3 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 4, 1, 1]],
        dtype=np.float32)

    (max_path_sum, subject_proposal_index,
     object_proposal_index) = utils.compute_max_path_sum(
         n_proposal=np.array([n_proposal]),
         n_triple=np.array([n_triple]),
         subject_to_proposal=np.expand_dims(
             np.stack([
                 subject_to_proposal_1, subject_to_proposal_2,
                 subject_to_proposal_3
             ], 0), 0),
         proposal_to_proposal=np.expand_dims(
             np.stack([
                 proposal_to_proposal_1, proposal_to_proposal_2,
                 proposal_to_proposal_3
             ], 0), 0),
         proposal_to_object=np.expand_dims(
             np.stack([
                 proposal_to_object_1, proposal_to_object_2,
                 proposal_to_object_3
             ], 0), 0))
    self.assertAllClose(max_path_sum, [[8, 7, 7]])
    self.assertAllEqual(subject_proposal_index, [[2, 0, 3]])
    self.assertAllEqual(object_proposal_index, [[3, 1, 1]])

  def test_compute_max_path_sum_batch1_triple2_mask_triple(self):
    """ Deals with two triples, the third should be masked, n_triple(2) < max_n_triple(3).
           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=5, p2-p3=4
      p0   p1     p2    p3
      \1    |1    |1   /2
            `object`
      solution: 8, subject -> p2 -> p3 -> object

           `subject`
       /4   |3    |2   \1
      p0   p1     p2    p3
       | X  |  X  |  X  |   p3-p1=4, p2-p3=4
      p0   p1     p2    p3
      \1    |2    |1   /1
            `object`
      solution: 7, subject -> p0 -> p1 -> object
    """

    n_triple, n_proposal, max_n_triple, max_n_proposal = 2, 4, 3, 4
    subject_to_proposal_1 = np.array([4, 3, 2, 1], dtype=np.float32)
    subject_to_proposal_2 = np.array([4, 3, 2, 1], dtype=np.float32)
    subject_to_proposal_3 = np.array([3, 3, 2, 1], dtype=np.float32)
    proposal_to_object_1 = np.array([1, 1, 1, 2], dtype=np.float32)
    proposal_to_object_2 = np.array([1, 2, 1, 1], dtype=np.float32)
    proposal_to_object_3 = np.array([1, 2, 1, 1], dtype=np.float32)
    proposal_to_proposal_1 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 5, 1, 1]],
        dtype=np.float32)
    proposal_to_proposal_2 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 4, 1, 1]],
        dtype=np.float32)
    proposal_to_proposal_3 = np.array(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 4], [1, 4, 1, 1]],
        dtype=np.float32)

    (max_path_sum, subject_proposal_index,
     object_proposal_index) = utils.compute_max_path_sum(
         n_proposal=np.array([n_proposal]),
         n_triple=np.array([n_triple]),
         subject_to_proposal=np.expand_dims(
             np.stack([
                 subject_to_proposal_1, subject_to_proposal_2,
                 subject_to_proposal_3
             ], 0), 0),
         proposal_to_proposal=np.expand_dims(
             np.stack([
                 proposal_to_proposal_1, proposal_to_proposal_2,
                 proposal_to_proposal_3
             ], 0), 0),
         proposal_to_object=np.expand_dims(
             np.stack([
                 proposal_to_object_1, proposal_to_object_2,
                 proposal_to_object_3
             ], 0), 0))
    self.assertAllClose(max_path_sum, [[8, 7, 0]])
    self.assertAllEqual(subject_proposal_index, [[2, 0, -1]])
    self.assertAllEqual(object_proposal_index, [[3, 1, -1]])

  def test_gather_overlapped_box_indicator_by_iou_batch1(self):
    (highly_overlapped,
     roughly_overlapped) = utils.gather_overlapped_box_indicator_by_iou(
         n_proposal=[4],
         proposals=np.array([[[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                              [0, 0, 0.499, 1]]],
                            dtype=np.float32),
         n_reference=[1],
         reference=np.array([[[0, 0, 1, 1]]], dtype=np.float32),
         threshold=0.5)

    self.assertAllClose(highly_overlapped, [[[False, False, True, False]]])
    self.assertAllClose(roughly_overlapped, [[[False, True, False, True]]])

  def test_gather_overlapped_box_indicator_by_iou_batch2(self):
    proposals1 = [[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                  [0, 0, 0.499, 1]]
    reference1 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    highly_overlappet_gt1 = [[False, False, True, False],
                             [False, False, True, False]]
    roughly_overlapped_gt1 = [[False, True, False, True],
                              [False, True, False, True]]

    proposals2 = [[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                  [0, 0, 0.499, 1]]
    reference2 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    highly_overlappet_gt2 = [[False, False, True, False],
                             [False, False, True, False]]
    roughly_overlapped_gt2 = [[False, True, False, True],
                              [False, True, False, True]]

    (highly_overlapped,
     roughly_overlapped) = utils.gather_overlapped_box_indicator_by_iou(
         n_proposal=[4, 4],
         proposals=np.array([proposals1, proposals2], dtype=np.float32),
         n_reference=[2, 2],
         reference=np.array([reference1, reference2], dtype=np.float32),
         threshold=0.5)

    self.assertAllEqual(highly_overlapped,
                        [highly_overlappet_gt1, highly_overlappet_gt2])
    self.assertAllEqual(roughly_overlapped,
                        [roughly_overlapped_gt1, roughly_overlapped_gt2])

  def test_gather_overlapped_box_indicator_by_iou_batch2_mask_reference(self):
    proposals1 = [[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                  [0, 0, 0.499, 1]]
    reference1 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    highly_overlappet_gt1 = [[False, False, True, False],
                             [False, False, True, False]]
    roughly_overlapped_gt1 = [[False, True, False, True],
                              [False, True, False, True]]

    proposals2 = [[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                  [0, 0, 0.499, 1]]
    reference2 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    highly_overlappet_gt2 = [[False, False, True, False],
                             [False, False, False, False]]
    roughly_overlapped_gt2 = [[False, True, False, True],
                              [False, False, False, False]]

    (highly_overlapped,
     roughly_overlapped) = utils.gather_overlapped_box_indicator_by_iou(
         n_proposal=[4, 4],
         proposals=np.array([proposals1, proposals2], dtype=np.float32),
         n_reference=[2, 1],
         reference=np.array([reference1, reference2], dtype=np.float32),
         threshold=0.5)

    self.assertAllEqual(highly_overlapped,
                        [highly_overlappet_gt1, highly_overlappet_gt2])
    self.assertAllEqual(roughly_overlapped,
                        [roughly_overlapped_gt1, roughly_overlapped_gt2])

  def test_gather_overlapped_box_indicator_by_iou_batch2_mask_proposal(self):
    proposals1 = [[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                  [0, 0, 0.499, 1]]
    reference1 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    highly_overlappet_gt1 = [[False, False, True, False],
                             [False, False, True, False]]
    roughly_overlapped_gt1 = [[False, True, False, False],
                              [False, True, False, False]]

    proposals2 = [[-1, -1, 0, 0], [0, 0, 0.1, 0.1], [0, 0, 0.5, 1],
                  [0, 0, 0.499, 1]]
    reference2 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    highly_overlappet_gt2 = [[False, False, True, False],
                             [False, False, False, False]]
    roughly_overlapped_gt2 = [[False, True, False, True],
                              [False, False, False, False]]

    (highly_overlapped,
     roughly_overlapped) = utils.gather_overlapped_box_indicator_by_iou(
         n_proposal=[3, 4],
         proposals=np.array([proposals1, proposals2], dtype=np.float32),
         n_reference=[2, 1],
         reference=np.array([reference1, reference2], dtype=np.float32),
         threshold=0.5)

    self.assertAllEqual(highly_overlapped,
                        [highly_overlappet_gt1, highly_overlappet_gt2])
    self.assertAllEqual(roughly_overlapped,
                        [roughly_overlapped_gt1, roughly_overlapped_gt2])

  def test_sample_one_based_ids_not_equal(self):
    source = tf.random.uniform([20, 5], minval=1, maxval=6, dtype=tf.int32)
    self.assertEqual(tf.reduce_max(source).numpy(), 5)
    self.assertEqual(tf.reduce_min(source).numpy(), 1)

    for _ in range(5):
      sampled = utils.sample_one_based_ids_not_equal(source, max_id=5)
      self.assertFalse(tf.reduce_any(sampled == source))
      self.assertEqual(tf.reduce_max(sampled).numpy(), 5)
      self.assertEqual(tf.reduce_min(sampled).numpy(), 1)

  def test_sample_index_not_equal(self):
    size = tf.random_uniform([800], minval=2, maxval=11, dtype=tf.int32)
    index = tf.random.uniform([800, 5],
                              minval=0,
                              maxval=9999999999,
                              dtype=tf.int32)
    index = tf.mod(index, tf.expand_dims(size, 1))
    self.assertTrue(tf.reduce_all(index < tf.expand_dims(size, 1)))

    sampled_index = utils.sample_index_not_equal(index, size)
    self.assertTrue(tf.reduce_all(sampled_index < tf.expand_dims(size, 1)))
    self.assertFalse(tf.reduce_any(tf.equal(sampled_index, index)))


if __name__ == '__main__':
  tf.test.main()
