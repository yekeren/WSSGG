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
