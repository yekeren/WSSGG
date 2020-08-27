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

import tensorflow as tf

from modeling.utils import masked_ops as ops

tf.compat.v1.enable_eager_execution()


class MaskedOpsTest(tf.test.TestCase):

  def test_masked_maximum(self):
    self.assertAllClose(
        ops.masked_maximum(data=[[-2.0, 1.0, 2.0, -1.0, 0.0],
                                 [-2.0, -1.0, -3.0, -5.0, -4.0]],
                           mask=tf.convert_to_tensor(
                               [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                               dtype=tf.float32)), [[2.0], [-1.0]])

    self.assertAllClose(
        ops.masked_maximum(data=[[-2.0, 1.0, 2.0, -1.0, 0.0],
                                 [-2.0, -1.0, -3.0, -5.0, -4.0]],
                           mask=tf.convert_to_tensor(
                               [[1, 1, 0, 1, 1], [0, 0, 1, 1, 1]],
                               dtype=tf.float32)), [[1.0], [-3.0]])

    self.assertAllClose(
        ops.masked_maximum(data=[[-2.0, 1.0, 2.0, -1.0, 0.0],
                                 [-2.0, -1.0, -3.0, -5.0, -4.0]],
                           mask=tf.convert_to_tensor(
                               [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=tf.float32)), [[-2.0], [-5.0]])

  def test_masked_minimum(self):
    self.assertAllClose(
        ops.masked_minimum(data=[[-2.0, 1.0, 2.0, -1.0, 0.0],
                                 [-2.0, -1.0, -3.0, -5.0, -4.0]],
                           mask=tf.convert_to_tensor(
                               [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                               dtype=tf.float32)), [[-2.0], [-5.0]])

    self.assertAllClose(
        ops.masked_minimum(data=[[-2.0, 1.0, 2.0, -1.0, 0.0],
                                 [-2.0, -1.0, -3.0, -5.0, -4.0]],
                           mask=tf.convert_to_tensor(
                               [[0, 1, 1, 0, 1], [1, 1, 1, 0, 1]],
                               dtype=tf.float32)), [[0.0], [-4.0]])

    self.assertAllClose(
        ops.masked_minimum(data=[[-2.0, 1.0, 2.0, -1.0, 0.0],
                                 [-2.0, -1.0, -3.0, -5.0, -4.0]],
                           mask=tf.convert_to_tensor(
                               [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=tf.float32)), [[2.0], [-1.0]])

  def test_masked_sum(self):
    self.assertAllClose(
        ops.masked_sum(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                       mask=tf.convert_to_tensor([[1, 0, 1], [0, 1, 0]],
                                                 dtype=tf.float32)), [[4], [5]])

    self.assertAllClose(
        ops.masked_sum(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                       mask=tf.convert_to_tensor([[0, 1, 0], [1, 0, 1]],
                                                 dtype=tf.float32)),
        [[2], [10]])

  def test_masked_avg(self):
    self.assertAllClose(
        ops.masked_avg(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                       mask=tf.convert_to_tensor([[1, 0, 1], [0, 1, 0]],
                                                 dtype=tf.float32)), [[2], [5]])

    self.assertAllClose(
        ops.masked_avg(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                       mask=tf.convert_to_tensor([[0, 1, 0], [1, 0, 1]],
                                                 dtype=tf.float32)), [[2], [5]])

    self.assertAllClose(
        ops.masked_avg(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                       mask=tf.convert_to_tensor([[0, 0, 0], [0, 0, 0]],
                                                 dtype=tf.float32)), [[0], [0]])

  def test_masked_sum_nd(self):
    self.assertAllClose(
        ops.masked_sum_nd(data=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11, 12.0]]],
                          mask=tf.convert_to_tensor([[1, 0, 1], [0, 1, 0]],
                                                    dtype=tf.float32)),
        [[[6, 8]], [[9, 10]]])

  def test_masked_avg_nd(self):
    self.assertAllClose(
        ops.masked_avg_nd(data=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                          mask=tf.convert_to_tensor([[1, 0, 1], [0, 1, 0]],
                                                    dtype=tf.float32)),
        [[[3, 4]], [[9, 10]]])

    self.assertAllClose(
        ops.masked_avg_nd(data=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                          mask=tf.convert_to_tensor([[0, 0, 0], [0, 0, 0]],
                                                    dtype=tf.float32)),
        [[[0, 0]], [[0, 0]]])

  def test_masked_max_nd(self):
    self.assertAllClose(
        ops.masked_max_nd(data=[[[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                          mask=tf.convert_to_tensor([[1, 0, 1], [0, 1, 0]],
                                                    dtype=tf.float32)),
        [[[5, 6]], [[9, 10]]])

    self.assertAllClose(
        ops.masked_max_nd(data=[[[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]],
                                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
                          mask=tf.convert_to_tensor([[1, 1, 0], [1, 0, 1]],
                                                    dtype=tf.float32)),
        [[[3, 6]], [[11, 12]]])

  def test_masked_softmax(self):
    self.assertAllClose(
        ops.masked_softmax(data=[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                           mask=tf.convert_to_tensor(
                               [[1, 1, 1, 1], [1, 1, 1, 1]], dtype=tf.float32)),
        [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])

    self.assertAllClose(
        ops.masked_softmax(data=[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                           mask=tf.convert_to_tensor(
                               [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=tf.float32)),
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])


if __name__ == '__main__':
  tf.test.main()
