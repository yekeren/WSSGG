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

from modeling.utils import box_ops

tf.compat.v1.enable_eager_execution()


class BoxOpsTest(tf.test.TestCase):

  def test_flip_left_right(self):
    self.assertAllClose(
        box_ops.flip_left_right([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                                 [0.0, 0.0, 0.5, 0.5]]),
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0]])

  def test_center(self):
    self.assertAllClose(
        box_ops.center([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                        [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]),
        [[0.5, 0.5], [0.25, 0.5], [0.25, 0.25], [0.0, 0.0]])

  def test_size(self):
    self.assertAllClose(
        box_ops.size([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                      [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]),
        [[1.0, 1.0], [0.5, 1.0], [0.5, 0.5], [0.0, 0.0]])

  def test_area(self):
    self.assertAllClose(
        box_ops.area([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0],
                      [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, -1.0, -1.0],
                      [0.0, 0.0, 0.0, 0.0]]), [1.0, 0.5, 0.25, 0.0, 0.0])

  def test_intersect(self):
    self.assertAllClose(
        box_ops.intersect(box1=[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 3.0],
                                [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0]],
                          box2=[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                                [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0],
                                [2.0, 2.0, 1.0, 1.0]]),
        [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
         [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]])

  def test_iou(self):
    self.assertAllClose(
        box_ops.iou(box1=[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 3.0],
                          [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0]],
                    box2=[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                          [0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0],
                          [2.0, 2.0, 1.0, 1.0]]),
        [0.25, 1.0 / 6, 1.0 / 6, 0.0, 0.0])

  def test_x_intersect_len(self):
    self.assertAllClose(
        box_ops.x_intersect_len(box1=[[0.0, 0.0, 2.0,
                                       2.0], [0.0, 0.0, 2.0, 3.0],
                                      [1.0, 1.0, 2.0,
                                       2.0], [0.0, 0.0, 1.0, 1.0],
                                      [0.0, 0.0, 1.0, 1.0]],
                                box2=[[1.0, 1.0, 2.0,
                                       2.0], [1.0, 1.0, 2.0, 2.0],
                                      [0.0, 0.0, 2.0,
                                       3.0], [1.0, 1.0, 1.0, 1.0],
                                      [2.0, 2.0, 1.0, 1.0]]),
        [1.0, 1.0, 1.0, 0.0, 0.0])

  def test_y_intersect_len(self):
    self.assertAllClose(
        box_ops.y_intersect_len(box1=[[0.0, 0.0, 2.0,
                                       2.0], [0.0, 0.0, 2.0, 3.0],
                                      [1.0, 1.0, 2.0,
                                       2.0], [0.0, 0.0, 1.0, 1.0],
                                      [0.0, 0.0, 1.0, 1.0]],
                                box2=[[1.5, 1.0, 2.0,
                                       2.0], [1.0, 1.0, 2.0, 2.0],
                                      [0.0, 0.0, 2.0,
                                       3.0], [1.0, 1.0, 1.0, 1.0],
                                      [2.0, 2.0, 1.0, 1.0]]),
        [0.5, 1.0, 1.0, 0.0, 0.0])

  def test_x_distance(self):
    self.assertAllClose(
        box_ops.x_distance(box1=[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 3.0],
                                 [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0],
                                 [0.0, 0.0, 1.0, 1.0]],
                           box2=[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                                 [0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0],
                                 [2.0, 2.0, 1.0, 1.0]]),
        [-0.5, 0.0, 0.0, -0.5, -1.0])

  def test_y_distance(self):
    self.assertAllClose(
        box_ops.y_distance(box1=[[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 3.0],
                                 [1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0],
                                 [0.0, 0.0, 1.0, 1.0]],
                           box2=[[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                                 [0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0],
                                 [2.0, 2.0, 1.0, 1.0]]),
        [-0.5, -0.5, 0.5, -0.5, -1.0])


if __name__ == '__main__':
  tf.test.main()
