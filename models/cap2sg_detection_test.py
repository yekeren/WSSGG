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
from google.protobuf import text_format
from protos import reader_pb2
from models import cap2sg_detection

tf.compat.v1.enable_eager_execution()


class Cap2SGDetectionTest(tf.test.TestCase):

  def test_compute_iou(self):
    n_box1, n_box2 = [2], [2]
    box1 = [[[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0, 0, 1, 1]]]
    box2 = [[[0.5, 0, 1, 1], [0, 0, 0.5, 0.5], [0, 0, 1, 1]]]
    iou = cap2sg_detection.compute_iou(n_box1, box1, n_box2, box2)
    self.assertAllClose(iou, [[[0.5, 0.25, 0], [0, 1, 0], [0, 0, 0]]])

    n_box1, n_box2 = [3], [3]
    box1 = [[[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0, 0, 1, 1]]]
    box2 = [[[0.5, 0, 1, 1], [0, 0, 0.5, 0.5], [0, 0, 1, 1]]]
    iou = cap2sg_detection.compute_iou(n_box1, box1, n_box2, box2)
    self.assertAllClose(iou, [[[0.5, 0.25, 1], [0, 1, 0.25], [0.5, 0.25, 1]]])

  def test_scatter_entity_labels(self):
    max_n_proposal, vocab_size = 3, 4
    index_proposal = [[0, 2], [1, 2]]
    index_entity = [[1, 3], [3, 3]]

    output = cap2sg_detection.scatter_entity_labels(
        tf.convert_to_tensor(index_proposal), index_entity, max_n_proposal,
        vocab_size)
    self.assertAllEqual(output, [
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
    ])

  def test_post_process_detection_labels(self):
    self.assertAllClose(
        cap2sg_detection.post_process_detection_labels(
            tf.convert_to_tensor([[[0.0, 2, 2]]])), [[[0, 0.5, 0.5]]])

    self.assertAllClose(
        cap2sg_detection.post_process_detection_labels(
            tf.convert_to_tensor([[[0.0, 0, 0]]])), [[[1, 0, 0]]])


if __name__ == '__main__':
  tf.test.main()
