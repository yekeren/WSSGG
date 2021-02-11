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
from models import cap2sg_relation

tf.compat.v1.enable_eager_execution()


class Cap2SGRelationTest(tf.test.TestCase):

  def test_scatter_entity_labels(self):
    max_n_proposal, vocab_size = 3, 4
    subject_proposal_id = [[0, 2], [1, 2]]
    object_proposal_id = [[2, 1], [1, 0]]
    relation_id = [[1, 3], [3, 3]]

    output = cap2sg_relation._scatter_relation_labels(
        tf.convert_to_tensor(subject_proposal_id),
        tf.convert_to_tensor(object_proposal_id),
        tf.convert_to_tensor(relation_id), max_n_proposal, vocab_size)
    self.assertAllEqual(output, [
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
        ],
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
    ])

  def test_postprocess_relations(self):
    num_detections = tf.convert_to_tensor([3])
    detection_proposal = tf.convert_to_tensor([[0, 1, 2]])
    detection_scores = tf.convert_to_tensor([[0.8, 0.6, 0.4]])
    detection_classes = tf.convert_to_tensor([[0, 2, 1]])
    relation_scores = tf.convert_to_tensor([[[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                             [[0, 0, 0], [0, 0, 0], [0, 0,
                                                                     0]]]])

    (num_relations, log_prob, relation_score, relation_class, subject_proposal,
     subject_score, subject_class, object_proposal, object_score,
     object_class) = cap2sg_relation._postprocess_relations(
         num_detections,
         detection_proposal,
         detection_scores,
         detection_classes,
         relation_scores,
         relation_max_total_size=3,
         relation_max_size_per_class=1,
         relation_threshold=0.01)

    self.assertAllEqual(num_relations, [2])
    self.assertAllClose(relation_score, [[1, 1, 0]])
    self.assertAllEqual(relation_class, [[2, 1, 0]])
    self.assertAllEqual(subject_proposal, [[0, 0, 0]])
    self.assertAllClose(subject_score, [[0.8, 0.8, 0]])
    self.assertAllEqual(subject_class, [[0, 0, 0]])
    self.assertAllEqual(object_proposal, [[1, 2, 0]])
    self.assertAllClose(object_score, [[0.6, 0.4, 0]])
    self.assertAllEqual(object_class, [[2, 1, 0]])


if __name__ == '__main__':
  tf.test.main()
