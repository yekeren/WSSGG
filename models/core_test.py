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
from models import core

tf.compat.v1.enable_eager_execution()


class CoreTest(tf.test.TestCase):

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
     object_proposal_index) = core.compute_max_path_sum(
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
     object_proposal_index) = core.compute_max_path_sum(
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
     object_proposal_index) = core.compute_max_path_sum(
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
     object_proposal_index) = core.compute_max_path_sum(
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
     object_proposal_index) = core.compute_max_path_sum(
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
     object_proposal_index) = core.compute_max_path_sum(
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


if __name__ == '__main__':
  tf.test.main()