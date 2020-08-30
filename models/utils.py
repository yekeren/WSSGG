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


def gather_grounded_proposal_box(proposals, proposal_index):
  """Gathers grounded proposal box.

  Args:
    proposals: A [batch, max_n_proposal, 4] float tensor.
    proposal_index: A [batch, max_n_triple] int tensor.

  Returns:
    A [batch, max_n_triple, 4] float tensor, the gathered proposals.
  """
  batch = proposals.shape[0].value
  max_n_triple = tf.shape(proposal_index)[1]
  max_n_proposal = tf.shape(proposals)[1]

  proposals = tf.broadcast_to(tf.expand_dims(proposals, 1),
                              [batch, max_n_triple, max_n_proposal, 4])

  batch_index = tf.broadcast_to(tf.expand_dims(tf.range(batch), 1),
                                [batch, max_n_triple])
  triple_index = tf.broadcast_to(tf.expand_dims(tf.range(max_n_triple), 0),
                                 [batch, max_n_triple])
  index = tf.stack([batch_index, triple_index, proposal_index], -1)
  return tf.gather_nd(proposals, index)
