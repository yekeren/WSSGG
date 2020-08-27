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
from google.protobuf import text_format

from modeling.utils import optimization
from protos import optimizer_pb2


class OptimizationTest(tf.test.TestCase):

  def test_create_optimizer(self):
    options_str = "adagrad{}"
    options = text_format.Merge(options_str, optimizer_pb2.Optimizer())
    opt = optimization.create_optimizer(options)
    self.assertIsInstance(opt, tf.compat.v1.train.AdagradOptimizer)

    options_str = "rmsprop{}"
    options = text_format.Merge(options_str, optimizer_pb2.Optimizer())
    opt = optimization.create_optimizer(options)
    self.assertIsInstance(opt, tf.compat.v1.train.RMSPropOptimizer)

    options_str = "adam{}"
    options = text_format.Merge(options_str, optimizer_pb2.Optimizer())
    opt = optimization.create_optimizer(options)
    self.assertIsInstance(opt, tf.compat.v1.train.AdamOptimizer)


if __name__ == '__main__':
  tf.test.main()
