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

from protos import optimizer_pb2


def create_optimizer(options, learning_rate=0.1):
  """Builds optimizer from options.

  Args:
    options: An instance of optimizer_pb2.Optimizer.
    learning_rate: A scalar tensor denoting the learning rate.

  Returns:
    A tensorflow optimizer instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, optimizer_pb2.Optimizer):
    raise ValueError('The options has to be an instance of Optimizer.')

  optimizer = options.WhichOneof('optimizer')
  options = getattr(options, optimizer)

  if 'adagrad' == optimizer:
    return tf.compat.v1.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=options.initial_accumulator_value)

  if 'rmsprop' == optimizer:
    return tf.compat.v1.train.RMSPropOptimizer(learning_rate,
                                               decay=options.decay,
                                               momentum=options.momentum,
                                               epsilon=options.epsilon,
                                               centered=options.centered)

  if 'adam' == optimizer:
    return tf.compat.v1.train.AdamOptimizer(learning_rate,
                                            beta1=options.beta1,
                                            beta2=options.beta2,
                                            epsilon=options.epsilon)

  raise ValueError('Invalid optimizer: {}.'.format(optimizer))
