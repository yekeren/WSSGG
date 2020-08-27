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

from protos import learning_rate_schedule_pb2


def create_learning_rate_schedule(options):
  """Builds learning_rate_schedule from options.

  Args:
    options: An instance of
      learning_rate_schedule_pb2.LearningRateSchedule.

  Returns:
    A tensorflow LearningRateSchedule instance.

  Raises:
    ValueError: if options is invalid.
  """
  if not isinstance(options, learning_rate_schedule_pb2.LearningRateSchedule):
    raise ValueError(
        'The options has to be an instance of LearningRateSchedule.')

  oneof = options.WhichOneof('learning_rate_schedule')

  if 'piecewise_constant_decay' == oneof:
    options = options.piecewise_constant_decay
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=options.boundaries[:], values=options.values[:])

  if 'exponential_decay' == oneof:
    options = options.exponential_decay
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=options.initial_learning_rate,
        decay_steps=options.decay_steps,
        decay_rate=options.decay_rate,
        staircase=options.staircase)

  if 'polynomial_decay' == oneof:
    options = options.polynomial_decay
    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=options.initial_learning_rate,
        decay_steps=options.decay_steps,
        end_learning_rate=options.end_learning_rate,
        power=options.power,
        cycle=options.cycle)

  raise ValueError('Invalid learning_rate_schedule: {}.'.format(oneof))
