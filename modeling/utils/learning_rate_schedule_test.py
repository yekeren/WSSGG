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

from modeling.utils import learning_rate_schedule
from protos import learning_rate_schedule_pb2


class LearningRateScheduleTest(tf.test.TestCase):

  def test_learning_rate_schedule(self):
    options_str = r"""
      piecewise_constant_decay{
        values: 0.001
      }
    """
    options = text_format.Merge(
        options_str, learning_rate_schedule_pb2.LearningRateSchedule())
    schedule = learning_rate_schedule.create_learning_rate_schedule(options)
    self.assertIsInstance(schedule,
                          tf.keras.optimizers.schedules.PiecewiseConstantDecay)

    options_str = r"""
      exponential_decay {
        initial_learning_rate: 0.001
        decay_steps: 1000
        decay_rate: 1.0
      }
    """
    options = text_format.Merge(
        options_str, learning_rate_schedule_pb2.LearningRateSchedule())
    schedule = learning_rate_schedule.create_learning_rate_schedule(options)
    self.assertIsInstance(schedule,
                          tf.keras.optimizers.schedules.ExponentialDecay)

    options_str = r"""
      polynomial_decay {
        initial_learning_rate: 0.001
        decay_steps: 1000
        end_learning_rate: 0.0001
        power: 1.0
        cycle: true
      }
    """
    options = text_format.Merge(
        options_str, learning_rate_schedule_pb2.LearningRateSchedule())
    schedule = learning_rate_schedule.create_learning_rate_schedule(options)
    self.assertIsInstance(schedule,
                          tf.keras.optimizers.schedules.PolynomialDecay)


if __name__ == '__main__':
  tf.test.main()
