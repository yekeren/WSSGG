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
from modeling.layers import token_to_id

tf.compat.v1.enable_eager_execution()


class TokenToIdLayerTest(tf.test.TestCase):

  def test_token_to_id(self):
    test_layer = token_to_id.TokenToIdLayer({'hello': 5, 'world': 11}, 97)

    output = test_layer(tf.convert_to_tensor(['hello', ',', 'world', '!']))
    self.assertAllEqual(output, [5, 97, 11, 97])

    output = test_layer(tf.convert_to_tensor(['hell', ',', 'world', '!!']))
    self.assertAllEqual(output, [97, 97, 11, 97])

  def test_token_to_id_2d(self):
    test_layer = token_to_id.TokenToIdLayer({
        'one': 2,
        'world': 3,
        'dream': 5
    }, 4)

    output = test_layer(
        tf.convert_to_tensor([['hello', ',', 'world', '!'],
                              ['hell', ',', 'world', '!!']]))
    self.assertAllEqual(output, [[4, 4, 3, 4], [4, 4, 3, 4]])

    output = test_layer(
        tf.convert_to_tensor([['one', 'world', 'one', 'dream'],
                              ['one', 'word', 'one', 'dream']]))
    self.assertAllEqual(output, [[2, 3, 2, 5], [2, 4, 2, 5]])


if __name__ == '__main__':
  tf.test.main()
