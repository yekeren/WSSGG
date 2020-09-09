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
from modeling.layers import id_to_token

tf.compat.v1.enable_eager_execution()


class IdToTokenLayerTest(tf.test.TestCase):

  def test_id_to_token(self):
    test_layer = id_to_token.IdToTokenLayer({5: 'hello', 11: 'world'}, 'OOV')

    output = test_layer(tf.convert_to_tensor([5, 1, 11, 2]))
    self.assertAllEqual(output, [b'hello', b'OOV', b'world', b'OOV'])

    output = test_layer(tf.convert_to_tensor([5, 1, 11]))
    self.assertAllEqual(output, [b'hello', b'OOV', b'world'])

  def test_id_to_token_2d(self):
    test_layer = id_to_token.IdToTokenLayer({
        2: 'one',
        3: 'world',
        6: 'dream'
    }, 'UNK')

    output = test_layer(tf.convert_to_tensor([[4, 4, 3, 4], [2, 3, 2, 6]]))
    self.assertAllEqual(output, [[b'UNK', b'UNK', b'world', b'UNK'],
                                 [b'one', b'world', b'one', b'dream']])


if __name__ == '__main__':
  tf.test.main()
