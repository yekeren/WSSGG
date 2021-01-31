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
from models import cap2sg_grounding

tf.compat.v1.enable_eager_execution()


class Cap2SGGroundingTest(tf.test.TestCase):

  def test_parse_entity_and_attributes(self):
    (entity, n_attribute,
     attributes) = cap2sg_grounding.parse_entity_and_attributes(
         tf.convert_to_tensor([['suitcase:small,packed', '', ''],
                               [
                                   'suitcase:small,packed', 'book shelf',
                                   'man:handsome,tall,smiling'
                               ]]))

    self.assertAllEqual(
        entity, [[b'suitcase', b'', b''], [b'suitcase', b'book shelf', b'man']])
    self.assertAllEqual(n_attribute, [[2, 0, 0], [2, 0, 3]])
    self.assertAllEqual(
        attributes,
        [[[b'small', b'packed', b''], [b'', b'', b''], [b'', b'', b'']],
         [[b'small', b'packed', b''], [b'', b'', b''],
          [b'handsome', b'tall', b'smiling']]])


if __name__ == '__main__':
  tf.test.main()
