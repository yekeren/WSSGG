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


class TokenToIdLayer(tf.keras.layers.Layer):
  """TokenToId layer."""

  def __init__(self, token2id, oov_id, **kwargs):
    """Initializes the tf.lookup.StaticHashTable.

    Args:
      token2id: A dict mapping from string tokens to integer ids.
      oov_id: Id of the `out-of-vocabulary` token.
    """
    super(TokenToIdLayer, self).__init__(**kwargs)

    keys, values = zip(*token2id.items())
    initializer = tf.lookup.KeyValueTensorInitializer(keys,
                                                      values,
                                                      key_dtype=tf.string,
                                                      value_dtype=tf.int32)
    self.table = tf.lookup.StaticHashTable(initializer, default_value=oov_id)

  def call(self, inputs):
    """Converts the inputs to token ids.

    Args:
      inputs: A tf.string tensor of arbitrary shape.

    Returns:
      A tf.int32 tensor which has the same shape as the inputs.
    """
    return self.table.lookup(inputs)
