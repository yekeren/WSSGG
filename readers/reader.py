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

from protos import reader_pb2
from readers import caption_graph_reader

_READERS = {
    'caption_graph_reader': caption_graph_reader,
}


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.Reader):
    raise ValueError('options has to be an instance of Reader.')

  reader_oneof = options.WhichOneof('reader_oneof')
  if not reader_oneof in _READERS:
    raise ValueError('Invalid reader %s!' % reader_oneof)

  return _READERS[reader_oneof].get_input_fn(getattr(options, reader_oneof),
                                             is_training=is_training)
