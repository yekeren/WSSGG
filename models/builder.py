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

from protos import model_pb2
from models.ws_scene_graph import WSSceneGraph
from models.ws_scene_graph_gnet import WSSceneGraphGNet
from models.ws_scene_graph_caption_gnet import WSSceneGraphCaptionGNet
from models.ws_scene_graph_rnn_refine import WSSceneGraphRnnRefine

MODELS = {
    model_pb2.WSSceneGraph.ext: WSSceneGraph,
    model_pb2.WSSceneGraphGNet.ext: WSSceneGraphGNet,
    model_pb2.WSSceneGraphRnnRefine.ext: WSSceneGraphRnnRefine,
    model_pb2.WSSceneGraphCaptionGNet.ext: WSSceneGraphCaptionGNet,
}


def build(options, is_training):
  """Builds a model based on the options.

  Args:
    options: A model_pb2.Model instance.

  Returns:
    A model instance.

  Raises:
    ValueError: If the model proto is invalid or cannot find a registered entry.
  """
  if not isinstance(options, model_pb2.Model):
    raise ValueError('The options has to be an instance of model_pb2.Model.')

  for extension, model_proto in options.ListFields():
    if extension in MODELS:
      return MODELS[extension](model_proto, is_training)

  raise ValueError('Invalid model config!')
