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

from absl import logging

import numpy as np
import tensorflow as tf

from protos import model_pb2
from models.cap2sg_data import DataTuple

from modeling.modules import graph_networks


def enrich_features(options, dt):
  """Enrich text features.

  Args:
    options: A Cap2SGLinguistic proto.
    dt: A DataTuple object.
  """
  if not isinstance(options, model_pb2.Cap2SGLinguistic):
    raise ValueError('Options has to be a Cap2SGLinguistic proto.')

  if not isinstance(dt, DataTuple):
    raise ValueError('Invalid DataTuple object.')

  gn = graph_networks.build_graph_network(options.graph_network,
                                          is_training=True)
  entity_embs, relation_embs = gn.compute_graph_embeddings(
      batch_n_node=dt.n_entity,
      batch_n_edge=dt.n_relation,
      batch_nodes=dt.entity_embs,
      batch_edges=dt.relation_embs,
      batch_senders=dt.relation_senders,
      batch_receivers=dt.relation_receivers)

  dt.refined_entity_embs = entity_embs
  dt.refined_relation_embs = relation_embs
  return dt
