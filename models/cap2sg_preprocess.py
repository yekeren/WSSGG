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
from protos import model_pb2

import numpy as np
import tensorflow as tf
from modeling.layers import id_to_token
from modeling.layers import token_to_id

from models.cap2sg_data import DataTuple


def initialize(options, dt):
  if not isinstance(options, model_pb2.Cap2SGPreprocess):
    raise ValueError('Options has to be a Cap2SGPreprocess proto.')

  if not isinstance(dt, DataTuple):
    raise ValueError('Invalid DataTuple object.')

  # Load GloVe embeddings.
  glove_dict = _load_glove_data(options.glove_vocabulary_file,
                                options.glove_embedding_file)

  # Initialize token2id and id2token functions.
  token2id, id2token = _read_vocabulary(options.vocabulary_file, glove_dict,
                                        options.minimum_frequency)
  dt.vocab_size = len(token2id)
  dt.token2id_func = token_to_id.TokenToIdLayer(token2id, oov_id=0)
  dt.id2token_func = id_to_token.IdToTokenLayer(id2token, oov='OOV')

  # Create word embeddings.
  dt.dims = options.embedding_dims
  if options.embedding_trainable:
    dt.embeddings = tf.get_variable('embeddings',
                                    initializer=_initialize_from_glove(
                                        glove_dict, token2id, dt.dims),
                                    trainable=options.embedding_trainable)
  else:
    dt.embeddings = tf.constant(_initialize_from_glove(glove_dict, token2id,
                                                       dt.dims),
                                name='embeddings')
  dt.embedding_func = lambda x: tf.nn.embedding_lookup(dt.embeddings, x)

  # Create class biases.
  (dt.bias_entity, dt.bias_attribute,
   dt.bias_relation) = _initialize_biases(dt.embeddings, options.bias_mode)
  return dt


def _load_glove_data(glove_vocabulary_file, glove_embedding_file):
  """Loads GloVe embedding vectors.

  Args:
    glove_vocabulary_file: GloVe vocabulary file.
    glove_embedding_file: GloVe word embedding file.

  Returns:
    glove_dict: GloVe embeddings, a dict keyed by tokens, values are np.array.
  """
  glove_vectors = np.load(glove_embedding_file).astype(np.float32)
  with tf.gfile.GFile(glove_vocabulary_file, 'r') as f:
    glove_tokens = [x.strip('\n') for x in f]
  return dict((k, v) for k, v in zip(glove_tokens, glove_vectors))


def _read_vocabulary(vocabulary_file, glove_dict, min_freq):
  """Reads vocabulary file.

  Args:
    vocabulary_file: Each line in the file is "word\tfreq\n"
    glove_dict: GloVe embeddings, a dict keyed by tokens, values are np.array.
    min_freq: Minimum frequency for the words to be considered.

  Returns:
    token2id: A dict of token-to-id mapping.
    id2token: A dict of id-to-token mapping.
  """
  token2id, id2token = {'OOV': 0}, {0: 'OOV'}
  id_offset = 1  # ZERO is reserved for OOV.
  with open(vocabulary_file, 'r') as f:
    for line in f:
      elems = line.strip('\n').split('\t')
      if len(elems) == 1:
        token, freq = elems[0].lower(), 1000
      elif len(elems) == 2:
        token, freq = elems

      if int(freq) < min_freq:
        break
      if any(word in glove_dict for word in token.split(' ')):
        id2token[id_offset] = token
        token2id[token] = id_offset
        id_offset += 1
  return token2id, id2token


def _initialize_from_glove(glove_dict, token2id, embedding_dims):
  """Initializes token embeddings from GloVe.

  Args:
    glove_dict: GloVe embeddings, a dict keyed by tokens, values are np.array.
    token2id: A dict mapping from multi-words tokens to id.
    embedding_dims: embedding dimensions.

  Returns:
    embeddings: A [vocab_size, dims] np.float32 array.
  """

  embeddings = np.zeros((len(token2id), embedding_dims), dtype=np.float32)
  for multi_word_token, token_id in token2id.items():
    total = 0
    for word in multi_word_token.split(' '):
      if word in glove_dict:
        embeddings[token_id] += glove_dict[word]
        total += 1
    assert total > 0 or multi_word_token == 'OOV'
    embeddings[token_id] /= max(1e-6, total)
  return embeddings


def _initialize_biases(embeddings, bias_mode):
  """Initializes biases.

  Args:
    embeddings: A [vocab_size, dims] float tensor.
    bias_mode: A BiasMode proto.

  Returns:
    bias_entity: Entity bias, a [vocab_size] float tensor.
    bias_attribute: Attribute bias, a [vocab_size] float tensor.
    bias_relation: Relation bias, a [vocab_size] float tensor.
  """
  vocab_size = embeddings.shape[0].value

  # Not using biases.
  if model_pb2.BIAS_MODE_ZERO == bias_mode:
    zeros = tf.zeros(shape=[vocab_size], dtype=tf.float32)
    return zeros, zeros, zeros

  # Use per-class-entry biases.
  elif model_pb2.BIAS_MODE_TRADITION == bias_mode:
    biases = [
        tf.get_variable(name,
                        initializer=tf.zeros_initializer(),
                        shape=[vocab_size],
                        trainable=True)
        for name in ['bias_entity', 'bias_attribute', 'bias_relation']
    ]
    return tuple(biases)

  # Use biases learned from word embeddings.
  elif model_pb2.BIAS_MODE_TRAIN_FROM_EMBEDDING == bias_mode:
    biases = [
        tf.squeeze(
            tf.layers.Dense(1,
                            kernel_initializer=tf.keras.initializers.RandomNormal(
                                mean=0.0, stddev=0.01),
                            name=name)(embeddings), -1)
        for name in ['entity_bias', 'attribute_bias', 'relation_bias']
    ]
    return tuple(biases)

  raise ValueError('Invalid bias mode %i' % bias_mode)
