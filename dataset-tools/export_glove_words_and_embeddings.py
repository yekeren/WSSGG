from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

flags.DEFINE_string('glove_file', 'zoo/glove.6B.300d.txt',
                    'Path to the pre-trained GloVe embedding file.')

flags.DEFINE_string('output_vocabulary_file', 'zoo/glove_word_tokens.txt',
                    'Vocabulary file to be exported.')

flags.DEFINE_string('output_vocabulary_word_embedding_file',
                    'zoo/glove_word_vectors.npy',
                    'Vocabulary word embedding file to be exported.')

FLAGS = flags.FLAGS


def _load_glove(filename):
  """Loads the pre-trained GloVe word embedding.

  Args:
    filename: path to the GloVe embedding file.

  Returns:
    words: A list of words.
    wordvecs: A [n_word, n_dim] np array.
  """
  with tf.gfile.GFile(filename, 'r') as fid:
    lines = fid.readlines()
  n_word = len(lines)
  n_dim = len(lines[0].strip('\n').split()) - 1

  words = [''] * n_word
  wordvecs = np.zeros((n_word, n_dim))

  for i, line in enumerate(lines):
    items = line.strip('\n').split()
    words[i], wordvecs[i] = items[0], [float(v) for v in items[1:]]
    if i % 10000 == 0:
      tf.logging.info('On load %s/%s', i, len(lines))
  return words, wordvecs


def main(_):
  words, wordvecs = _load_glove(FLAGS.glove_file)

  with tf.gfile.GFile(FLAGS.output_vocabulary_file, 'w') as f:
    f.write('\n'.join(words))

  with tf.gfile.GFile(FLAGS.output_vocabulary_word_embedding_file, 'wb') as fp:
    np.save(fp, wordvecs)

  tf.logging.info("Shape of word embeddings: %s", wordvecs.shape)


if __name__ == '__main__':
  app.run(main)
