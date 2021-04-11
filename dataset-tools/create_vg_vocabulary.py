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

import os
import json

from absl import app
from absl import flags
from absl import logging

import collections

import zipfile
import numpy as np
import tensorflow as tf
import pickle
from graph_nets import utils_np

flags.DEFINE_string('scenegraph_annotations_file', '',
                    'Scene graph annotations JSON file.')
flags.DEFINE_string('output_file', '',
                    'Path to store the output vocabulary file.')

FLAGS = flags.FLAGS


def _create_vocabulary_from_annotations(scenegraph_annotations_file,
                                        vocabulary_file):
  """Creates tf record files from scenegraphs annotations.

  Args:
    scenegraph_annotations_file: JSON file containing scene graph annotations.
    proposal_npz_directory: Path to the directory saving proposal data.
    tf_record_file: Tf record file containing tf.example protos.
    num_output_parts: Number of output partitions.
  """
  with tf.io.gfile.GFile(scenegraph_annotations_file, 'r') as fid:
    annots = json.load(fid)
  logging.info('Original images: %i', len(annots))

  counter = collections.Counter()
  for i, annot in enumerate(annots):
    for sg in annot['scene_graphs']:
      entities, relations = sg['entities'], sg['relations']
      for e in entities:
        counter[e['head']] += 1  # entity +1
        for att in e['modifiers']:
          if att['dep'] not in ['det', 'nummod']:
            counter[att['span']] += 1  # attribute +1
    for r in relations:
      counter[r['relation']] += 1  # relation +1

  with open(vocabulary_file, 'w') as f:
    for token, freq in counter.most_common():
      try:
        token.encode('ascii')
      except:
        continue
      f.write('%s\t%i\n' % (token, freq))
  logging.info('Done')


def main(_):
  assert FLAGS.scenegraph_annotations_file, '`scenegraph_annotations_file` missing.'
  assert FLAGS.output_file, '`output_directory` missing.'

  logging.set_verbosity(logging.INFO)

  _create_vocabulary_from_annotations(FLAGS.scenegraph_annotations_file,
                                      FLAGS.output_file)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
