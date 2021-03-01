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

import subprocess
import tensorflow as tf
import threading
import sng_parser

flags.DEFINE_string('caption_annotations_file', '',
                    '[input] Caption annotations JSON file.')
flags.DEFINE_string('scenegraph_annotations_file', '',
                    '[output] Scenegraph annotations JSON file.')

FLAGS = flags.FLAGS


def _create_scenegraphs_from_captions(caption_annotations_file,
                                      scenegraph_annotations_file):
  """Creates scene graphs from captions annotations.

  Args:
    caption_annotations_file: JSON file containing caption annotations.
    scenegraph_annotations_file: JSON file containing caption annotations.
  """
  with tf.io.gfile.GFile(caption_annotations_file, 'r') as fid:
    annots = json.load(fid)

  images, annots = annots['images'], annots['annotations']
  id_to_captions = {}
  for annot in annots:
    image_id = annot['image_id']
    id_to_captions.setdefault(image_id, []).append(annot['caption'])
  assert len(images) == len(id_to_captions)

  for i, image in enumerate(images):
    if i % 1000 == 0:
      logging.info('On image %i/%i', 1 + i, len(images))
    captions = id_to_captions[image['id']]
    scene_graphs = []
    for caption in captions:
      caption = caption.strip().replace('\n', '').lower()
      if caption[-1] != '.':
        caption += '.'
      scene_graphs.append(sng_parser.parse(caption))
    image['scene_graphs'] = scene_graphs
    image['captions'] = captions

  with tf.io.gfile.GFile(scenegraph_annotations_file, 'w') as fid:
    json.dump(images, fid)


def main(_):
  assert FLAGS.caption_annotations_file, '`caption_annotations_file` missing.'
  assert FLAGS.scenegraph_annotations_file, '`scenegraph_annotations_file` missing.'

  logging.set_verbosity(logging.INFO)

  _create_scenegraphs_from_captions(FLAGS.caption_annotations_file,
                                    FLAGS.scenegraph_annotations_file)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
