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
from multiprocessing import Pool
import sng_parser

flags.DEFINE_string('caption_annotations_file', '',
                    '[input] Caption annotations JSON file.')
flags.DEFINE_string('scenegraph_annotations_file', '',
                    '[output] Scenegraph annotations JSON file.')
flags.DEFINE_integer('number_of_processes', 30, 'Number of threads.')

FLAGS = flags.FLAGS


def _extract_scene_graphs(image_id, captions):
  """Starts worker.  """
  logging.info('Processing %s.', image_id)

  paired_captions = []
  scene_graphs = []
  for caption in captions:
    caption = caption.strip().replace('\n', '').lower()
    if not caption: continue
    if caption[-1] != '.':
      caption += '.'
    paired_captions.append(caption)
    scene_graphs.append(sng_parser.parse(caption))

  return {
      'image_id': image_id,
      'captions': paired_captions,
      'scene_graphs': scene_graphs,
  }


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

  res_list = []
  with Pool(processes=FLAGS.number_of_processes) as pool:
    for i, image in enumerate(images):
      image_id = image['id']
      captions = id_to_captions[image_id]
      scene_graphs = []

      res_list.append(pool.apply_async(_extract_scene_graphs, (image_id, captions)))
    pool.close()
    pool.join()

  for image, res in zip(images, res_list):
    res = res.get()
    assert image['id'] == res['image_id']
    image['captions'] = res['captions']
    image['scene_graphs'] = res['scene_graphs']

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
