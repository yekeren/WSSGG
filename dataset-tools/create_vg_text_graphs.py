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
import cv2
import json
import zipfile

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import sng_parser

flags.DEFINE_string('caption_annotations_file', '',
                    'Path to the caption annotations.')
flags.DEFINE_string('scenegraph_annotations_file', '',
                    'Path to the scenegraphs annotations.')
flags.DEFINE_integer('number_of_threads', 5, 'Number of threads.')
flags.DEFINE_integer('number_of_processes', 30, 'Number of threads.')

FLAGS = flags.FLAGS


def _extract_scene_graphs(image_id, captions):
  """Starts thread worker.  """
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


def _create_scene_graphs(input_file, output_file):
  """Extracts scene graphs.

  Args:
    zip_file: ZIP file containing the image files.
  """
  with tf.io.gfile.GFile(input_file, 'r') as f:
    annots = json.load(f)

  res_list = []
  with Pool(processes=FLAGS.number_of_processes) as pool:
    for i, annot in enumerate(annots):
      image_id = annot['id']
      captions = [x['phrase'] for x in annot['regions']]
      res_list.append(
          pool.apply_async(_extract_scene_graphs, (image_id, captions)))

    pool.close()
    pool.join()

  data = []
  for res in res_list:
    data.append(res.get())

  with tf.io.gfile.GFile(output_file, 'w') as fid:
    json.dump(data, fid)
  logging.info('Done')


def main(_):
  logging.set_verbosity(logging.INFO)

  _create_scene_graphs(FLAGS.caption_annotations_file,
                       FLAGS.scenegraph_annotations_file)


if __name__ == '__main__':
  app.run(main)
