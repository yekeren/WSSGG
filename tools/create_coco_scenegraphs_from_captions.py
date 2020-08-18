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

flags.DEFINE_string('train_caption_annotations_file', '',
                    'Training annotations JSON file.')
flags.DEFINE_string('val_caption_annotations_file', '',
                    'Validation annotations JSON file.')
flags.DEFINE_string(
    'java_classpath', 'tools/stanford-corenlp-full-2015-12-09/*:tools',
    'Default JAVA classpath (Change it if you download StanfordNLP to a different directory).'
)
flags.DEFINE_string('output_directory', '',
                    'Path to store the output annotation file.')
flags.DEFINE_integer('number_of_threads', 10, 'Number of threads.')

FLAGS = flags.FLAGS


def _thread_func(thr_id, images, id_to_captions):
  """Starts thread worker.

  Args:
    images: A list of image meta info.
    id_to_captions: A dict mapping from image id to captions.
  """
  logging.info('Thread %i started, #images=%i.', thr_id, len(images))

  command = ["java", "-mx2g", "-cp", FLAGS.java_classpath, "SceneGraphDemo"]
  proc = subprocess.Popen(command,
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL,
                          universal_newlines=True)

  for i, image in enumerate(images):
    captions = id_to_captions[image['id']]
    sg_list = []

    for caption in captions:
      caption = caption.replace('\n', '')
      proc.stdin.write(caption + '\n')
      proc.stdin.flush()
      outs = proc.stdout.readline().strip('\n')

      # Parse scene graph from string.
      sg = json.loads(outs)
      sg_list.append(sg)
      assert sg['phrase'] == caption

    image['scene_graphs'] = sg_list
    if (i + 1) % 200 == 0:
      logging.info('Thread %i, on %i/%i.', thr_id, i + 1, len(images))

  proc.kill()
  logging.info('Thread %i finished.', thr_id)


def _create_scenegraphs_from_captions(caption_annotations_file,
                                      scenegraph_annotations_file):
  """Creates scene graphs from captions annotations.

  Args:
    caption_annotations_file: JSON file containing caption annotations.
    scenegraph_annotations_file: JSON file containing caption annotations.

  Returns:
    A JSON object with scene graphs injected.
  """
  with tf.io.gfile.GFile(caption_annotations_file, 'r') as fid:
    annots = json.load(fid)
  images, annots = annots['images'], annots['annotations']

  id_to_captions = {}
  for annot in annots:
    image_id = annot['image_id']
    id_to_captions.setdefault(image_id, []).append(annot['caption'])
  assert len(images) == len(id_to_captions)

  # Partition data.
  splits = []
  for thr_id in range(FLAGS.number_of_threads):
    splits.append([
        x for (i, x) in enumerate(images)
        if i % FLAGS.number_of_threads == thr_id
    ])

  # Send data to different threads.
  threads = []
  for thr_id in range(FLAGS.number_of_threads):
    thr = threading.Thread(target=_thread_func,
                           args=(thr_id, splits[thr_id], id_to_captions))

    threads.append(thr)
    thr.start()

  for thr in threads:
    thr.join()

  # Merge scene graphs.
  for i, image in enumerate(images):
    thr_id = i % FLAGS.number_of_threads
    image_with_sg = splits[thr_id][i // FLAGS.number_of_threads]
    assert image['id'] == image_with_sg['id']
    image['scene_graphs'] = image_with_sg['scene_graphs']

  with tf.io.gfile.GFile(scenegraph_annotations_file, 'w') as fid:
    json.dump(images, fid)


def main(_):
  assert FLAGS.train_caption_annotations_file, '`train_caption_annotations_file` missing.'
  assert FLAGS.val_caption_annotations_file, '`val_caption_annotations_file` missing.'
  assert FLAGS.output_directory, '`output_directory` missing.'

  logging.set_verbosity(logging.INFO)

  output_val_file = os.path.join(FLAGS.output_directory,
                                 'scenegraphs_val2017.json')
  output_train_file = os.path.join(FLAGS.output_directory,
                                   'scenegraphs_train2017.json')

  _create_scenegraphs_from_captions(FLAGS.val_caption_annotations_file,
                                    output_val_file)
  _create_scenegraphs_from_captions(FLAGS.train_caption_annotations_file,
                                    output_train_file)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
