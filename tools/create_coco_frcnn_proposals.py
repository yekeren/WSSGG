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
import tf_slim as slim
from multiprocessing import Pool
import PIL.Image

from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util

flags.DEFINE_string('train_image_file', '', 'Training image zip file.')
flags.DEFINE_string('val_image_file', '', 'Validation image zip file.')
flags.DEFINE_string('test_image_file', '', 'Test image zip file.')
flags.DEFINE_string('output_directory', '', 'Output directory.')

flags.DEFINE_integer('number_of_processes', 2, 'Size of the process pool.')
flags.DEFINE_string('detection_pipeline_proto', '',
                    'Path to the pipeline.config file..')
flags.DEFINE_string('detection_checkpoint_file', '',
                    'Path to the detection checkpoint file..')

FLAGS = flags.FLAGS

_NUM_PROPOSAL_SUBDIRS = 10


def load_model_proto(filename):
  """Loads object detection model proto.

  Args:
    filename: path to the pipeline proto file.

  Returns:
    model_pb2.DetectionModel.
  """
  model_proto = pipeline_pb2.TrainEvalPipelineConfig()
  with open(filename, 'r') as fp:
    text_format.Merge(fp.read(), model_proto)
  return model_proto.model


def _proc_initializer():
  """Runs initializer for the process. """
  logging.info('Proc initializer is called, pid=%i', os.getpid())

  config = load_model_proto(FLAGS.detection_pipeline_proto)
  model = model_builder.build(config, is_training=False)

  image = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)
  preprocessed_inputs, true_image_shapes = model.preprocess(
      tf.cast(tf.expand_dims(image, 0), tf.float32))
  predictions = model.predict(preprocessed_inputs=preprocessed_inputs,
                              true_image_shapes=true_image_shapes)
  num_proposals = tf.squeeze(predictions['num_proposals'])
  proposals = tf.squeeze(predictions['proposal_boxes_normalized'], 0)
  proposal_features = tf.reduce_mean(predictions['box_classifier_features'],
                                     [1, 2])
  init_fn = slim.assign_from_checkpoint_fn(FLAGS.detection_checkpoint_file,
                                           tf.global_variables())

  global tf_session
  global tf_inputs
  global tf_outputs

  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True

  tf_session = tf.Session(config=config)
  tf_inputs = image
  tf_outputs = [num_proposals, proposals, proposal_features]

  tf_session.run(tf.global_variables_initializer())
  init_fn(tf_session)
  uninitialized_variable_names = tf.report_uninitialized_variables()
  assert len(tf_session.run(uninitialized_variable_names)) == 0


def _extract_frcnn_proposals(filename, encoded_jpg):
  """Extracts frcnn proposals for a single example.

  Args:
    filename: Filename of the image.
    encoded_jpg: JPG data.
  """
  file_bytes = np.fromstring(encoded_jpg, dtype=np.uint8)
  bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  rgb = bgr[::-1]

  # Proposal generation.
  num_proposals, proposals, proposal_features = tf_session.run(
      tf_outputs, feed_dict={tf_inputs: rgb})

  # Write to output file.
  _, file_id = os.path.split(filename)
  file_id = file_id.split('.')[0]

  assert file_id.isdigit()
  npz_path = os.path.join(FLAGS.output_directory,
                          str(int(file_id) % _NUM_PROPOSAL_SUBDIRS),
                          '{}.npz'.format(file_id))
  with open(npz_path, 'wb') as fid:
    np.savez(fid,
             num_proposals=num_proposals,
             proposals=proposals,
             proposal_features=proposal_features)

  logging.info(
      'Finished processing %s, size=%i, num_proposals=%s, npz_path=%s.',
      filename, len(encoded_jpg), num_proposals, npz_path)


def _create_frcnn_proposals_nparray(zip_file):
  """Extracts frcnn proposals from MSCOCO images.

  Args:
    zip_file: ZIP file containing the image files.
  """

  def _enum_zip(zip_file):
    with zipfile.ZipFile(zip_file) as zf:
      for zi in zf.infolist():
        if zi.filename.endswith('.jpg'):
          with zf.open(zi) as fid:
            encoded_jpg = fid.read()
            yield zi.filename, encoded_jpg

  with Pool(processes=FLAGS.number_of_processes,
            initializer=_proc_initializer) as pool:
    for filename, encoded_jpg in _enum_zip(zip_file):
      pool.apply_async(_extract_frcnn_proposals, (filename, encoded_jpg))
    pool.close()
    pool.join()
  logging.info('Done')


def main(_):
  assert FLAGS.train_image_file, '`train_image_file` missing.'
  assert FLAGS.val_image_file, '`val_image_file` missing.'
  assert FLAGS.test_image_file, '`test_image_file` missing.'
  assert FLAGS.output_directory, '`output_directory` missing.'

  logging.set_verbosity(logging.INFO)

  for i in range(_NUM_PROPOSAL_SUBDIRS):
    tf.gfile.MakeDirs(os.path.join(FLAGS.output_directory, str(i)))

  _create_frcnn_proposals_nparray(FLAGS.train_image_file)
  # _create_frcnn_proposals_nparray(FLAGS.val_image_file)
  # _create_frcnn_proposals_nparray(FLAGS.test_image_file)


if __name__ == '__main__':
  app.run(main)
