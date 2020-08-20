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

flags.DEFINE_string('train_image_file', '', 'Training image zip file.')
flags.DEFINE_string('val_image_file', '', 'Validation image zip file.')
flags.DEFINE_string('test_image_file', '', 'Test image zip file.')
flags.DEFINE_string('output_directory', '', 'Output directory.')
flags.DEFINE_integer('number_of_processes', 20, 'Size of the process pool.')
flags.DEFINE_integer('number_of_splits', 10, 'Number of directory splits.')

FLAGS = flags.FLAGS


def _proc_initializer():
  """Runs initializer for the process. """
  logging.info('Proc initializer is called, pid=%i', os.getpid())
  cv2.setUseOptimized(True)
  cv2.setNumThreads(4)

  global cv_selective_search
  cv_selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation(
  )


def _extract_ssbox(filename, encoded_jpg):
  """Extracts ssbox for a single example.

  Args:
    filename: Filename of the image.
    encoded_jpg: JPG data.
  """
  file_bytes = np.fromstring(encoded_jpg, dtype=np.uint8)
  bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  # Resolving an OpenCV issue - resize the image for cv_selective_search.
  height, width = bgr.shape[0], bgr.shape[1]
  if height / width >= 2.2:
    width = int(height / 2.2)
    bgr = cv2.resize(bgr, (width, height))
  elif width / height >= 2.2:
    height = int(width / 2.2)
    bgr = cv2.resize(bgr, (width, height))
  height, width = bgr.shape[0], bgr.shape[1]

  cv_selective_search.setBaseImage(bgr)
  cv_selective_search.switchToSelectiveSearchQuality()

  rects = cv_selective_search.process()
  rects = np.stack([x for x in rects if x[2] >= 20 and x[3] >= 20], axis=0)

  # Normalize the proposals.
  x, y, w, h = [rects[:, i] for i in range(4)]
  proposals = np.stack(
      [y / height, x / width, (y + h) / height, (x + w) / width], axis=-1)

  # Write output NPY file.
  _, file_id = os.path.split(filename)
  file_id = file_id.split('.')[0]

  assert file_id.isdigit()
  output_path = os.path.join(FLAGS.output_directory,
                             str(int(file_id) % FLAGS.number_of_splits),
                             '{}.npy'.format(file_id))

  with open(output_path, 'wb') as fid:
    np.save(fid, proposals)

  logging.info('Finished processing %s, size=%i, shape=%s, output_path=%s.',
               filename, len(encoded_jpg), proposals.shape, output_path)


def _create_ssbox_nparray(zip_file):
  """Extracts ssbox proposals from MSCOCO images.

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
    pool.starmap(_extract_ssbox, _enum_zip(zip_file))


def main(_):
  assert FLAGS.train_image_file, '`train_image_file` missing.'
  assert FLAGS.val_image_file, '`val_image_file` missing.'
  assert FLAGS.test_image_file, '`test_image_file` missing.'
  assert FLAGS.output_directory, '`output_directory` missing.'

  logging.set_verbosity(logging.INFO)

  for i in range(FLAGS.number_of_splits):
    tf.gfile.MakeDirs(os.path.join(FLAGS.output_directory, str(i)))

  _create_ssbox_nparray(FLAGS.val_image_file)
  _create_ssbox_nparray(FLAGS.test_image_file)
  _create_ssbox_nparray(FLAGS.train_image_file)


if __name__ == '__main__':
  app.run(main)
