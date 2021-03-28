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

from absl import app
from absl import flags
from absl import logging

import os
import tensorflow as tf
from google.protobuf import text_format

from protos import pipeline_pb2
from protos import model_pb2
from modeling import trainer

flags.DEFINE_string('type', None,
                    'Module type, reserved for distributed training.')

flags.DEFINE_string('model_dir', None,
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto', None, 'Path to the pipeline proto file.')

flags.DEFINE_boolean('use_mirrored_strategy', False,
                     'If true, use mirrored strategy for training.')

flags.DEFINE_enum('job', 'train_and_evaluate',
                  ['train_and_evaluate', 'train', 'evaluate', 'test', 'debug'],
                  'Job type.')

flags.DEFINE_string('closed_vocabulary_file', None, 'Path to closed vocab file.')
flags.DEFINE_string('testing_input_pattern', None,
                    'Path to input testing file.')

flags.DEFINE_string('testing_res_file', None,
                    'Path to output testing result file.')

FLAGS = flags.FLAGS


def _load_pipeline_proto(filename):
  """Loads pipeline proto from file.

  Args:
    filename: Path to the pipeline config file.

  Returns:
    An instance of pipeline_pb2.Pipeline.
  """
  with tf.io.gfile.GFile(filename, 'r') as fp:
    return text_format.Merge(fp.read(), pipeline_pb2.Pipeline())


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  tf.io.gfile.makedirs(FLAGS.model_dir)

  saved_pipeline_proto = os.path.join(FLAGS.model_dir, 'pipeline.pbtxt')
  if os.path.isfile(saved_pipeline_proto):
    pipeline_proto = _load_pipeline_proto(saved_pipeline_proto)
  else:
    pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)
    tf.io.gfile.copy(FLAGS.pipeline_proto, saved_pipeline_proto, overwrite=True)

  tf.set_random_seed(pipeline_proto.seed)

  if 'train_and_evaluate' == FLAGS.job:
    trainer.train_and_evaluate(
        pipeline_proto=pipeline_proto,
        model_dir=FLAGS.model_dir,
        use_mirrored_strategy=FLAGS.use_mirrored_strategy)
  elif 'train' == FLAGS.job:
    trainer.train(pipeline_proto=pipeline_proto,
                  model_dir=FLAGS.model_dir,
                  use_mirrored_strategy=FLAGS.use_mirrored_strategy)
  elif 'evaluate' == FLAGS.job:
    trainer.evaluate(pipeline_proto=pipeline_proto, model_dir=FLAGS.model_dir)
  elif 'test' == FLAGS.job:
    if FLAGS.testing_input_pattern:
      pipeline_proto.test_reader.caption_graph_reader.input_pattern[:] = [
          FLAGS.testing_input_pattern
      ]
    if FLAGS.closed_vocabulary_file:
      pipeline_proto.model.Extensions[model_pb2.Cap2SG.ext].preprocess_options.closed_vocabulary_file = FLAGS.closed_vocabulary_file
    testing_res_file = FLAGS.testing_res_file if FLAGS.testing_res_file else 'testing_result_file.csv'
    trainer.evaluate(pipeline_proto=pipeline_proto,
                     model_dir=FLAGS.model_dir,
                     testing=True,
                     testing_res_file=testing_res_file)
  elif 'debug' == FLAGS.job:
    trainer.debug(pipeline_proto=pipeline_proto, model_dir=FLAGS.model_dir)
  else:
    raise ValueError('Invalid job type %s!' % FLAGS.job)


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
