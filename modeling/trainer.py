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

from absl import logging

import tensorflow as tf

from modeling.utils import optimization
from modeling.utils import learning_rate_schedule

from models import builder
from readers import reader
from protos import pipeline_pb2


def _summarize_variables(var_list):
  """Summarizes variables.

  Args:
    var_list: A list of variables.
  """
  for var in var_list:
    if 'global_step' not in var.op.name:
      var_norm = tf.norm(var)
      tf.summary.scalar('summarize_vars/' + var.op.name, var_norm)


def _create_model_fn(pipeline_proto, is_chief=True):
  """Creates a callable that build the model.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.

  Returns:
    A callable that takes [features, labels, mode, params] as inputs.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  def _model_fn(features, labels, mode, params):
    """Creates the model.

    Args:
      features: A dict mapping from names to tensors, denoting the features.
      labels: A dict mapping from names to tensors, denoting the labels.
      mode: Mode parameter required by the estimator.
      params: Additional parameters used for creating the model.

    Returns:
      An instance of EstimatorSpec.
    """
    is_training = (tf.estimator.ModeKeys.TRAIN == mode)
    logging.info("Current mode is %s, is_training=%s", mode, is_training)

    model = builder.build(pipeline_proto.model, is_training)

    # Predict resutls.
    predictions = model.predict(features)

    # Compute losses. Note: variables created in build_loss are not trainable.
    losses = model.build_losses(features, predictions)
    for name, loss in losses.items():
      tf.compat.v1.summary.scalar('metrics/' + name, loss)
      tf.losses.add_loss(loss)
    for loss in tf.compat.v1.losses.get_regularization_losses():
      tf.summary.scalar(
          "regularization/" + '/'.join(loss.op.name.split('/')[:2]), loss)
    total_loss = tf.compat.v1.losses.get_total_loss(
        add_regularization_losses=True)

    # Get variables_to_train.
    variables_to_train = model.get_variables_to_train()
    scaffold = model.get_scaffold()

    train_op = None
    eval_metric_ops = None

    if tf.estimator.ModeKeys.TRAIN == mode:
      _summarize_variables(tf.compat.v1.global_variables())
      global_step = tf.compat.v1.train.get_global_step()

      # Set learning rate.
      train_config = pipeline_proto.train_config
      lr_schedule_fn = learning_rate_schedule.create_learning_rate_schedule(
          train_config.learning_rate_schedule)
      learning_rate = lr_schedule_fn(global_step)
      tf.compat.v1.summary.scalar('metrics/learning_rate', learning_rate)

      # Use optimizer to minimize loss.
      optimizer = optimization.create_optimizer(train_config.optimizer,
                                                learning_rate=learning_rate)

      def transform_grads_fn(grads):
        if train_config.HasField('max_gradient_norm'):
          grads = tf.contrib.training.clip_gradient_norms(
              grads, max_norm=train_config.max_gradient_norm)
        return grads

      train_op = tf.contrib.training.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=variables_to_train,
          transform_grads_fn=transform_grads_fn,
          summarize_gradients=True)

      if callable(getattr(model, 'get_adversarial_variables_to_train', None)):
        logging.info('Create optimizer for adversarial training.')

        adversarial_loss = 0
        for name, loss in losses.items():
          adversarial_loss -= loss * train_config.adversarial_loss_weight

        adversarial_optimizer = optimization.create_optimizer(
            train_config.optimizer, learning_rate=learning_rate)

        (adversarial_variables_to_train
        ) = model.get_adversarial_variables_to_train()
        adversarial_train_op = tf.contrib.training.create_train_op(
            adversarial_loss,
            adversarial_optimizer,
            variables_to_train=adversarial_variables_to_train,
            transform_grads_fn=transform_grads_fn,
            summarize_gradients=True)
        train_op = tf.group(train_op, adversarial_train_op)

    elif tf.estimator.ModeKeys.EVAL == mode:

      eval_metric_ops = model.build_metrics(features, predictions)

    elif tf.estimator.ModeKeys.PREDICT == mode:

      # Add input tensors to the predictions.
      predictions.update(features)

      # Create additional tensors if specified.
      create_additional_predictions = params.get(
          'create_additional_predictions', None)

      if create_additional_predictions:
        assert callable(create_additional_predictions)

        predictions.update(create_additional_predictions(
            tf.get_default_graph()))

    # Merge summaries.
    summary_saver_hook = tf.estimator.SummarySaverHook(
        summary_op=tf.compat.v1.summary.merge_all(),
        save_steps=pipeline_proto.train_config.save_summary_steps)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      training_hooks=[summary_saver_hook],
                                      scaffold=scaffold)

  return _model_fn


def train_and_evaluate(pipeline_proto, model_dir, use_mirrored_strategy=False):
  """Starts the estimator trainval loop.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    model_dir: Path to the directory saving checkpoint files.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  # Create train_spec.
  train_config = pipeline_proto.train_config
  train_input_fn = reader.get_input_fn(pipeline_proto.train_reader,
                                       is_training=True)
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                      max_steps=train_config.max_steps)

  # Create eval_spec.
  eval_config = pipeline_proto.eval_config
  eval_input_fn = reader.get_input_fn(pipeline_proto.eval_reader,
                                      is_training=False)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=eval_config.steps,
      start_delay_secs=eval_config.start_delay_secs,
      throttle_secs=eval_config.throttle_secs,
      exporters=[])

  # Create run_config.
  strategy = None
  if use_mirrored_strategy:
    strategy = tf.contrib.distribute.MirroredStrategy()
  run_config = tf.estimator.RunConfig(
      train_distribute=strategy,
      session_config=tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=tf.GPUOptions(
                                        allow_growth=True,
                                        per_process_gpu_memory_fraction=1.0)),
      save_summary_steps=train_config.save_summary_steps,
      save_checkpoints_steps=train_config.save_checkpoints_steps,
      keep_checkpoint_max=train_config.keep_checkpoint_max,
      log_step_count_steps=train_config.log_step_count_steps)

  # Train and evaluate.
  model_fn = _create_model_fn(pipeline_proto, is_chief=run_config.is_chief)
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     config=run_config)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def train(pipeline_proto, model_dir, use_mirrored_strategy=False):
  """Starts the estimator training loop.

  Args:
    pipeline_proto: An instance of pipeline_pb2.Pipeline.
    model_dir: Path to the directory saving checkpoint files.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  # Create train_spec.
  train_config = pipeline_proto.train_config
  train_input_fn = reader.get_input_fn(pipeline_proto.train_reader,
                                       is_training=True)

  # Create run_config.
  strategy = None
  if use_mirrored_strategy:
    strategy = tf.contrib.distribute.MirroredStrategy()

  run_config = tf.estimator.RunConfig(
      train_distribute=strategy,
      session_config=tf.ConfigProto(
          allow_soft_placement=True,
          gpu_options=tf.GPUOptions(allow_growth=True)),
      save_summary_steps=train_config.save_summary_steps,
      save_checkpoints_steps=train_config.save_checkpoints_steps,
      keep_checkpoint_max=train_config.keep_checkpoint_max,
      log_step_count_steps=train_config.log_step_count_steps)

  # Train and evaluate.
  model_fn = _create_model_fn(pipeline_proto, is_chief=run_config.is_chief)
  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     config=run_config)
  estimator.train(train_input_fn, max_steps=train_config.max_steps)


def predict(pipeline_proto,
            model_dir=None,
            yield_single_examples=False,
            params=None):
  """Generates inference results.

  Args:
    pipeline_proto: A pipeline_pb2.Pipeline proto.
    model_dir: Path to the directory saving model checkpoints.
    yield_single_examples: If true, yield a single example.
    params: Additional parameters to be passed to tf.Estimator.

  Yields:
    example: inference results.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  predict_input_fn = reader.get_input_fn(pipeline_proto.eval_reader,
                                         is_training=False)

  # Create estimator.

  model_fn = _create_model_fn(pipeline_proto)

  run_config = tf.estimator.RunConfig(session_config=tf.ConfigProto(
      gpu_options=tf.GPUOptions(allow_growth=True)))

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     config=run_config,
                                     params=params)

  # Predict results.

  checkpoint_path = tf.train.latest_checkpoint(model_dir)
  assert checkpoint_path is not None
  for example in estimator.predict(input_fn=predict_input_fn,
                                   checkpoint_path=checkpoint_path,
                                   yield_single_examples=yield_single_examples):
    yield example
