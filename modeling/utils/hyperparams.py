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
"""The implementation is adapted from Tensorflow Object Detection API.
Please refer to `https://github.com/tensorflow/models/blob/master/research/object_detection/builders/hyperparams_builder.py` for the original version."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tf_slim as slim

from protos import hyperparams_pb2


class IdentityContextManager(object):
  """Returns an identity context manager that does nothing.

  This is helpful in setting up conditional `with` statement as below:

  with slim.arg_scope(x) if use_slim_scope else IdentityContextManager():
    do_stuff()

  """

  def __enter__(self):
    return None

  def __exit__(self, exec_type, exec_value, traceback):
    del exec_type
    del exec_value
    del traceback
    return False


def build_hyperparams(options, is_training):
  """Builds tf-slim arg_scope for tensorflow ops.

  Args:
    options: an hyperparams_pb2.Hyperparams instance.
    is_training: whether the network is in training mode.

  Returns:
    tf-slim arg_scope containing hyperparameters for ops.

  Raises:
    ValueError: if the options is invalid.
  """
  if not isinstance(options, hyperparams_pb2.Hyperparams):
    raise ValueError('The options has to be an instance of Hyperparams.')

  batch_norm = None
  batch_norm_params = None
  if options.HasField('batch_norm'):
    batch_norm = slim.batch_norm
    batch_norm_params = _build_batch_norm_params(options.batch_norm,
                                                 is_training)

  affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
  if options.op == hyperparams_pb2.Hyperparams.FC:
    affected_ops = [slim.fully_connected]

  def scope_fn():
    with (slim.arg_scope([slim.batch_norm], **batch_norm_params)
          if batch_norm_params is not None else IdentityContextManager()):
      with slim.arg_scope(
          affected_ops,
          weights_regularizer=_build_slim_regularizer(options.regularizer),
          weights_initializer=_build_initializer(options.initializer),
          activation_fn=_build_activation_fn(options.activation),
          normalizer_fn=batch_norm) as sc:
        return sc

  return scope_fn


def _build_activation_fn(activation_fn):
  """Builds a callable activation from config.

  Args:
    activation_fn: hyperparams_pb2.Hyperparams.activation

  Returns:
    Callable activation function.

  Raises:
    ValueError: On unknown activation function.
  """
  if activation_fn == hyperparams_pb2.Hyperparams.NONE:
    return None
  if activation_fn == hyperparams_pb2.Hyperparams.RELU:
    return tf.nn.relu
  if activation_fn == hyperparams_pb2.Hyperparams.RELU_6:
    return tf.nn.relu6
  raise ValueError('Unknown activation function: {}'.format(activation_fn))


def _build_slim_regularizer(regularizer):
  """Builds a tf-slim regularizer from config.

  Args:
    regularizer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf-slim regularizer.

  Raises:
    ValueError: On unknown regularizer.
  """
  regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
  if regularizer_oneof == 'l1_regularizer':
    return slim.l1_regularizer(scale=float(regularizer.l1_regularizer.weight))
  if regularizer_oneof == 'l2_regularizer':
    return slim.l2_regularizer(scale=float(regularizer.l2_regularizer.weight))
  if regularizer_oneof is None:
    return None
  raise ValueError('Unknown regularizer function: {}'.format(regularizer_oneof))


def _build_initializer(initializer):
  """Build a tf initializer from config.

  Args:
    initializer: hyperparams_pb2.Hyperparams.regularizer proto.

  Returns:
    tf initializer.

  Raises:
    ValueError: On unknown initializer.
  """
  initializer_oneof = initializer.WhichOneof('initializer_oneof')
  if initializer_oneof == 'truncated_normal_initializer':
    return tf.truncated_normal_initializer(
        mean=initializer.truncated_normal_initializer.mean,
        stddev=initializer.truncated_normal_initializer.stddev)
  if initializer_oneof == 'random_normal_initializer':
    return tf.random_normal_initializer(
        mean=initializer.random_normal_initializer.mean,
        stddev=initializer.random_normal_initializer.stddev)
  if initializer_oneof == 'variance_scaling_initializer':
    enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.DESCRIPTOR.
                       enum_types_by_name['Mode'])
    mode = enum_descriptor.values_by_number[
        initializer.variance_scaling_initializer.mode].name
    return slim.variance_scaling_initializer(
        factor=initializer.variance_scaling_initializer.factor,
        mode=mode,
        uniform=initializer.variance_scaling_initializer.uniform)
  if initializer_oneof == 'glorot_normal_initializer':
    return tf.glorot_normal_initializer()
  if initializer_oneof == 'glorot_uniform_initializer':
    return tf.glorot_uniform_initializer()

  raise ValueError('Unknown initializer function: {}'.format(initializer_oneof))


def _build_batch_norm_params(batch_norm, is_training):
  """Build a dictionary of batch_norm params from config.

  Args:
    batch_norm: hyperparams_pb2.ConvHyperparams.batch_norm proto.
    is_training: Whether the models is in training mode.

  Returns:
    A dictionary containing batch_norm parameters.
  """
  batch_norm_params = {
      'decay': batch_norm.decay,
      'center': batch_norm.center,
      'scale': batch_norm.scale,
      'epsilon': batch_norm.epsilon,
      'is_training': is_training and batch_norm.train,
  }
  return batch_norm_params
