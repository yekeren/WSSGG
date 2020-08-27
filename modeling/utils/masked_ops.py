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

import tensorflow as tf

_EPSILON = 1e-10
_INF = 1e10


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.

  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = tf.reduce_min(data, dim, keepdims=True)
  masked_maximums = tf.reduce_max(tf.multiply(data - axis_minimums, mask),
                                  dim,
                                  keepdims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.

  Returns:
    masked_minimum: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = tf.reduce_max(data, dim, keepdims=True)
  masked_minimums = tf.reduce_min(tf.multiply(data - axis_maximums, mask),
                                  dim,
                                  keepdims=True) + axis_maximums
  return masked_minimums


def masked_argmax(data, mask, dim=1):
  """Computes the axis wise argmax over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the argmax.

  Returns:
    masked_argmax: N-D `Tensor`.
  """
  axis_minimums = tf.reduce_min(data, dim, keepdims=True)
  return tf.argmax(tf.multiply(data - axis_minimums, mask), dim)


def masked_argmin(data, mask, dim=1):
  """Computes the axis wise argmin over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the argmin.

  Returns:
    masked_argmin: N-D `Tensor`.
  """
  axis_maximums = tf.reduce_max(data, dim, keepdims=True)
  return tf.argmin(tf.multiply(data - axis_maximums, mask), dim)


def masked_sum(data, mask, dim=1):
  """Computes the axis wise sum over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the sum.

  Returns:
    masked_sum: N-D `Tensor`.
      The summed dimension is of size 1 after the operation.
  """
  return tf.reduce_sum(tf.multiply(data, mask), dim, keepdims=True)


def masked_avg(data, mask, dim=1):
  """Computes the axis wise avg over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the avg.

  Returns:
    masked_avg: N-D `Tensor`.
      The averaged dimension is of size 1 after the operation.
  """
  masked_sums = masked_sum(data, mask, dim)
  masked_avgs = tf.div(
      masked_sums, tf.maximum(_EPSILON, tf.reduce_sum(mask, dim,
                                                      keepdims=True)))
  return masked_avgs


def masked_sum_nd(data, mask, dim=1):
  """Computes the axis wise sum over chosen elements.

  Args:
    data: 3-D float `Tensor` of size [n, m, d].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the sum.

  Returns:
    masked_sum: N-D `Tensor`.
      The summed dimension is of size 1 after the operation.
  """
  return tf.reduce_sum(tf.multiply(data, tf.expand_dims(mask, axis=-1)),
                       dim,
                       keepdims=True)


def masked_avg_nd(data, mask, dim=1):
  """Computes the axis wise avg over chosen elements.

  Args:
    data: 3-D float `Tensor` of size [n, m, d].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the avg.

  Returns:
    masked_avg: N-D `Tensor`.
      The averaged dimension is of size 1 after the operation.
  """
  masked_sums = masked_sum_nd(data, mask, dim)
  masked_avgs = tf.div(
      masked_sums,
      tf.maximum(
          _EPSILON,
          tf.expand_dims(tf.reduce_sum(mask, dim, keepdims=True), axis=-1)))
  return masked_avgs


def masked_max_nd(data, mask, dim=1):
  """Computes the axis wise max over chosen elements.

  Args:
    data: 3-D float `Tensor` of size [n, m, d].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the avg.

  Returns:
    masked_max: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = tf.reduce_min(data, dim, keepdims=True)
  masked_maximums = tf.reduce_max(tf.multiply(data - axis_minimums,
                                              tf.expand_dims(mask, -1)),
                                  dim,
                                  keepdims=True) + axis_minimums
  return masked_maximums


def masked_softmax(data, mask, dim=-1):
  """Computes the axis wise softmax over chosen elements.

  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the softmax.

  Returns:
    masked_softmax: 2-D float `Tensor` of size [n, m].
  """
  mask = _INF * (1.0 - mask)
  return tf.nn.softmax(data - mask, axis=dim)
