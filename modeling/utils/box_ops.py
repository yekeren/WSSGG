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
"""Utility functions for bounding boxes, box are [ymin, xmin, ymax, xmax]. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_EPSILON = 1e-10


def flip_left_right(box):
  """Flips the box left-to-right.

  Args:
    box: A [i1,...,iN,  4] float tensor.

  Returns:
    flipped_box: A [i1,...,iN, 4] float tensor.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  return tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=-1)


def area(box):
  """Computes the box area.

  Args:
    box: A [i1,...,iN,  4] float tensor.

  Returns:
    area: Box areas, a [i1,...,iN] float tensor.
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  return tf.maximum(ymax - ymin, 0.0) * tf.maximum(xmax - xmin, 0.0)


def intersect(box1, box2):
  """Computes the intersect box. 

  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    A [i1,...,iN, 4] float tensor.
  """
  ymin1, xmin1, ymax1, xmax1 = tf.unstack(box1, axis=-1)
  ymin2, xmin2, ymax2, xmax2 = tf.unstack(box2, axis=-1)

  ymin = tf.maximum(ymin1, ymin2)
  xmin = tf.maximum(xmin1, xmin2)
  ymax = tf.minimum(ymax1, ymax2)
  xmax = tf.minimum(xmax1, xmax2)

  return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def iou(box1, box2):
  """Computes the IoU between box1 and box2.

  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    iou: A [i1,...,iN] float tensor.
  """
  area_intersect = area(intersect(box1, box2))
  area_union = area(box1) + area(box2) - area_intersect
  return tf.math.divide(area_intersect, tf.maximum(area_union, _EPSILON))
