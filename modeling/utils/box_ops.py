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

import numpy as np
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


def center(box):
  """Computes the box center.
  Args:
    box: A [i1,...,iN,  4] float tensor.

  Returns:
    center: Box centers, a [i1,...,iN, 2] tensor denoting [ycenter, xcenter].
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  ycenter = (ymin + ymax) / 2
  xcenter = (xmin + xmax) / 2
  return tf.stack([ycenter, xcenter], axis=-1)


def size(box):
  """Computes the box size.

  Args:
    box: A [i1,...,iN,  4] float tensor.

  Returns:
    size: Box sizes, a [i1,...,iN, 2] tensor denoting [height, width].
  """
  ymin, xmin, ymax, xmax = tf.unstack(box, axis=-1)
  height = ymax - ymin
  width = xmax - xmin
  return tf.stack([height, width], axis=-1)


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


def x_intersect_len(box1, box2):
  """Computes the length of the x intersect.
  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    A [i1,...,iN] float tensor.
  """
  ymin1, xmin1, ymax1, xmax1 = tf.unstack(box1, axis=-1)
  ymin2, xmin2, ymax2, xmax2 = tf.unstack(box2, axis=-1)

  xmin = tf.maximum(xmin1, xmin2)
  xmax = tf.minimum(xmax1, xmax2)

  return tf.maximum(xmax - xmin, 0.0)


def y_intersect_len(box1, box2):
  """Computes the length of the y intersect.
  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    A [i1,...,iN] float tensor.
  """
  ymin1, xmin1, ymax1, xmax1 = tf.unstack(box1, axis=-1)
  ymin2, xmin2, ymax2, xmax2 = tf.unstack(box2, axis=-1)

  ymin = tf.maximum(ymin1, ymin2)
  ymax = tf.minimum(ymax1, ymax2)

  return tf.maximum(ymax - ymin, 0.0)


def x_distance(box1, box2):
  """Computes the x-distance between the two centers.
  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    distance: A [i1,...,iN] float tensor.
  """
  x1 = tf.unstack(center(box1), axis=-1)[1]
  x2 = tf.unstack(center(box2), axis=-1)[1]
  return x1 - x2


def y_distance(box1, box2):
  """Computes the y-distance between the two centers.
  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    distance: A [i1,...,iN] float tensor.
  """
  y1 = tf.unstack(center(box1), axis=-1)[0]
  y2 = tf.unstack(center(box2), axis=-1)[0]
  return y1 - y2


def py_area(box):
  """Computes the box area.

  Args:
    box: A [i1,...,iN, 4] float array.

  Returns:
    area: Box areas, a [batch] float tensor.
  """
  ymin, xmin, ymax, xmax = [
      np.squeeze(x, -1) for x in np.split(box, [1, 2, 3], axis=-1)
  ]
  return np.maximum(ymax - ymin, 0.0) * np.maximum(xmax - xmin, 0.0)


def py_intersect(box1, box2):
  """Computes the intersect box. 

  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    A [i1,...,iN, 4] float tensor.
  """
  ymin1, xmin1, ymax1, xmax1 = [
      np.squeeze(x, -1) for x in np.split(box1, [1, 2, 3], axis=-1)
  ]
  ymin2, xmin2, ymax2, xmax2 = [
      np.squeeze(x, -1) for x in np.split(box2, [1, 2, 3], axis=-1)
  ]

  ymin = np.maximum(ymin1, ymin2)
  xmin = np.maximum(xmin1, xmin2)
  ymax = np.minimum(ymax1, ymax2)
  xmax = np.minimum(xmax1, xmax2)

  return np.stack([ymin, xmin, ymax, xmax], axis=-1)


def py_iou(box1, box2):
  """Computes the IoU between box1 and box2.

  Args:
    box1: A [i1,...,iN, 4] float tensor.
    box2: A [i1,...,iN, 4] float tensor.

  Returns:
    iou: A [i1,...,iN] float tensor.
  """
  area_intersect = py_area(py_intersect(box1, box2))
  area_union = py_area(box1) + py_area(box2) - area_intersect
  return np.divide(area_intersect, np.maximum(area_union, _EPSILON))
