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

import numpy as np
import tensorflow as tf

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from object_detection.utils.visualization_utils import STANDARD_COLORS
from object_detection.utils.visualization_utils import draw_bounding_box_on_image_array


def draw_arrow_on_image(image,
                        y1,
                        x1,
                        y2,
                        x2,
                        color='red',
                        thickness=4,
                        display_str_list=(),
                        use_normalized_coordinates=True):
  """Adds an arrow to an image.

  Arrow coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    y1: y position of the starting point.
    x1: x position of the starting point.
    y2: y position of the ending point.
    x2: x position of the ending point.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (x1, x2, y1, y2) = (x1 * im_width, x2 * im_width, y1 * im_height,
                        y2 * im_height)

  if thickness > 0:
    draw.line([(x1, y1), (x2, y2)], width=thickness, fill=color)
    draw.ellipse(
        (x2 - thickness, y2 - thickness, x2 + thickness, y2 + thickness),
        fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 48)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  text_bottom = (y1 + y2) // 2
  text_left = (x1 + x2) // 2

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(text_left, text_bottom - text_height - 2 * margin),
                    (text_left + text_width, text_bottom)],
                   fill=color)
    draw.text((text_left + margin, text_bottom - text_height - margin),
              display_str,
              fill='black',
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_arrow_on_image_array(image,
                              y1,
                              x1,
                              y2,
                              x2,
                              color='red',
                              thickness=4,
                              display_str_list=(),
                              use_normalized_coordinates=True):
  """Adds an arrow to an image (numpy array).

  Arrow coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    y1: y position of the starting point.
    x1: x position of the starting point.
    y2: y position of the ending point.
    x2: x position of the ending point.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_arrow_on_image(image_pil, y1, x1, y2, x2, color, thickness,
                      display_str_list, use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_arrow_py_func_fn(*args):
  """Arrow drawing function that can be wrapped in a tf.py_func.

  Args:
    * args: First 5 positional arguments must be:
      image - uint8 numpy array with shape (height, width, 3).
      total - a integer denoting the actual number of arrows.
      y1_list - a numpy array of shape [max_pad_num],
      x1_list - a numpy array of shape [max_pad_num],
      y2_list - a numpy array of shape [max_pad_num],
      x2_list - a numpy array of shape [max_pad_num],
      labels - a numpy array of shape [max_pad_num].
      scores - a numpy array of shape [max_pad_num].
  """
  image, total, y1_list, x1_list, y2_list, x2_list, labels, scores = args
  for i in range(total - 1, -1, -1):
    y1, x1, y2, x2 = y1_list[i], x1_list[i], y2_list[i], x2_list[i]
    display_str = ''
    if labels is not None and scores is not None:
      display_str = '%i%% %s' % (int(scores[i] * 100), labels[i].decode('utf8'))
    elif labels is not None:
      display_str = '%s' % (labels[i].decode('utf8'))
    elif scores is not None:
      display_str = '%i%%' % (int(scores[i] * 100))
    color = STANDARD_COLORS[i % len(STANDARD_COLORS)]

    draw_arrow_on_image_array(image,
                              y1,
                              x1,
                              y2,
                              x2,
                              color,
                              display_str_list=[display_str])
  return image


def draw_bounding_box_py_func_fn(*args):
  """Bounding box drawing function that can be wrapped in a tf.py_func.

  Args:
    *args: First 5 positional arguments must be:
      image - uint8 numpy array with shape (height, width, 3).
      total - a integer denoting the actual number of boxes.
      boxes - a numpy array of shape [max_pad_num, 4].
      labels - a numpy array of shape [max_pad_num].
      scores - a numpy array of shape [max_pad_num].

  Returns:
    uint8 numpy array with shape (height, width, 3) with overlaid boxes.
  """
  image, total, boxes, labels, scores = args[:5]
  thickness = 1
  if len(args) > 5:
    thickness = args[5]
  for i in range(total - 1, -1, -1):
    ymin, xmin, ymax, xmax = boxes[i]
    display_str = ''
    if labels is not None and scores is not None:
      display_str = '%i%% %s' % (int(scores[i] * 100), labels[i].decode('utf8'))
    elif labels is not None:
      try:
        display_str = '%s' % (labels[i].decode('utf8'))
      except Exception as ex:
        display_str = labels[i]
    elif scores is not None:
      display_str = '%i%%' % (int(scores[i] * 100))
    color = STANDARD_COLORS[i % len(STANDARD_COLORS)]

    display_str_list = [display_str] if display_str else []
    draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color,
                                     thickness=thickness,
                                     display_str_list=display_str_list)
  return image


def draw_bounding_boxes_on_image_tensors(image, total, boxes, labels, scores):
  """Draws bounding boxes on batch of image tensors.

  Args:
    image: A [batch, height, width, 3] uint8 tensor.
    total: A [batch] int tensor denoting number of boxes.
    boxes: A [batch, max_pad_num, 4] float tensor, normalized boxes,
      in the format of [ymin, xmin, ymax, xmax].
    labels: A [batch, max_pad_num] string tensor denoting labels.
    scores: A [batch, max_pad_num] float tensor denoting scores.

  Returns:
    a [batch, height, width, 3] uint8 tensor with boxes drawn on top.
  """

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    image_with_boxes = tf.py_func(draw_bounding_box_py_func_fn,
                                  image_and_detections, tf.uint8)
    return image_with_boxes

  elems = [image, total, boxes, labels, scores]
  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images
