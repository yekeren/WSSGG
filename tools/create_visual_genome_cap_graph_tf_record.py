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
import json
import pickle
import gc

import numpy as np
import tensorflow as tf
from graph_nets import utils_np

from modeling.utils.box_ops import py_iou

gc.disable()

flags.DEFINE_string('text_graphs_json_file', '',
                    'Json file storing the text graphs.')
flags.DEFINE_string('split_pkl_file', '',
                    'Pickle file denoting the train/test splits.')
flags.DEFINE_integer('num_val_examples', 1000,
                     'Number of examples for validation.')
flags.DEFINE_string('proposal_feature_dir', '',
                    'Directory storing proposal feature.')
flags.DEFINE_string('scene_graph_pkl_file', '',
                    'Pickle file storing scene graphs annotations.')
flags.DEFINE_string('embedding_pkl_file', '',
                    'Path to the file storing entity/predicate embeddings.')
flags.DEFINE_string('output_directory', '', 'Path to the output directory')

FLAGS = flags.FLAGS

_FEATURE_DIMS = 1536

np.random.seed(286)

_NUM_PROPOSAL_SUBDIRS = 10


def _read_proposal_features(image_id, proposal_feature_dir):
  """Reads proposal feature from lmdb. 

  Args:
    image_id: Image id.

  Returns:
    A [num_proposals, _FEATURE_DIMS] nparray.
  """
  filename = os.path.join(proposal_feature_dir,
                          str(image_id % _NUM_PROPOSAL_SUBDIRS),
                          '{}.npz'.format(image_id))
  if not os.path.isfile(filename):
    logging.info('File %s does not exist.', filename)
    return None, None

  with tf.io.gfile.GFile(filename, 'rb') as fid:
    data = np.load(fid)
  return data['proposals'], data['proposal_features']


def _create_tf_example(image_id, proposals, proposal_features, scene_graph,
                       caption_graph):
  """Creates tf example.

  Args:
    image_id: Image id.
    proposals: A [num_proposals, 4] nparray, denoting [x1, y1, x2, y2].
    proposal_features: A [num_proposals, _FEATURE_DIMS] nparray.
    scene_graph: A list of scene graph triples, including the following
      keys: `subject`, `object`, `predicate`, `subject_box`, `object_box`.
    caption_graph: A dict containing graph-like data, including the 
      following keys: `n_node`, `n_edge`, `nodes`, `edges`, `senders`, `receivers`.

  Returns:
    tf_example: A tf.train.Example proto.
  """

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode('utf8')]))

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _string_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[x.encode('utf8') for x in value]))

  def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  # Encode proposals.
  feature_dict = {
      'id':
          _int64_feature(image_id),
      'image/n_proposal':
          _int64_feature(len(proposals)),
      'image/proposal/bbox/ymin':
          _float_feature_list(proposals[:, 0].tolist()),
      'image/proposal/bbox/xmin':
          _float_feature_list(proposals[:, 1].tolist()),
      'image/proposal/bbox/ymax':
          _float_feature_list(proposals[:, 2].tolist()),
      'image/proposal/bbox/xmax':
          _float_feature_list(proposals[:, 3].tolist()),
      'image/proposal/feature':
          _float_feature_list(proposal_features.flatten().tolist()),
  }

  feature_dict.update({
      # Caption pseudo graph.
      'caption_graph/caption':
          _string_feature_list(caption_graph['caption']),
      'caption_graph/n_node':
          _int64_feature_list(caption_graph['n_node']),
      'caption_graph/n_edge':
          _int64_feature_list(caption_graph['n_edge']),
      'caption_graph/nodes':
          _string_feature_list(caption_graph['nodes']),
      'caption_graph/edges':
          _string_feature_list(caption_graph['edges']),
      'caption_graph/senders':
          _int64_feature_list(caption_graph['senders']),
      'caption_graph/receivers':
          _int64_feature_list(caption_graph['receivers']),
      # Ground-truth.
      'scene_graph/n_relation':
          _int64_feature(scene_graph['n_relation']),
      'scene_graph/subject':
          _string_feature_list(scene_graph['subject']),
      'scene_graph/predicate':
          _string_feature_list(scene_graph['predicate']),
      'scene_graph/object':
          _string_feature_list(scene_graph['object']),
      'scene_graph/subject/bbox/ymin':
          _float_feature_list(scene_graph['subject_box'][:, 1].tolist()),
      'scene_graph/subject/bbox/xmin':
          _float_feature_list(scene_graph['subject_box'][:, 0].tolist()),
      'scene_graph/subject/bbox/ymax':
          _float_feature_list(scene_graph['subject_box'][:, 3].tolist()),
      'scene_graph/subject/bbox/xmax':
          _float_feature_list(scene_graph['subject_box'][:, 2].tolist()),
      'scene_graph/object/bbox/ymin':
          _float_feature_list(scene_graph['object_box'][:, 1].tolist()),
      'scene_graph/object/bbox/xmin':
          _float_feature_list(scene_graph['object_box'][:, 0].tolist()),
      'scene_graph/object/bbox/ymax':
          _float_feature_list(scene_graph['object_box'][:, 3].tolist()),
      'scene_graph/object/bbox/xmax':
          _float_feature_list(scene_graph['object_box'][:, 2].tolist()),
  })

  tf_example = tf.train.Example(features=tf.train.Features(
      feature=feature_dict))
  return tf_example


def _create_tf_record_from_annotations(image_ids, proposal_feature_dir,
                                       id_to_scenegraph, tf_record_file,
                                       num_output_parts):
  """Creates tf record files.

  Args:
    image_ids: A list of integer denoting the image ids.
    id_to_scenegraph: A dict mapping from image ids to scene graph annotations.
    tf_record_file: A tfrecord filename denoting the output file.
  """
  writers = []
  for i in range(num_output_parts):
    filename = tf_record_file + '-%05d-of-%05d' % (i, num_output_parts)
    writers.append(tf.io.TFRecordWriter(filename))

  for i, image_id in enumerate(image_ids):
    if image_id not in id_to_scenegraph:
      continue

    scene_graph = id_to_scenegraph[image_id]['scene_graph']
    caption_graph = id_to_scenegraph[image_id]['caption_graph']

    proposals, proposal_features = _read_proposal_features(
        image_id, proposal_feature_dir)
    if proposals is None or proposal_features is None:
      continue
    # TODO, trim flag.
    proposals, proposal_features = proposals[:20, :], proposal_features[:20, :]
    assert proposals.shape[0] == proposal_features.shape[0]

    tf_example = _create_tf_example(image_id, proposals, proposal_features,
                                    scene_graph, caption_graph)
    writers[i % num_output_parts].write(tf_example.SerializeToString())

    if (i + 1) % 100 == 0:
      logging.info('On example %i/%i', i + 1, len(image_ids))

  for writer in writers:
    writer.close()
  logging.info('Processed %i examples.', len(image_ids))


def _insert_entity(entity_names, entity_boxes, box, name):
  for i in range(len(entity_names)):
    if name == entity_names[i] and py_iou(box, entity_boxes[i]) > 0.5:
      return i
  entity_names.append(name)
  entity_boxes.append(box)
  return len(entity_names) - 1


def _read_scene_graph_annotations_keyed_by_image_id(scene_graph_pkl_file,
                                                    scene_graph_meta, id_to_textgraph):
  """Reads scene graphs keyed by image id.

  Args:
    scene_graph_pkl_file: Path to the ground-truth scene graph annotations pickle file.
    scene_graph_meta: A meta structure provided by the VSPNET authors,
      it maps from type (an integer) to text labels.

  Returns:
    A dict mapping image id to scene graphs.
  """
  id_to_entity, id_to_predicate = (scene_graph_meta['idx_to_label'],
                                   scene_graph_meta['idx_to_predicate'])
  with tf.io.gfile.GFile(scene_graph_pkl_file, 'rb') as fid:
    sg_dict = pickle.load(fid)

  count = count_entity = count_triples = 0
  skipped = 0

  data = {}
  for image_id, entity_label, pred_label, triples in zip(
      sg_dict['img_ids'], sg_dict['entity_label'], sg_dict['pred_label'],
      sg_dict['triples']):
    assert len(triples) == len(pred_label)

    if len(entity_label) == 0:
      skipped += 1
      continue

    count += 1
    count_entity += len(entity_label)
    count_triples += len(triples)

    # Scene graph data (sg_*).
    sg_sub_names, sg_obj_names, sg_pred_names = [], [], []
    sg_sub_boxes, sg_obj_boxes = [], []

    scene_graph = []
    for triple in triples:
      try:
        sub, pred, obj = (id_to_entity[str(triple['subject_type'])],
                          id_to_predicate[str(triple['predicate_type'])],
                          id_to_entity[str(triple['object_type'])])
      except Exception as ex:
        sub, pred, obj = (id_to_entity[triple['subject_type']],
                          id_to_predicate[triple['predicate_type']],
                          id_to_entity[triple['object_type']])

      # Ground-truth scene graph.
      sg_sub_names.append(sub)
      sg_obj_names.append(obj)
      sg_pred_names.append(pred)
      sg_sub_boxes.append(triple['subject_box'])
      sg_obj_boxes.append(triple['object_box'])
      scene_graph.append({
          'predicate': pred,
          'subject': sub,
          'subject_box': triple['subject_box'],
          'object': obj,
          'object_box': triple['object_box'],
      })
    sg_n_relation = len(sg_pred_names)

    # Caption graph data (cg_*).
    data_dict_list = []
    captions = id_to_textgraph[image_id]['captions']

    for sg in id_to_textgraph[image_id]['scene_graphs']:
      entities, relations = sg['entities'], sg['relations']

      # Entities.
      nodes = []
      for e in entities:
        att_str = ','.join([
            x['span']
            for x in e['modifiers']
            if x['dep'] not in ['det', 'nummod']
        ])
        nodes.append(e['head'] + (':' + att_str if att_str else ''))

      # Edges.
      senders, receivers, edges = [], [], []
      for r in relations:
        senders.append(r['subject'])
        receivers.append(r['object'])
        edges.append(r['relation'])

      data_dict_list.append({
          "nodes": nodes,
          "edges": edges,
          "senders": senders,
          "receivers": receivers
      })

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dict_list)
    data[image_id] = {
        'scene_graph': {
            'n_relation':
                sg_n_relation,
            'subject':
                sg_sub_names,
            'subject_box':
                np.stack(sg_sub_boxes) if sg_n_relation else np.zeros((0, 4)),
            'object':
                sg_obj_names,
            'object_box':
                np.stack(sg_obj_boxes) if sg_n_relation else np.zeros((0, 4)),
            'predicate':
                sg_pred_names
        },
        'caption_graph': {
            'caption': captions,
            'n_node': graphs_tuple.n_node,
            'n_edge': graphs_tuple.n_edge,
            'nodes': graphs_tuple.nodes,
            'edges': graphs_tuple.edges,
            'senders': graphs_tuple.senders,
            'receivers': graphs_tuple.receivers,
        }
    }

  logging.info('In total: %i objects, %i relationships, skipped %i.',
               count_entity, count_triples, skipped)
  logging.info('In average: %.2lf objects, %.2lf relationships.',
               count_entity / count, count_triples / count)
  return data


def main(_):
  assert FLAGS.split_pkl_file, '`split_pkl_file` missing.'
  assert FLAGS.proposal_feature_dir, '`proposal_feature_dir` missing.'
  assert FLAGS.scene_graph_pkl_file, '`scene_graph_pkl_file` missing.'
  assert FLAGS.embedding_pkl_file, '`embedding_pkl_file` missing.'
  assert FLAGS.output_directory, '`output_directory` missing.'

  output_directory = FLAGS.output_directory
  tf.gfile.MakeDirs(output_directory)

  # Load scene_graph_meta, entity/predicate embeddings used in the VSPNET.
  # Note: the first token in the embedding matrix is OOV.
  with tf.io.gfile.GFile(FLAGS.embedding_pkl_file, 'rb') as fid:
    scene_graph_meta, entity_emb, predicate_emb = pickle.load(fid)

  with tf.io.gfile.GFile(
      os.path.join(output_directory, 'scene_graph_meta.json'), 'w') as fid:
    json.dump(scene_graph_meta, fid, indent=2)
  np.save(os.path.join(output_directory, 'entity_emb.npy'), entity_emb)
  np.save(os.path.join(output_directory, 'predicate_emb.npy'), predicate_emb)

  logging.info('entity_emb shape=%s, predicate_emb shape=%s', entity_emb.shape,
               predicate_emb.shape)

  # Data splits, details are in `https://github.com/alirezazareian/vspnet`.
  with tf.io.gfile.GFile(FLAGS.split_pkl_file, 'rb') as fid:
    image_ids, train_indices, test_indices = pickle.load(fid)
  train_ids, test_ids = image_ids[train_indices], image_ids[test_indices]
  np.random.shuffle(train_ids)
  train_ids, val_ids = (train_ids[FLAGS.num_val_examples:],
                        train_ids[:FLAGS.num_val_examples])
  logging.info("Data splits: #train=%i, #val=%i, #test=%i", len(train_ids),
               len(val_ids), len(test_ids))

  # # Load proposals.
  # with tf.io.gfile.GFile(FLAGS.proposal_coord_pkl_file, 'rb') as fid:
  #   id_to_proposals = pickle.load(fid)
  # logging.info('Proposals: #=%i', len(id_to_proposals))

  # Load scene graphs.
  id_to_textgraph = _read_text_graphs(FLAGS.text_graphs_json_file)
  id_to_scenegraph = _read_scene_graph_annotations_keyed_by_image_id(
      FLAGS.scene_graph_pkl_file, scene_graph_meta, id_to_textgraph)

  # Generate tfrecord files.
  _create_tf_record_from_annotations(
      val_ids, FLAGS.proposal_feature_dir, id_to_scenegraph,
      os.path.join(output_directory, 'val.tfrecord'), 1)
  _create_tf_record_from_annotations(
      train_ids, FLAGS.proposal_feature_dir, id_to_scenegraph,
      os.path.join(output_directory, 'train.tfrecord'), 1)
  _create_tf_record_from_annotations(
      test_ids, FLAGS.proposal_feature_dir, id_to_scenegraph,
      os.path.join(output_directory, 'test.tfrecord'), 1)

  logging.info('Done')

def _read_text_graphs(file_name):
  with tf.io.gfile.GFile(file_name, 'rb') as f:
    annots = json.load(f)
  data = {}
  for annot in annots :
    data[annot['image_id']] = annot
  logging.info('Load %i text graphs examples.', len(data))
  return data


if __name__ == '__main__':
  app.run(main)
