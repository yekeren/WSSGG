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
import lmdb
import pickle

import numpy as np
import tensorflow as tf

from modeling.utils.box_ops import py_iou

flags.DEFINE_string('split_pkl_file', '',
                    'Pickle file denoting the train/test splits.')
flags.DEFINE_integer('num_val_examples', 5000,
                     'Number of examples for validation.')
flags.DEFINE_string('proposal_coord_pkl_file', '',
                    'Pickle file storing proposal coordinates.')
flags.DEFINE_string('proposal_feature_lmdb_file', '',
                    'LMDB file storing proposal feature.')
flags.DEFINE_string('scene_graph_pkl_file', '',
                    'Pickle file storing scene graphs annotations.')
flags.DEFINE_string('embedding_pkl_file', '',
                    'Path to the file storing entity/predicate embeddings.')
flags.DEFINE_string('output_directory', '', 'Path to the output directory')

FLAGS = flags.FLAGS

_FEATURE_DIMS = 1536

np.random.seed(286)


def _read_proposal_features(image_id, txn):
  """Reads proposal feature from lmdb. 

  Args:
    image_id: Image id.
    txn: Transaction object for readong lmdb file.

  Returns:
    A [num_proposals, _FEATURE_DIMS] nparray.
  """
  proposal_features = np.frombuffer(txn.get(str(image_id).encode('utf8')),
                                    'float32')
  return proposal_features.reshape((-1, _FEATURE_DIMS))


def _create_tf_example(image_id,
                       proposals,
                       proposal_features,
                       scene_graph_triples,
                       scene_graph_graph,
                       dedup_training_split=False):
  """Creates tf example.

  Args:
    image_id: Image id.
    proposals: A [num_proposals, 4] nparray, denoting [x1, y1, x2, y2].
    proposal_features: A [num_proposals, _FEATURE_DIMS] nparray.
    scene_graph_triples: A list of scene graph triples, including the following
      keys: `subject`, `object`, `predicate`, `subject_box`, `object_box`.
    scene_graph_graph: A dict containing graph-like data, including the 
      following keys: `n_node`, `n_edge`, `nodes`, `edges`, `senders`, `receivers`.
    dedup_training_split: If true, run de-duplicate for the training split.

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
          _float_feature_list(proposals[:, 1].tolist()),
      'image/proposal/bbox/xmin':
          _float_feature_list(proposals[:, 0].tolist()),
      'image/proposal/bbox/ymax':
          _float_feature_list(proposals[:, 3].tolist()),
      'image/proposal/bbox/xmax':
          _float_feature_list(proposals[:, 2].tolist()),
      'image/proposal/feature':
          _float_feature_list(proposal_features.flatten().tolist()),
  }

  # Encode ground-truth scene graph triples.
  dedup_set = set()
  subjects, predicates, objects = [], [], []
  subject_boxes, object_boxes = [], []
  for triple in scene_graph_triples:
    if dedup_training_split and (triple['subject'], triple['predicate'],
                                 triple['object']) in dedup_set:
      continue
    dedup_set.add((triple['subject'], triple['predicate'], triple['object']))

    subjects.append(triple['subject'])
    predicates.append(triple['predicate'])
    objects.append(triple['object'])

    subject_boxes.append(triple['subject_box'])
    object_boxes.append(triple['object_box'])

  if scene_graph_triples:
    subject_boxes, object_boxes = (np.stack(subject_boxes),
                                   np.stack(object_boxes))
  else:
    subject_boxes, object_boxes = (np.zeros((0, 4)), np.zeros((0, 4)))

  feature_dict.update({
      # Triplet.
      'scene_graph/n_triple':
          _int64_feature(len(subjects)),
      'scene_graph/subject':
          _string_feature_list(subjects),
      'scene_graph/predicate':
          _string_feature_list(predicates),
      'scene_graph/object':
          _string_feature_list(objects),
      # Subject box.
      'scene_graph/subject/bbox/ymin':
          _float_feature_list(subject_boxes[:, 1].tolist()),
      'scene_graph/subject/bbox/xmin':
          _float_feature_list(subject_boxes[:, 0].tolist()),
      'scene_graph/subject/bbox/ymax':
          _float_feature_list(subject_boxes[:, 3].tolist()),
      'scene_graph/subject/bbox/xmax':
          _float_feature_list(subject_boxes[:, 2].tolist()),
      # Object box.
      'scene_graph/object/bbox/ymin':
          _float_feature_list(object_boxes[:, 1].tolist()),
      'scene_graph/object/bbox/xmin':
          _float_feature_list(object_boxes[:, 0].tolist()),
      'scene_graph/object/bbox/ymax':
          _float_feature_list(object_boxes[:, 3].tolist()),
      'scene_graph/object/bbox/xmax':
          _float_feature_list(object_boxes[:, 2].tolist()),
      # Pseudo graph.
      'scene_pseudo_graph/n_node':
          _int64_feature(scene_graph_graph['n_node']),
      'scene_pseudo_graph/n_edge':
          _int64_feature(scene_graph_graph['n_edge']),
      'scene_pseudo_graph/nodes':
          _string_feature_list(scene_graph_graph['nodes']),
      'scene_pseudo_graph/edges':
          _string_feature_list(scene_graph_graph['edges']),
      'scene_pseudo_graph/senders':
          _int64_feature_list(scene_graph_graph['senders']),
      'scene_pseudo_graph/receivers':
          _int64_feature_list(scene_graph_graph['receivers']),
  })

  tf_example = tf.train.Example(features=tf.train.Features(
      feature=feature_dict))
  return tf_example


def _create_tf_record_from_annotations(image_ids, id_to_proposals,
                                       id_to_scenegraph, txn, tf_record_file,
                                       num_output_parts, dedup_training_split):
  """Creates tf record files.

  Args:
    image_ids: A list of integer denoting the image ids.
    id_to_proposals: A dict mapping from image ids to pre-extracted proposals.
    id_to_scenegraph: A dict mapping from image ids to scene graph annotations.
    txn: Transaction object for readong lmdb file.
    tf_record_file: A tfrecord filename denoting the output file.
  """
  writers = []
  for i in range(num_output_parts):
    filename = tf_record_file + '-%05d-of-%05d' % (i, num_output_parts)
    writers.append(tf.io.TFRecordWriter(filename))

  for i, image_id in enumerate(image_ids):
    if image_id not in id_to_scenegraph:
      continue

    scene_graph_triples = id_to_scenegraph[image_id]['triples']
    scene_graph_graph = id_to_scenegraph[image_id]['graph']

    proposals, proposal_features = (id_to_proposals[image_id],
                                    _read_proposal_features(image_id, txn))
    assert proposals.shape[0] == proposal_features.shape[0]

    tf_example = _create_tf_example(image_id, proposals, proposal_features,
                                    scene_graph_triples, scene_graph_graph,
                                    dedup_training_split)
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


def _read_scene_graph_annotations_keyed_by_image_id(filename, scene_graph_meta):
  """Reads scene graphs keyed by image id.

  Args:
    filename: Path to the scene graph annotations pickle file.
    scene_graph_meta: A meta structure provided by the VSPNET authors,
      it maps from type (an integer) to text labels.

  Returns:
    A dict mapping image id to scene graphs.
  """
  id_to_entity, id_to_predicate = (scene_graph_meta['idx_to_label'],
                                   scene_graph_meta['idx_to_predicate'])
  with tf.io.gfile.GFile(filename, 'rb') as fid:
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

    # Get the set of entities, merge overlapped entities.
    # TODO: NMS instead.
    entity_names, entity_boxes = [], []
    predicate_names = []
    sub_ent_ids, obj_ent_ids = [], []

    triple_annots = []
    for triple in triples:
      try:
        sub, pred, obj = (id_to_entity[str(triple['subject_type'])],
                          id_to_predicate[str(triple['predicate_type'])],
                          id_to_entity[str(triple['object_type'])])
      except Exception as ex:
        sub, pred, obj = (id_to_entity[triple['subject_type']],
                          id_to_predicate[triple['predicate_type']],
                          id_to_entity[triple['object_type']])

      triple_annots.append({
          'predicate': pred,
          'subject': sub,
          'subject_box': triple['subject_box'],
          'object': obj,
          'object_box': triple['object_box'],
      })

      sub_box = triple['subject_box']
      obj_box = triple['object_box']
      sub_ent_ids.append(
          _insert_entity(entity_names, entity_boxes, sub_box, sub))
      obj_ent_ids.append(
          _insert_entity(entity_names, entity_boxes, obj_box, obj))
      predicate_names.append(pred)

    data[image_id] = {
        'triples': triple_annots,
        'graph': {
            'n_node': len(entity_names),
            'n_edge': len(triples),
            'nodes': entity_names,
            'edges': predicate_names,
            'senders': sub_ent_ids,
            'receivers': obj_ent_ids,
        }
    }

  logging.info('In total: %i objects, %i relationships, skipped %i.',
               count_entity, count_triples, skipped)
  logging.info('In average: %.2lf objects, %.2lf relationships.',
               count_entity / count, count_triples / count)
  return data


def main(_):
  assert FLAGS.split_pkl_file, '`split_pkl_file` missing.'
  assert FLAGS.proposal_coord_pkl_file, '`proposal_coord_pkl_file` missing.'
  assert FLAGS.proposal_feature_lmdb_file, '`proposal_feature_lmdb` missing.'
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

  # Load proposals.
  with tf.io.gfile.GFile(FLAGS.proposal_coord_pkl_file, 'rb') as fid:
    id_to_proposals = pickle.load(fid)
  logging.info('Proposals: #=%i', len(id_to_proposals))

  # Load scene graphs.
  id_to_scenegraph = _read_scene_graph_annotations_keyed_by_image_id(
      FLAGS.scene_graph_pkl_file, scene_graph_meta)

  # Generate tfrecord files.
  with lmdb.open(flags.FLAGS.proposal_feature_lmdb_file,
                 map_size=1e12,
                 readonly=True,
                 lock=False) as env:
    with env.begin() as txn:
      _create_tf_record_from_annotations(
          train_ids, id_to_proposals, id_to_scenegraph, txn,
          os.path.join(output_directory, 'train.tfrecord'), 10, False)
      _create_tf_record_from_annotations(
          val_ids, id_to_proposals, id_to_scenegraph, txn,
          os.path.join(output_directory, 'val.tfrecord'), 1, False)
      _create_tf_record_from_annotations(
          test_ids, id_to_proposals, id_to_scenegraph, txn,
          os.path.join(output_directory, 'test.tfrecord'), 5, False)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
