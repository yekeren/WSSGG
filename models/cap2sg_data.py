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


class GroundingTuple(object):

  # Proposal id to be associated with each text entity, a [batch, max_n_entity] in tensor, each value is in the range [0, max_n_proposal).
  entity_proposal_id = None

  # Proposal box to be associated with each text entity, a [batch, max_n_entity, 4] float tensor.
  entity_proposal_box = None

  # Proposal score to be associated with each text entity, a [batch, max_n_entity] float tensor.
  entity_proposal_score = None

  # Proposal feature to be associated with each text entity, a [batch, max_n_entity, feature_dims] float tensor.
  entity_proposal_feature = None


class DetectionTuple(object):

  # Number of detections, a [batch] int tensor.
  valid_detections = None

  # Final proposals that are chosen, a [batch, max_n_detection] int tensor.
  nmsed_proposal_id = None

  # Detection boxes, a [batch, max_n_detection, 4] float tensor.
  nmsed_boxes = None

  # Detection scores, a [batch, max_n_detection] float tensor.
  nmsed_scores = None

  # Detection classes, a [batch, max_n_detection] string tensor.
  nmsed_classes = None

  # Detection attribute scores, a [batch, max_n_detetion] float tensor.
  nmsed_attribute_scores = None

  # Detection attribute classes, a [batch, max_n_detetion] string tensor.
  nmsed_attribute_classes = None

  # Detection region features, a [batch, max_n_detection, feature_dims] tensor.
  nmsed_features = None


class RelationTuple(object):

  # Number of relations, a [batch] int tensor.
  num_relations = None

  # Log probability of the (subject, relation, object), a [batch, max_n_relation] float tensor.
  log_prob = None

  # Relation score, a [batch, max_n_relation] float tensor.
  relation_score = None

  # Relation class, a [batch, max_n_relation] string tensor.
  relation_class = None

  # Proposal id associated to the subject, a [batch, max_n_relation] int tensor, each value is in the range [0, max_n_proposal).
  subject_proposal = None

  # Subject score, a [batch, max_n_relation] float tensor.
  subject_score = None

  # Subject class, a [batch, max_n_relation] string tensor.
  subject_class = None

  # Proposal id associated to the object, a [batch, max_n_relation] int tensor, each value is in the range [0, max_n_proposal).
  object_proposal = None

  # Object score, a [batch, max_n_relation] float tensor.
  object_score = None

  # Object class, a [batch, max_n_relation] string tensor.
  object_class = None


class DataTuple(object):

  ####################################################
  # Objects created by preprocess.initialize.
  ####################################################

  # A callable converting tokens to integer ids.
  token2id_func = None

  # A callable converting integer ids to tokens.
  id2token_func = None

  # Length of the vocabulary.
  vocab_size = None

  # Embedding dimensions.
  dims = None

  # Word embeddings, a [vocab_size, dims] float tensor.
  embeddings = None

  # A callable converting token ids to embedding vectors.
  embedding_func = None

  # Entity bias, a [vocab_size] float tensor.
  bias_entity = None

  # Attribute bias, a [vocab_size] float tensor.
  bias_attribute = None

  # Relation bias, a [vocab_size] float tensor.
  bias_relation = None

  ####################################################
  # Objects parsed from TF record files.
  ####################################################

  # Batch size.
  batch = None

  # Number of proposals. a [batch] int tensor.
  n_proposal = None

  # Maximum proposals in the batch, a scalar int tensor.
  max_n_proposal = None

  # Proposal masks, a [batch, max_n_proposal] float tensor, each value is in {0, 1}, denoting the validity.
  proposal_masks = None

  # Proposal boxes, a [batch, max_n_proposal, 4] float tensor.
  proposals = None

  # Proposal features, a [batch, max_n_proposal, feature_dims] float tensor.
  proposal_features = None

  # Number of text entities, a [batch] int tensor.
  n_entity = None

  # Maximum entities in the batch, a scalar int tensor.
  max_n_entity = None

  # Entity masks, a [batch, max_n_entity] float tensor, each value is in {0, 1}, denoting the validity.
  entity_masks = None

  # Text entity ids, a [batch, max_n_entity] int tensor, each value is in the range [0, vocab_size).
  entity_ids = None

  # Text entity embeddings, a [batch, max_n_entity, dims] float tensor.
  entity_embs = None

  # Number of attributes per each text entity, a [batch, max_n_entity] int tensor.
  per_ent_n_att = None

  # Per-entity attribute ids, a [batch, max_n_entity, max_per_ent_n_att] int tensor, each value is in the range [0, vocab_size).
  per_ent_att_ids = None

  # Per-entity attribute embeddings, a [batch, max_n_entity, max_per_ent_n_att, dims] float tensor.
  per_ent_att_embs = None

  # Image-level one-hot text entity labels, a [batch, max_n_entity, vocab_size] tensor, only one value in the last dimension is 1.
  entity_image_labels = None

  # Image-level multi-hot text attribute labels, a [batch, max_n_entity, vocab_size] tensor, multiple values in the last dimension may be 1.
  attribute_image_labels = None

  # Number of text relations, a [batch] int tensor.
  n_relation = None

  # Maximum relations in the batch, a scalar int tensor.
  max_n_relation = None

  # Relation masks, a [batch, max_n_relation] float tensor, each value is in {0, 1}, denoting the validity.
  relation_masks = None

  # Text relation ids, a [batch, max_n_relation] int tensor, each value is in the range [0, vocab_size).
  relation_ids = None

  # Text relation embeddings, a [batch, max_n_relation, dims] float tensor.
  relation_embs = None

  # Index of the subject entity, referring the entity in the entity_ids, a [batch, max_n_relation] int tensor, each value is in the range [0, max_n_entity].
  relation_senders = None

  # Index of the object entity, referring the entity in the entity_ids, a [batch, max_n_relation] int tensor, each value is in the range [0, max_n_entity].
  relation_receivers = None

  ####################################################
  # Objects created by grounding.ground_entities.
  ####################################################

  # Image-level text entity prediction, a [batch, max_n_entity, vocab_size] tensor.
  entity_image_logits = None

  # Image-level text attribute prediction, a [batch, max_n_entity, vocab_size] tensor.
  attribute_image_logits = None

  # Grounding results.
  grounding = GroundingTuple()

  ####################################################
  # Objects created by detection.detect_entities.
  ####################################################

  # Entity detection logits, a [batch, max_n_proposal, vocab_size] float tensor.
  detecton_instance_logits = None

  # Normalized entity detection scores, a [batch, max_n_proposal, vocab_size] float tensor.
  detecton_instance_scores = None

  # Entity detection labels, a [batch, max_n_proposal, vocab_size] float tensor.
  detecton_instance_labels = None

  # Attribute detection logits, a [batch, max_n_proposal, vocab_size] float tensor.
  attribute_instance_logits = None

  # Normalized attribute detection scores, a [batch, max_n_proposal, vocab_size] float tensor.
  attribute_instance_scores = None

  # Attribute detection labels, a [batch, max_n_proposal, vocab_size] float tensor.
  attribute_instance_labels = None

  # Detection results.
  detection = DetectionTuple()

  # Grounding results.
  refined_grounding = GroundingTuple()

  ####################################################
  # Objects created by relation.detect_relations.
  ####################################################

  # Subject boxes, a [batch, max_n_relation, 4] float tensor.
  subject_boxes = None

  # Subject labels, a [batch, max_n_relation] float tensor, each value is in the range [0, vocab_size).
  subject_labels = None

  # Object boxes, a [batch, max_n_relation, 4] float tensor.
  object_boxes = None

  # Object labels , a [batch, max_n_relation] float tensor, each value is in the range [0, vocab_size).
  object_labels = None

  # Predicate labels, a [batch, max_n_relation] float tensor, each value is in the range [0, vocab_size).
  predicate_labels = None

  # Sequence prediction of subject, a [batch, max_n_relation, vocab_size] float tensor.
  subject_logits = None

  # Sequence prediction of object, a [batch, max_n_relation, vocab_size] float tensor.
  object_logits = None

  # Sequence prediction of predicate, a [batch, max_n_relation, vocab_size] float tensor.
  predicate_logits = None

  # # Relation detection logits, a [batch, max_n_proposal, max_n_proposal, vocab_size] float tensor.
  # relation_instance_logits = None

  # # Normalized relation detection scores, a [batch, max_n_proposal, max_n_proposal, vocab_size] float tensor.
  # relation_instance_scores = None

  # # Relation detection labels, a [batch, max_n_proposal, max_n_proposal, vocab_size] float tensor.
  # relation_instance_labels = None

  # Relation results.
  relation = RelationTuple()

  refined_relation = RelationTuple()
