#!/bin/sh

set -o errexit
set -o nounset
set -x

download() {
  local -r url=$1
  local -r data_dir=$2
  local -r data_file=$3

  if [ ! -f "${data_dir}/${data_file}" ]; then
    wget -O "${data_dir}/${data_file}" "${url}" 
  fi
}

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 DATADIR"
  exit 1
fi

data_dir=$1
mkdir -p "${data_dir}"

################################################################################
# Download the `metadata` from `https://github.com/alirezazareian/vspnet`.
################################################################################
if [ ! -d "${data_dir}/metadata" ]; then
  download \
    "https://www.dropbox.com/sh/oa8u7qolfpf1op0/AACivQp5RmtmykbqmWeupOZEa?dl=0" \
    "${data_dir}" \
    "metadata.zip"

  unzip \
    "${data_dir}/metadata.zip" \
    -x "/" -d "${data_dir}/metadata" 
fi

################################################################################
# Download visual genome images.
################################################################################
if [ ! -d "${data_dir}/images" ]; then
  download \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" \
    "${data_dir}" "image.zip"
  download \
    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" \
    "${data_dir}" "image2.zip"
  
  unzip -q "${data_dir}/image.zip" -d "${data_dir}/images"
  unzip -q "${data_dir}/image2.zip" -d "${data_dir}/images"

  find "${data_dir}/images/VG_100K/" -name "*.jpg" -exec mv {} "${data_dir}/images" \;
  find "${data_dir}/images/VG_100K_2/" -name "*.jpg" -exec mv {} "${data_dir}/images" \;
fi

################################################################################
# Extract proposals using Faster-RCNN.
################################################################################
if [ ! -d "${data_dir}/frcnn_proposals/" ]; then
  python "dataset-tools/create_vg_frcnn_proposals.py" \
    --logtostderr \
    --image_dir="${data_dir}/images" \
    --output_directory="${data_dir}/frcnn_proposals" \
    --detection_pipeline_proto="zoo/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28/pipeline.config" \
    --detection_checkpoint_file="zoo/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28/model.ckpt"
fi

exit 0


##########################################################
# Create the tfrecord files.
##########################################################

if [ ! -f "${data_dir}/scenegraphs.json" ]; then
  python "dataset-tools/create_visual_genome_text_graphs.py" \
    --logtostderr \
    --caption_annotations_file="${data_dir}/region_descriptions.json" \
    --scenegraph_annotations_file="${data_dir}/scenegraphs.json"
fi

python "dataset-tools/create_visual_genome_vocabulary.py" \
  --scenegraph_annotations_file="${data_dir}/scenegraphs.json" \
  --output_file="${data_dir}/vocab-vg.txt"

exit 0

python "dataset-tools/create_visual_genome_gt_graph_tf_record.py" \
  --split_pkl_file="${data_dir}/metadata/VG/hanwang/split.pkl" \
  --proposal_feature_dir="${data_dir}/frcnn_proposals_50_iou0.7/" \
  --scene_graph_pkl_file="${data_dir}/metadata/VG/hanwang/sg.pkl" \
  --embedding_pkl_file="${data_dir}/metadata/VG/hanwang/class_embs.pkl" \
  --output_directory="${data_dir}/tfrecords/caption-graph-hanwang-v3"

python "dataset-tools/create_visual_genome_gt_graph_tf_record.py" \
  --split_pkl_file="${data_dir}/metadata/VG/stanford/split_stanford.pkl" \
  --proposal_feature_dir="${data_dir}/frcnn_proposals_50_iou0.7/" \
  --scene_graph_pkl_file="${data_dir}/metadata/VG/stanford/sg_stanford_with_duplicates.pkl" \
  --embedding_pkl_file="${data_dir}/metadata/VG/stanford/word_emb_stanford_2.pkl" \
  --output_directory="${data_dir}/tfrecords/caption-graph-stanford-v3"

python "dataset-tools/create_visual_genome_cap_graph_tf_record.py" \
  --text_graphs_json_file="${data_dir}/scenegraphs.json" \
  --split_pkl_file="${data_dir}/metadata/VG/hanwang/split.pkl" \
  --proposal_feature_dir="${data_dir}/frcnn_proposals_50_iou0.7/" \
  --scene_graph_pkl_file="${data_dir}/metadata/VG/hanwang/sg.pkl" \
  --embedding_pkl_file="${data_dir}/metadata/VG/hanwang/class_embs.pkl" \
  --output_directory="${data_dir}/tfrecords/cap-graph-hanwang"

python "dataset-tools/create_visual_genome_cap_graph_tf_record.py" \
  --text_graphs_json_file="${data_dir}/scenegraphs.json" \
  --split_pkl_file="${data_dir}/metadata/VG/stanford/split_stanford.pkl" \
  --proposal_feature_dir="${data_dir}/frcnn_proposals_50_iou0.7/" \
  --scene_graph_pkl_file="${data_dir}/metadata/VG/stanford/sg_stanford_with_duplicates.pkl" \
  --embedding_pkl_file="${data_dir}/metadata/VG/stanford/word_emb_stanford_2.pkl" \
  --output_directory="${data_dir}/tfrecords/cap-graph-stanford"
