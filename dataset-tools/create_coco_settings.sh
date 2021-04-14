#!/bin/sh

set -o errexit
set -o nounset

download() {
  local -r dir=$1
  local -r filename=$2
  local -r output_dir=$3

  if [ ! -f "${output_dir}/${filename}" ]; then
    wget -O "${output_dir}/${filename}" "${dir}/${filename}"
  fi
}

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DATA_DIR VG_DIR"
  exit 1
fi

data_dir=$1
vg_dir=$2

mkdir -p "${data_dir}"

################################################################################
# Download the COCO images and annotations.
################################################################################
image_dir="http://images.cocodataset.org/zips"
train_images="train2017.zip"
val_images="val2017.zip"

download "${image_dir}" "${train_images}" "${data_dir}"
download "${image_dir}" "${val_images}" "${data_dir}"

annotation_dir="http://images.cocodataset.org/annotations"
trainval_annotations="annotations_trainval2017.zip"

download "${annotation_dir}" "${trainval_annotations}" "${data_dir}"
unzip -n "${data_dir}/${trainval_annotations}" -d "${data_dir}"

##########################################################
# Extract the text graphs and gather the open vocabulary.
##########################################################
if [ ! -f "${data_dir}/annotations/scenegraphs_train2017.json" ]; then
  python "dataset-tools/create_coco_text_graphs.py" \
    --logtostderr \
    --caption_annotations_file="${data_dir}/annotations/captions_train2017.json" \
    --scenegraph_annotations_file="${data_dir}/annotations/scenegraphs_train2017.json"
fi

if [ ! -f "${data_dir}/vocab-coco.txt" ]; then
  python "dataset-tools/create_coco_vocabulary.py" \
    --logtostderr \
    --scenegraph_annotations_file="${data_dir}/annotations/scenegraphs_train2017.json" \
    --output_file="${data_dir}/vocab-coco.txt"
fi

##########################################################
# Extract proposals using Faster-RCNN.
##########################################################
if [ ! -d "${data_dir}/frcnn_proposals/" ]; then
  python "dataset-tools/create_coco_frcnn_proposals.py" \
    --logtostderr \
    --train_image_file="${data_dir}/train2017.zip" \
    --output_directory="${data_dir}/frcnn_proposals" \
    --detection_pipeline_proto="zoo/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28/pipeline.config" \
    --detection_checkpoint_file="zoo/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28/model.ckpt"
fi

##########################################################
# Create the tfrecord files.
##########################################################
if [ ! -d "${data_dir}/tfrecords/coco-cap-graph-zareian" ]; then
  python "dataset-tools/create_coco_tf_record.py" \
    --logtostderr \
    --split_pkl_file="${vg_dir}/metadata/VG/hanwang/split.pkl" \
    --vg_meta_file="${vg_dir}/image_data.json" \
    --scenegraph_annotations_file="${data_dir}/annotations/scenegraphs_train2017.json" \
    --proposal_npz_directory="${data_dir}/frcnn_proposals" \
    --output_directory="${data_dir}/tfrecords/coco-cap-graph-zareian"
fi

if [ ! -d "${data_dir}/tfrecords/coco-cap-graph-xu" ]; then
  python "dataset-tools/create_coco_tf_record.py" \
    --logtostderr \
    --split_pkl_file="${vg_dir}/metadata/VG/stanford/split_stanford.pkl" \
    --vg_meta_file="${vg_dir}/image_data.json" \
    --scenegraph_annotations_file="${data_dir}/annotations/scenegraphs_train2017.json" \
    --proposal_npz_directory="${data_dir}/frcnn_proposals" \
    --output_directory="${data_dir}/tfrecords/coco-cap-graph-xu"
fi
