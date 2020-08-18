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

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 DIRECTORY"
  exit 1
fi

raw_data=$1

mkdir -p "${raw_data}"

# Download MSCOCO images.

image_dir="http://images.cocodataset.org/zips"
train_images="train2017.zip"
val_images="val2017.zip"
test_images="test2017.zip"

download "${image_dir}" "${train_images}" "${raw_data}"
download "${image_dir}" "${val_images}" "${raw_data}"
download "${image_dir}" "${test_images}" "${raw_data}"

# Download MSCOCO annotations.

annotation_dir="http://images.cocodataset.org/annotations"
trainval_annotations="annotations_trainval2017.zip"
testdev_annotations="image_info_test2017.zip"

download "${annotation_dir}" "${trainval_annotations}" "${raw_data}"
download "${annotation_dir}" "${testdev_annotations}" "${raw_data}"
unzip -n "${raw_data}/${trainval_annotations}" -d "${raw_data}"
unzip -n "${raw_data}/${testdev_annotations}" -d "${raw_data}"

# Extract scene graphs from captions.

python "tools/create_coco_scenegraphs_from_captions.py" \
  --logtostderr \
  --train_caption_annotations_file="${raw_data}/annotations/captions_train2017.json" \
  --val_caption_annotations_file="${raw_data}/annotations/captions_val2017.json" \
  --output_directory="${raw_data}/annotations"
