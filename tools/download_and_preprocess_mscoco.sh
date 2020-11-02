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
if [ ! -f "${raw_data}/annotations/scenegraphs_val2017.json" ] || [ ! -f "${raw_data}/annotations/scenegraphs_val2017.json" ]; then
  python "tools/create_coco_scenegraphs_from_captions.py" \
    --logtostderr \
    --number_of_threads="20" \
    --train_caption_annotations_file="${raw_data}/annotations/captions_train2017.json" \
    --val_caption_annotations_file="${raw_data}/annotations/captions_val2017.json" \
    --output_directory="${raw_data}/annotations"
fi

# if [ ! -d "${raw_data}/ss_proposals/" ]; then
#   python "tools/create_coco_ss_proposals.py" \
#     --logtostderr \
#     --train_image_file="${raw_data}/train2017.zip" \
#     --val_image_file="${raw_data}/val2017.zip" \
#     --test_image_file="${raw_data}/test2017.zip" \
#     --output_directory="${raw_data}/proposals"
# fi

exit 0

python "tools/create_coco_tf_record.py" \
  --logtostderr \
  --train_image_file="${raw_data}/train2017.zip" \
  --val_image_file="${raw_data}/val2017.zip" \
  --train_scenegraph_annotations_file="${raw_data}/annotations/scenegraphs_train2017.json" \
  --val_scenegraph_annotations_file="${raw_data}/annotations/scenegraphs_val2017.json" \
  --proposal_nparray_directory="${raw_data}/proposals" \
  --output_directory="${raw_data}/tfrecords"

exit 0
# 
# 
# # Preview the extracted scene graphs from captions.
# # Or, you can also use jupyter and `tools/show_coco_scenegraphs.ipynb`.
# python "tools/show_coco_scenegraphs.py" \
#   --logtostderr \
#   --scenegraph_annotations_file="${raw_data}/annotations/scenegraphs_val2017.json"
