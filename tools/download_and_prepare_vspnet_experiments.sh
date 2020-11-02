#!/bin/sh

# This script helps to setup the settings to compare to the `Zareian et al., CVPR2020`.
#   - The `stanford setting` should be from `Xu et al., CVPR2017`
#   - The `hanwang setting` should be from `Zhang et al., ICCV2017` - .
# Please also refer to `Weakly Supervised Visual Semantic Parsing` (https://github.com/alirezazareian/vspnet).

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

# Download the `metadata` from `https://github.com/alirezazareian/vspnet`.
if [ ! -d "${data_dir}/metadata" ]; then
  download \
    "https://www.dropbox.com/sh/oa8u7qolfpf1op0/AACivQp5RmtmykbqmWeupOZEa?dl=0" \
    "${data_dir}" \
    "metadata.zip"

  unzip \
    "${data_dir}/metadata.zip" \
    -x "/" -d "${data_dir}/metadata" 
fi

# Download the `data` from `https://github.com/alirezazareian/vspnet`.
mkdir -p "${data_dir}/data/VG/xfeat_proposals"

if [ ! -d "${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop" ]; then
  download \
    "https://www.dropbox.com/sh/eb60553z4md36x2/AAC0wBVD7yQxvSaChCd5xsg9a/VG/xfeat_proposals/iresnet_oi_lowprop?dl=0" \
    "${data_dir}/data/VG/xfeat_proposals/" \
    "iresnet_oi_lowprop.zip"
  
  unzip \
    "${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop.zip" \
    -x "/" -d "${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop"
fi

if [ ! -d "${data_dir}/data/VG/xfeat_proposals/iresnet_oi" ]; then
  mkdir -p "${data_dir}/data/VG/xfeat_proposals/iresnet_oi/xfeat_proposals_inresnet_float32_300x1536.lmdb"
  cd "${data_dir}/data/VG/xfeat_proposals/iresnet_oi/"
  download \
    "https://www.dropbox.com/sh/eb60553z4md36x2/AAD-SfzJD9aXzZHurJTjuB8Ua/VG/xfeat_proposals/iresnet_oi/coords.pkl?dl=0" \
    "." \
    "coords.pkl"
  download \
    "https://www.dropbox.com/sh/eb60553z4md36x2/AABcFReJUxs6NSp4WHyaGmJca/VG/xfeat_proposals/iresnet_oi/xfeat_proposals_inresnet_float32_300x1536.lmdb/lock.mdb?dl=0" \
    "xfeat_proposals_inresnet_float32_300x1536.lmdb/" \
    "lock.mdb"
  download \
    "https://www.dropbox.com/sh/eb60553z4md36x2/AAANrJvebu-iDrhCWRFu-ELGa/VG/xfeat_proposals/iresnet_oi/xfeat_proposals_inresnet_float32_300x1536.lmdb/data.mdb?dl=0" \
    "xfeat_proposals_inresnet_float32_300x1536.lmdb/" \
    "data.mdb"
  cd -
fi

# Download visual genome images.

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

# ##########################################################
# # Using triplets: 20 proposal setting.
# ##########################################################
# 
# # Prepare the tfrecord files for hanwang setting.
# 
# python "tools/create_vspnet_vg_tf_record.py" \
#   --split_pkl_file="${data_dir}/metadata/VG/hanwang/split.pkl" \
#   --proposal_coord_pkl_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/coords.pkl" \
#   --proposal_feature_lmdb_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/feats_float32_20x1536.lmdb" \
#   --scene_graph_pkl_file="${data_dir}/metadata/VG/hanwang/sg.pkl" \
#   --embedding_pkl_file="${data_dir}/metadata/VG/hanwang/class_embs.pkl" \
#   --output_directory="${data_dir}/tfrecords/hanwang-dedup"
# 
# # Prepare the tfrecord files for stanford setting.
# 
# python "tools/create_vspnet_vg_tf_record.py" \
#   --split_pkl_file="${data_dir}/metadata/VG/stanford/split_stanford.pkl" \
#   --proposal_coord_pkl_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/coords.pkl" \
#   --proposal_feature_lmdb_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/feats_float32_20x1536.lmdb" \
#   --scene_graph_pkl_file="${data_dir}/metadata/VG/stanford/sg_stanford_with_duplicates.pkl" \
#   --embedding_pkl_file="${data_dir}/metadata/VG/stanford/word_emb_stanford_2.pkl" \
#   --output_directory="${data_dir}/tfrecords/stanford-dedup"

# ##########################################################
# # Using triplets: 300 proposal setting.
# ##########################################################
# 
# # Prepare the tfrecord files for hanwang300 setting.
# 
# python "tools/create_vspnet_vg_tf_record.py" \
#   --split_pkl_file="${data_dir}/metadata/VG/hanwang/split.pkl" \
#   --proposal_coord_pkl_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi/coords.pkl" \
#   --proposal_feature_lmdb_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi/xfeat_proposals_inresnet_float32_300x1536.lmdb" \
#   --scene_graph_pkl_file="${data_dir}/metadata/VG/hanwang/sg.pkl" \
#   --embedding_pkl_file="${data_dir}/metadata/VG/hanwang/class_embs.pkl" \
#   --output_directory="${data_dir}/tfrecords/hanwang300-dedup"
# 
# # Prepare the tfrecord files for stanford300 setting.
# 
# python "tools/create_vspnet_vg_tf_record.py" \
#   --split_pkl_file="${data_dir}/metadata/VG/stanford/split_stanford.pkl" \
#   --proposal_coord_pkl_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi/coords.pkl" \
#   --proposal_feature_lmdb_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi/xfeat_proposals_inresnet_float32_300x1536.lmdb" \
#   --scene_graph_pkl_file="${data_dir}/metadata/VG/stanford/sg_stanford_with_duplicates.pkl" \
#   --embedding_pkl_file="${data_dir}/metadata/VG/stanford/word_emb_stanford_2.pkl" \
#   --output_directory="${data_dir}/tfrecords/stanford300-dedup"

##########################################################
# Using the ideal Graphs: 20 proposal setting.
##########################################################

# # Prepare the tfrecord files for hanwang setting.
# 
# python "tools/create_vspnet_vg_graph_tf_record.py" \
#   --split_pkl_file="${data_dir}/metadata/VG/hanwang/split.pkl" \
#   --proposal_coord_pkl_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/coords.pkl" \
#   --proposal_feature_lmdb_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/feats_float32_20x1536.lmdb" \
#   --scene_graph_pkl_file="${data_dir}/metadata/VG/hanwang/sg.pkl" \
#   --embedding_pkl_file="${data_dir}/metadata/VG/hanwang/class_embs.pkl" \
#   --output_directory="${data_dir}/tfrecords/graph-hanwang"

##########################################################
# Using the caption Graphs: 20 proposal setting.
##########################################################

# Prepare the tfrecord files for hanwang setting.

python "tools/create_vspnet_vg_caption_graph_tf_record.py" \
  --split_pkl_file="${data_dir}/metadata/VG/hanwang/split.pkl" \
  --proposal_coord_pkl_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/coords.pkl" \
  --proposal_feature_lmdb_file="${data_dir}/data/VG/xfeat_proposals/iresnet_oi_lowprop/feats_float32_20x1536.lmdb" \
  --scene_graph_pkl_file="${data_dir}/metadata/VG/hanwang/sg.pkl" \
  --caption_graph_json_file="${data_dir}/text_graphs/text_graphs.jsonl" \
  --embedding_pkl_file="${data_dir}/metadata/VG/hanwang/class_embs.pkl" \
  --output_directory="${data_dir}/tfrecords/caption-graph-hanwang"



