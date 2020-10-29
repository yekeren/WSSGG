#!/bin/sh

set -o errexit
set -o nounset
set -x

download() {
  local -r dir=$1
  local -r filename=$2
  local -r output_dir=$3

  if [ ! -f "${output_dir}/${filename}" ]; then
    wget -O "${output_dir}/${filename}" "${dir}/${filename}"
  fi
}

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 DATADIR"
  exit 1
fi

raw_data=$1
mkdir -p "${raw_data}"

# Download visual genome annotations. 
download "https://visualgenome.org/static/data/dataset" "region_descriptions.json.zip" "${raw_data}"
unzip "${raw_data}/region_descriptions.json.zip" -d "${raw_data}"

if [ ! -f "${raw_data}/scenegraphs.json" ]; then
  python "tools/create_vg_scenegraphs_from_captions.py" \
    --logtostderr \
    --number_of_threads="20" \
    --caption_annotations_file="${raw_data}/region_descriptions.json" \
    --output_file_path="${raw_data}/text_graphs.json"
fi
