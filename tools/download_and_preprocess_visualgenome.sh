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

data_dir=$1
mkdir -p "${data_dir}"

# Download visual genome annotations. 

download "https://visualgenome.org/static/data/dataset" "region_descriptions.json.zip" "${data_dir}"

unzip "${data_dir}/region_descriptions.json.zip" -d "${data_dir}"
