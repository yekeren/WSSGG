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

# Create the directory if it does not exist.
data_dir=$1
mkdir -p "${data_dir}"

# Download Stanford corenlp 3.6.0.
download "https://nlp.stanford.edu/software" "stanford-corenlp-full-2015-12-09.zip" "${data_dir}"
unzip "${data_dir}/stanford-corenlp-full-2015-12-09.zip" -d "${data_dir}"
cd "${data_dir}/stanford-corenlp-full-2015-12-09"

# Download scene graph parser. 
download "https://nlp.stanford.edu/projects/scenegraph" "scenegraph-1.0.jar" "."

# Download json-simple jar file.
download "http://json-simple.googlecode.com/files" "json-simple-1.1.1.jar" "."
