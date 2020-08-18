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
  echo "Usage: $0 DATADIR"
  exit 1
fi

# Create the directory if it does not exist.
data_dir=$1
mkdir -p "${data_dir}"
data_dir_full="`readlink -f ${data_dir}`"

# Download Stanford corenlp 3.6.0.
download "https://nlp.stanford.edu/software" "stanford-corenlp-full-2015-12-09.zip" "${data_dir}"
unzip "${data_dir}/stanford-corenlp-full-2015-12-09.zip" -d "${data_dir}"
cd "${data_dir}/stanford-corenlp-full-2015-12-09"

# Download scene graph parser. 
download "https://nlp.stanford.edu/projects/scenegraph" "scenegraph-1.0.jar" "."

# Download json-simple jar file.
download "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/json-simple" "json-simple-1.1.1.jar" "."


# Show information to setup java CLASSPATH.
cd ..
echo -e "\n\n\n"
echo "Commands to test the Stanford Scene Graph Parser:"

echo "cd \"${data_dir}\""
echo "javac -cp \"stanford-corenlp-full-2015-12-09/*:.\" \"SceneGraphDemo.java\""
echo "java -mx2g -cp \"stanford-corenlp-full-2015-12-09/*:.\" \"SceneGraphDemo\""
