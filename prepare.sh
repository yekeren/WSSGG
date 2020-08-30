#!/bin/sh

set -o errexit
set -o nounset
set -x

git clone "https://github.com/tensorflow/models.git" "tensorflow_models"
ln -s "tensorflow_models/research/object_detection" .
