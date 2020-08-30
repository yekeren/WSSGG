#!/bin/sh

set -o errexit
set -o nounset
set -x

protoc protos/*.proto --python_out=.
