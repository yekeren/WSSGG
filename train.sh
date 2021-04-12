#!/bin/bash

set -o errexit
set -o nounset
set -x

export PYTHONPATH="`pwd`"
export PYTHONPATH="`pwd`/tensorflow_models/research:$PYTHONPATH"
export PYTHONPATH="`pwd`/tensorflow_models/research/slim:$PYTHONPATH"

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 CONFIG_PATH MODEL_DIR"
  exit 1
fi

mkdir -p log

HOST_NAME="127.0.0.1"
PORT_BASE=4300

CONFIG_PATH=$1
MODEL_DIR=$2

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Config file ${CONFIG_PATH} does not exist."
  exit 1
fi

PS="${HOST_NAME}:$((PORT_BASE+0))"
CHIEF="${HOST_NAME}:$((PORT_BASE+1))"
WORKER0="${HOST_NAME}:$((PORT_BASE+2))"
WORKER1="${HOST_NAME}:$((PORT_BASE+3))"
WORKER2="${HOST_NAME}:$((PORT_BASE+4))"

server_list=(${PS} ${CHIEF} ${WORKER0} ${WORKER1} ${WORKER2})
CLUSTER="{\"chief\": [\"${CHIEF}\"], \"ps\": [\"${PS}\"], \"worker\": [\"${WORKER0}\", \"${WORKER1}\", \"${WORKER2}\"]}"

# Trainers.
declare -A type_dict=(
  [${PS}]="ps"
  [${CHIEF}]="chief"
  [${WORKER0}]="worker"
  [${WORKER1}]="worker"
  [${WORKER2}]="worker"
)
declare -A index_dict=(
  [${PS}]="0"
  [${CHIEF}]="0"
  [${WORKER0}]="0"
  [${WORKER1}]="1"
  [${WORKER2}]="2"
)
declare -A gpu_dict=(
  [${PS}]=""
  [${CHIEF}]="1"
  [${WORKER0}]="2"
  [${WORKER1}]="3"
  [${WORKER2}]="4"
)

for server in ${server_list[@]}; do
  gpu=${gpu_dict[${server}]}
  type=${type_dict[${server}]}
  index=${index_dict[${server}]}

  export CUDA_VISIBLE_DEVICES="${gpu}"
  export TF_CONFIG="{\"cluster\": ${CLUSTER}, \"task\": {\"type\": \"${type}\", \"index\": ${index}}}"  

  python "modeling/trainer_main.py" \
    --alsologtostderr \
    --type="${type}${index}" \
    --pipeline_proto=${CONFIG_PATH} \
    --model_dir="${MODEL_DIR}" \
    >> "log/run.${type}${index}.log" 2>&1 &
  sleep 1
done

# Evaluator.
export CUDA_VISIBLE_DEVICES=0
type="evaluator"
index=0
export TF_CONFIG="{\"cluster\": ${CLUSTER}, \"task\": {\"type\": \"${type}\", \"index\": ${index}}}"  
python "modeling/trainer_main.py" \
  --alsologtostderr \
  --type="${type}${index}" \
  --pipeline_proto=${CONFIG_PATH} \
  --model_dir="${MODEL_DIR}" \
  >> "log/run.${type}${index}.log" 2>&1 &

