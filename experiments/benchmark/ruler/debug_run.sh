#!/bin/bash
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# example
# 
# bash experiments/ruler/debug_run.sh model/Llama-3.1-8B-Instruct/ sparse_llm ruler_result vertical_slash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
RULER_PATH=$(dirname $0)
python -c "import nltk; nltk.download('punkt')"

SEQ_LENGTHS=(
    4096
    # 8192
    # 16384
    # 32768
    # 65536
    # 131072
)

TASKS=(
    "niah_single_1"
    "qa_1"
)

# Experiment Setup
NUM_SAMPLES=25
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# The model
MODEL_NAME=$1
BENCHMARK="synthetic"
MODEL_TEMPLATE_TYPE="base"
MODEL_FRAMEWORK=$2

# Gpu and output path
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR=$3 # the path that stores generated task samples and model predictions.

# MInference
STARTING_LAYER=-1
KV_CACHE_CPU="false"
USE_SNAPKV="false"
TRUST_REMOTE_CODE="true"

if [ "${MODEL_FRAMEWORK}" == "minference" ]; then
    MINFERENCE_PARAMS="--starting_layer ${STARTING_LAYER}"

    if [ -n "${CONFIG_PATH}" ]; then
        MINFERENCE_PARAMS="${MINFERENCE_PARAMS} --config_path ${CONFIG_PATH}"
    fi

    if [ "${USE_SNAPKV}" == "true" ]; then
        MINFERENCE_PARAMS="${MINFERENCE_PARAMS} --use_snapkv"
    fi

    echo "MInference enabled with params: ${MINFERENCE_PARAMS}"
fi

# SparseLLM
ATTENTION_TYPE=$4
ATTENTION_CONFIG=$5

if [ "${MODEL_FRAMEWORK}" == "sparse_llm" ]; then
    SPARSELLM_PARAMS="--attention_type ${ATTENTION_TYPE}"

    if [ -n "${ATTENTION_CONFIG}" ]; then
        SPARSELLM_PARAMS="${SPARSELLM_PARAMS} --attention_config ${ATTENTION_CONFIG}"
    fi

    echo "SparseLLM enabled"
    echo "Attention type: ${ATTENTION_TYPE}"
    echo "Attention config: ${ATTENTION_CONFIG}"
fi

if [ "${TRUST_REMOTE_CODE}" == "true" ]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} --trust_remote_code"
fi

if [ "${KV_CACHE_CPU}" == "true" ]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} --kv_cache_cpu --kv_cache_cpu_device cpu"
fi


for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}_${MODEL_FRAMEWORK}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python ${RULER_PATH}/data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${MODEL_NAME} \
            --tokenizer_type "hf" \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        python ${RULER_PATH}/pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_NAME} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            ${SPARSELLM_PARAMS} \
            ${MINFERENCE_PARAMS} \
            ${EXTRA_PARAMS} \
            ${STOP_WORDS}
    done

    python ${RULER_PATH}/eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done
