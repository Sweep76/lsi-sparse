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

tasks=('longbook_sum_eng' 'longbook_qa_eng' 'longbook_choice_eng' 'longdialogue_qa_eng' 'longbook_qa_chn' 'code_debug' 'math_find' 'passkey' 'number_string' 'kv_retrieval')

for s in "${tasks[@]}"
do
    accelerate launch --main_process_port 25678 experiments/benchmark/run_infinitebench.py \
        --model model/Llama-3.1-8B-Instruct \
        --max_length 131072 \
        --chat \
        --task "${s}" \
        --attention flash \
        --tag flash
done


for s in "${tasks[@]}"
do
    accelerate launch --main_process_port 25678 experiments/benchmark/run_infinitebench.py \
        --model model/glm-4-9b-chat-1m \
        --max_length 160000 \
        --chat \
        --task "${s}" \
        --attention flash \
        --tag flash
done


for s in "${tasks[@]}"
do
    accelerate launch --main_process_port 25678 experiments/benchmark/run_infinitebench.py \
        --model model/Yi-9B-200K \
        --max_length 204800 \
        --task "${s}" \
        --attention flash \
        --tag flash
done


for s in "${tasks[@]}"
do
    accelerate launch --main_process_port 25678 experiments/benchmark/run_infinitebench.py \
        --model model/Qwen2-7B-Instruct \
        --max_length 131072 \
        --chat \
        --task "${s}" \
        --attention flash \
        --tag flash
done