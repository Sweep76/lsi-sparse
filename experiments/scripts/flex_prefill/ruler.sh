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

accelerate launch --main_process_port 25678 experiments/benchmark/run_ruler.py \
    --model model/Llama-3.1-8B-Instruct \
    --chat \
    --task ruler \
    --attention flex_prefill \
    --cfg block_size=128,flex_prefill_min_budget=1024,flex_prefill_gamma=0.95,flex_prefill_tau=0.1 \
    --tag flex_prefill

accelerate launch --main_process_port 25678 experiments/benchmark/run_ruler.py \
    --model model/glm-4-9b-chat-1m \
    --chat \
    --task ruler \
    --attention flex_prefill \
    --cfg block_size=128,flex_prefill_min_budget=1024,flex_prefill_gamma=0.95,flex_prefill_tau=0.1 \
    --tag flex_prefill

accelerate launch --main_process_port 25678 experiments/benchmark/run_ruler.py \
    --model model/Yi-9B-200K \
    --task ruler \
    --attention flex_prefill \
    --cfg block_size=128,flex_prefill_min_budget=1024,flex_prefill_gamma=0.9,flex_prefill_tau=0.1 \
    --tag flex_prefill

accelerate launch --main_process_port 25678 experiments/benchmark/run_ruler.py \
    --model model/Qwen2-7B-Instruct \
    --chat \
    --task ruler \
    --attention flex_prefill \
    --cfg block_size=128,flex_prefill_min_budget=1024,flex_prefill_gamma=0.9,flex_prefill_tau=0.1 \
    --tag flex_prefill