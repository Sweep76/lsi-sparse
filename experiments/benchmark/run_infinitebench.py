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
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import torch
import torch.distributed as dist
import json
import os
from flex_prefill import patch_model
from utils import (
    tok_encode_middle_trunc,
    tok_batch_encode_middle_trunc,
    convert_to_json_compatible,
    seed_everything,
    get_args,
    str_to_dict,
    fixed_generate_until,
)
import warnings

warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")

os.environ["ENABLE_PREFILL_TIMER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# use middle truncate instead of left or right
HFLM.tok_encode = tok_encode_middle_trunc
HFLM.tok_batch_encode = tok_batch_encode_middle_trunc
HFLM.generate_until = fixed_generate_until


def main():
    args = get_args()
    # set random seed
    seed_everything(args.seed)
    # save dir
    model_name = args.model.strip("/").split("/")[-1]
    save_dir = os.path.join(args.save_dir, "infinitebench", model_name)
    if args.tag != "":
        save_dir = os.path.join(save_dir, args.tag)
    save_name = f"{args.task}_len{args.max_length}_{'greedy' if args.top_p <0 else 'topp'+str(args.top_p)+'_temp'+str(args.temperature)}{'_chat' if args.chat else ''}_seed{args.seed}.json"
    if args.tag != "":
        save_name = f"{args.tag}_" + save_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, save_name)
    # warp model using lm_eval huggingface model
    model = HFLM(
        pretrained=args.model,
        backend="causal",
        max_length=args.max_length,
        dtype=torch.bfloat16,
        batch_size=args.batch_size if args.batch_size > 0 else "auto",
        # additional kwargs for model loading
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    # set attention pattern and additional model config
    attention_pattern = args.attention
    attention_config = str_to_dict(args.cfg)
    patch_model(model._model, attention_pattern, attention_config)
    # use your own tasks
    task_manager = TaskManager(
        include_path="experiments/benchmark/infinitebench", include_defaults=False
    )
    # evaluate
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=args.task.split(","),
        task_manager=task_manager,
        apply_chat_template=args.chat,  # use True to enable chat template
        gen_kwargs=f"do_sample={'False' if args.top_p <= 0 else 'True'},top_p={args.top_p},temperature={args.temperature}",
        log_samples=args.limit
        > 0,  # Log all samples to debug. If use log_samples=True, the result file will be very large.
        limit=args.limit if args.limit > 0 else None,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    # save result
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        results = convert_to_json_compatible(results)
        results["run_args"] = vars(args)
        results["attention"] = attention_pattern
        results["attention_config"] = attention_config
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
