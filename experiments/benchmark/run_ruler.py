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
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import sys
from tqdm.auto import tqdm
import yaml
import importlib
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from flex_prefill import patch_model
from utils import (
    seed_everything,
    get_args,
    str_to_dict,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()

SEQ_LENGTHS = ["4096", "8192", "16384", "32768", "65536", "131072"]


TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

TASK_TO_MAX_NEW_TOKNES = {
    "niah_single_1": 256,
    "niah_single_2": 256,
    "niah_single_3": 256,
    "niah_multikey_1": 256,
    "niah_multikey_2": 256,
    "niah_multikey_3": 256,
    "niah_multivalue": 256,
    "niah_multiquery": 256,
    "vt": 256,
    "cwe": 256,
    "fwe": 256,
    "qa_1": 256,
    "qa_2": 256,
}


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataloader(data_list):
    data_loader = DataLoader(ListDataset(data_list), batch_size=1, shuffle=False)
    return data_loader


def get_tasks(task_str: str):
    if task_str == "ruler":
        tasks = []
        for t in TASKS:
            for s in SEQ_LENGTHS:
                tasks.append((t, s))
        return tasks
    elif task_str.startswith("ruler"):
        tasks = []
        length = task_str.split(",")[-1]
        for t in TASKS:
            tasks.append((t, length))
        return tasks
    else:
        task, length = task_str.split(",")
        return [(task, length)]


def remove_duplicates_by_index(list_of_dicts):
    seen_indices = set()
    unique_list = []

    for item in list_of_dicts:
        index = item.get("index")
        if index not in seen_indices:
            unique_list.append(item)
            seen_indices.add(index)

    return unique_list


def main():
    args = get_args()
    # set random seed
    seed_everything(args.seed)
    # save dir
    model_name = args.model.strip("/").split("/")[-1]
    save_dir = os.path.join(args.save_dir, "ruler", model_name)
    save_name = f"{args.task.split(',')[0]}_{'greedy' if args.top_p <0 else 'topp'+str(args.top_p)+'_temp'+str(args.temperature)}{'_chat' if args.chat else ''}_seed{args.seed}"
    if args.tag != "":
        save_name = f"{args.tag}_" + save_name
    save_dir = os.path.join(save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    # set additional pattern and model config
    attention_pattern = args.attention
    attention_config = str_to_dict(args.cfg)
    patch_model(model, attention_pattern, attention_config)
    # RULER load
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.append(os.path.join(curr_folder, "ruler"))
        module = importlib.import_module(f"data.synthetic.constants")
    except ImportError:
        print(f"Module data.synthetic.constants not found.")
    tasks_base = module.TASKS
    with open(
        os.path.join(curr_folder, os.path.join(curr_folder, "ruler", "synthetic.yaml")),
        "r",
    ) as f:
        tasks_customized = yaml.safe_load(f)
    # get dataloader
    dataloaders = []
    all_tasks = get_tasks(args.task)
    for task, length in all_tasks:
        if task not in tasks_customized:
            raise ValueError(f"{task} is not found in config_tasks.yaml")
        config = tasks_customized.get(task)
        config.update(tasks_base[config["task"]])
        # ruler data
        if "llama" in model_name.lower():
            task_file = os.path.join(
                "experiments/benchmark/ruler/data/llama",
                length,
                task,
                "validation.jsonl",
            )
        elif "yi" in model_name.lower():
            task_file = os.path.join(
                "experiments/benchmark/ruler/data/llama",
                length,
                task,
                "validation.jsonl",
            )
        elif "qwen" in model_name.lower():
            task_file = os.path.join(
                "experiments/benchmark/ruler/data/qwen",
                length,
                task,
                "validation.jsonl",
            )
        elif "glm" in model_name.lower():
            task_file = os.path.join(
                "experiments/benchmark/ruler/data/glm", length, task, "validation.jsonl"
            )
        os.makedirs(os.path.join(save_dir, length), exist_ok=True)
        pred_file = os.path.join(save_dir, length, f"{task}.jsonl")
        # Load data
        if os.path.exists(pred_file):
            pred_index = [sample["index"] for sample in read_manifest(pred_file)]
            data = [
                sample
                for sample in read_manifest(task_file)
                if sample["index"] not in pred_index
            ]
        else:
            data = read_manifest(task_file)

        dataloaders.append(get_dataloader(data))

    # accelerate
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)

    for loader, (task, length) in zip(dataloaders, all_tasks):
        loader = accelerator.prepare_data_loader(loader)
        # get pred
        pred_file = os.path.join(save_dir, length, f"{task}.jsonl")
        outputs_parallel = []

        def get_output(index, input, outputs, others, truncation, length):
            if args.chat:
                try:
                    input_ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": input}],
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)
                except:
                    input_ids = tokenizer(
                        input, return_tensors="pt", return_attention_mask=False
                    ).input_ids.to(model.device)
            else:
                input_ids = tokenizer(
                    input, return_tensors="pt", return_attention_mask=False
                ).input_ids.to(model.device)
            do_sample = False if args.top_p <= 0 else True
            generation_config = dict(
                do_sample=do_sample,
                max_new_tokens=TASK_TO_MAX_NEW_TOKNES[task],
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                generation_config["top_p"] = args.top_p
                generation_config["temperature"] = args.temperature
            output = model.generate(input_ids, **generation_config)
            generated_text = tokenizer.decode(
                output[0][input_ids.shape[1] :], skip_special_tokens=True
            )
            # remove the input form the generated text
            if generated_text.startswith(input):
                generated_text = generated_text[len(input) :]
            # remove the </s> from llama-3-8b-262k
            if generated_text.find("</s>") > 0:
                generated_text = generated_text[: generated_text.find("</s>")]
            pred = {"text": [generated_text]}

            if len(pred["text"]) > 0:
                return {
                    "index": int(index),
                    "pred": pred["text"][0],
                    "input": input,
                    "outputs": outputs,
                    "others": others,
                    "truncation": truncation,
                    "length": length,
                }

        pbar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)

        for idx, data_point in enumerate(loader):
            output = get_output(
                data_point["index"][0],
                data_point["input"][0],
                data_point["outputs"][0],
                data_point.get("others", [{}])[0],
                data_point.get("truncation", [-1])[0],
                int(data_point.get("length", [-1])[0]),
            )
            outputs_parallel.append(output)
            pbar.set_description(desc=f"task {task}, len {length}")
            pbar.update(1)
        outputs_parallel = accelerator.gather_for_metrics(outputs_parallel)
        outputs_parallel = remove_duplicates_by_index(outputs_parallel)
        if accelerator.is_main_process:
            with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
                for idx in range(len(outputs_parallel)):
                    fout.write(json.dumps(outputs_parallel[idx]) + "\n")
        accelerator.wait_for_everyone()

    all_length = set([length for _, length in all_tasks])

    for length in all_length:
        if accelerator.is_main_process:
            pred_dir = os.path.join(save_dir, length)
            cmd = f"python experiments/benchmark/ruler/eval/evaluate.py --data_dir {pred_dir} --benchmark synthetic"
            os.system(cmd)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
