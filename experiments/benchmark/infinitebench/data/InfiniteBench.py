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
import os
import json
import datasets


task_list = [
    # Retrieval tasks
    "passkey",
    "number_string",
    "kv_retrieval",
    # Book tasks
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longbook_qa_chn",
    "longdialogue_qa_eng",
    # Math tasks
    "math_find",
    "math_calc",
    # Code tasks
    "code_run",
    "code_debug",
]


class InfiniteBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("2.21.0"), **kwargs)


class InfiniteBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        InfiniteBenchConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "input": datasets.Value("string"),
                "context": datasets.Value("string"),
                "answer": [datasets.Value("string")],
                "options": [datasets.Value("string")],
            }
        )
        return datasets.DatasetInfo(
            description="Infinite Bench Dataset",
            features=features,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"{task_name}.jsonl"),
                },
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                key = f"{self.config.name}-{idx}"
                item = json.loads(line)
                yield key, {
                    "input": item["input"],
                    "context": item["context"],
                    "answer": item["answer"],
                    "options": item.get("options", []),
                }
