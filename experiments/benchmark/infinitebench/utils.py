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
import re


yarn_mistral_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",  # noqa
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa
    "longdialogue_qa_eng": 'Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely',  # noqa
}


def doc_to_text_code_run(doc):
    template = "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is"
    find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", doc["input"])
    func_call = find_result[0]
    func = func_call.split("(")[0]
    return template.format(
        func=func,
        func_call=func_call,
        context=doc["context"],
    )


def doc_to_text_code_debug(doc):
    template = "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:"
    return template.format(
        context=doc["context"],
        OPTION_A=doc["options"][0],
        OPTION_B=doc["options"][1],
        OPTION_C=doc["options"][2],
        OPTION_D=doc["options"][3],
    )


def doc_to_target_code_debug(doc):
    OPTIONS = "ABCD"
    return "===answerspliter===".join(
        [doc["answer"][0], OPTIONS[doc["options"].index(doc["answer"][0])]]
    )


def doc_to_text_longbook_choice_eng(doc):
    template = "Read the book and answer the question.\n\n{context}\n\nQuestion: {input}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is"
    return template.format(
        context=doc["context"],
        input=doc["input"],
        OPTION_A=doc["options"][0],
        OPTION_B=doc["options"][1],
        OPTION_C=doc["options"][2],
        OPTION_D=doc["options"][3],
    )


def doc_to_target_longbook_choice_eng(doc):
    OPTIONS = "ABCD"
    return "===answerspliter===".join(
        [doc["answer"][0], OPTIONS[doc["options"].index(doc["answer"][0])]]
    )


def doc_to_text_math_find(doc):
    template = "{prefix}\n\n{context}\n\n{input}"
    prompt = doc["input"]
    context = doc["context"]
    # Find "the * number" from the prompt
    find_result = re.findall(r"The .+ of", prompt)
    assert find_result, f"Cannot find the target number in {prompt}"
    target_number = find_result[0].lower()[:-3]
    # Replace the number with the answer
    prefix = f"What is {target_number} in the following list?"
    return template.format(
        prefix=prefix,
        context=context,
        input=prompt,
    )
