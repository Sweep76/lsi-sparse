<h1>FlexPrefill</h1>

## TL;DR

FlexPrefill is a dynamic and context-aware sparse attention mechanism that optimizes computational efficiency during long-sequence inference for large language models (LLMs). It achieves this by dynamically adjusting sparse attention patterns and computational budgets in real-time based on input demands and attention head requirements.

## Requirements

To use FlexPrefill, you will need the following packages:

- `torch==2.4.0`
- `triton==3.0.0`
- `transformers==4.44.0`
- `flash_attn==2.6.3` (optional)
- `vllm==0.5.4` (optional)

## Installation

You can install FlexPrefill using pip:
```shell
pip install git+https://github.com/lsi-sparse/Sweep76.git
```

## Quick Start

### Example Test

You can execute the `tests/test_llm.py` script to run a basic test on a specified model. This test includes examples with token lengths ranging from 4k to 128k and logs the model's total execution time.

```shell
# default transformers model inference
python tests/test_llm.py --model meta-llama/Llama-3.1-8B-Instruct --pattern default
# sparse attention inference
python tests/test_llm.py --model meta-llama/Llama-3.1-8B-Instruct --pattern flex_prefill
```

### FlexPrefill Sparse Attention Function

You can invoke flex prefill sparse attention using the following codes. Note: The current version only supports inference with a batch size of 1 and has only been tested with bfloat16 precision.

```python
import torch
from flex_prefill import flex_prefill_attention

B, N, H, D = 1, 64000, 32, 64
gamma = 0.9
tau = 0.1

q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, N, H // 4, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, N, H // 4, D, device="cuda", dtype=torch.bfloat16)

flex_prefill_output = flex_prefill_attention(
    q,
    k,
    v,
    gamma,
    tau,
    min_budget=512,
    max_budget=None,
)
```

### Hugging Face Transformers Model Inference

FlexPrefill supports models from Hugging Face transformers. You can convert a model to use sparse attention by using `flex_prefill.patch_model`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from flex_prefill import patch_model


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
).cuda()

flex_prefill_config =  {
    "block_size": 128,
    "flex_prefill_gamma": 0.9,
    "flex_prefill_tau": 0.1,
    "flex_prefill_min_budget": 512,
    "flex_prefill_max_budget": None,
}

patch_model(model, "flex_prefill", flex_prefill_config)

input_ids = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).input_ids.cuda()
output_ids = model.generate(input_ids, max_new_tokens=64)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### vLLM Model Inference

FlexPrefill also supports vLLM models. You can convert a vLLM model to use sparse attention using `flex_prefill.patch_model`. However, please note that support for vLLM has not yet been thoroughly tested.

```python
from vllm import LLM, SamplingParams
from flex_prefill import patch_model


model = LLM("meta-llama/Llama-3.1-8B-Instruct", enable_chunked_prefill=False, max_num_seqs=1)
sampling_params = SamplingParams(temperature=0, max_tokens=64)

flex_prefill_config =  {
    "block_size": 128,
    "flex_prefill_gamma": 0.9,
    "flex_prefill_tau": 0.1,
    "flex_prefill_min_budget": 512,
    "flex_prefill_max_budget": None,
}

patch_model(model, "flex_prefill", flex_prefill_config)

model.generate(prompts=[prompt], sampling_params=sampling_params)
output = outputs[0].outputs[0].text
```

### Supported Models

Currently, `flex_prefill.patch_model` only supports the following models:
- LLaMA: [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Qwen2: [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- ChatGLM4: [THUDM/glm-4-9b-chat-1m](https://huggingface.co/THUDM/glm-4-9b-chat-1m)
- Yi: [01-ai/Yi-9B-200K](https://huggingface.co/01-ai/Yi-9B-200K)
- Other models with LLaMA architecture

## Experiments

Experiment scripts are provided in the `experiments` folder. First, you need to install dependencies, and download the necessary models:

```shell
bash install.sh
bash experiments/download_model.sh
```

Next, you need to download and preprocess the RULER and InfiniteBench datasets:

```shell
bash experiments/benchmark/ruler/download_dataset.sh
bash experiments/benchmark/infinitebench/download_dataset.sh
```

Finally, you can run the experiments using the scripts in the `experiments/scripts` directory. For example:

```shell
bash experiments/scripts/flex_prefill/ruler.sh
bash experiments/scripts/flex_prefill/infinitebench.sh
```

The results will be saved in the `experiments/result` directory.

## Related Projects

This codebase leverages [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) for evaluations on both [RULER](https://github.com/NVIDIA/RULER) and [InfiniteBench](https://github.com/OpenBMB/InfiniteBench). Additionally, it incorporates code snippets from [Minference](https://github.com/microsoft/MInference). Our kernels are implemented using [Triton](https://github.com/triton-lang/triton). We extend our gratitude to the community for their valuable contributions!


## Acknowledgments

We acknowledge the support from our collaborators and the community. Thank you for your contributions and feedback.
