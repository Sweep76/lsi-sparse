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
import math
import torch
import triton
from flex_prefill.ops.flex_prefill_attention import (
    triton_block_wise_prefill_attention,
    triton_flash_prefill,
)
from flash_attn import flash_attn_func


def build_random_block_idx(B, N, H, sparsity, block_size, offset):
    num_block = math.ceil(offset / block_size) + math.ceil((N - offset) / block_size)
    total_blocks = num_block * (num_block + 1) // 2
    topk = math.ceil(total_blocks * (1 - sparsity))
    block_idx = torch.randn(
        B, H, num_block, num_block, device="cuda", dtype=torch.bfloat16
    )
    causal_mask = (
        torch.arange(num_block)[:, None] >= torch.arange(num_block)[None, :]
    ).to("cuda")
    block_idx.masked_fill_(~causal_mask, float("-inf"))
    block_idx[..., 0] = float("inf")
    block_idx = torch.topk(block_idx.view(B, H, -1), k=topk, dim=-1).indices
    return block_idx


if __name__ == "__main__":
    torch.manual_seed(0)

    # benchmark
    print("Runing speed benchmark...")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 8)],
            line_arg="provider",
            line_vals=["flash", "triton-flash", "sparse-75", "sparse-95"],
            line_names=["Flash", "Triton-Flash", "Sparse-75%", "Sparse-95%"],
            styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="sparse attention speed test",
            args={"B": 1, "H": 32, "D": 128, "K": 128},
        )
    )
    def benchmark(B, N, H, D, K, provider):
        q = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
        block_idx_95 = build_random_block_idx(B, N, H, 0.95, K, 0)
        block_idx_75 = build_random_block_idx(B, N, H, 0.75, K, 0)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: flash_attn_func(q, k, v, causal=True),
                quantiles=quantiles,
            )
        if provider == "triton-flash":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_flash_prefill(q, k, v, causal=True),
                quantiles=quantiles,
            )
        if provider == "sparse-95":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_block_wise_prefill_attention(q, k, v, block_idx_95, K),
                quantiles=quantiles,
            )
        if provider == "sparse-75":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_block_wise_prefill_attention(q, k, v, block_idx_75, K),
                quantiles=quantiles,
            )
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)
