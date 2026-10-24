### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [x] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

`speculative_sampling_classic_kernel` (`python/sglang/kernels/ops/speculative/reject_sampling.py`) computes every per-request base pointer as `pid * stride_*_b` with `pid = tl.program_id(0)` left as int32:

https://github.com/sgl-project/sglang/blob/d0f517168d/python/sglang/kernels/ops/speculative/reject_sampling.py#L37

`target_probs` is `[bs, num_draft_tokens, vocab]` and contiguous, so `stride_tp_b = num_draft_tokens * vocab`. That stride is below 2^31, so Triton specializes it as i32, and `pid * stride_tp_b` is an i32 × i32 product. Once `bs * num_draft_tokens * vocab >= 2^31`, the product wraps negative for the trailing rows. Those rows then read `TargetProbs` / `DraftProbs` out of bounds and crash with an illegal memory access. In-bounds wraps would silently sample from another request's distribution instead.

The neighbouring kernels already fixed this exact pattern. For example, `topk1.py`:

```python
# int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
row = tl.program_id(0).to(tl.int64)
```

`renorm_triton.py` has the same fix; `reject_sampling.py` does not.

**Who hits it:** CUDA EAGLE/MTP with `--speculative-use-rejection-sampling`, with or without `--speculative-use-block-verification`. That path calls `chain_speculative_sampling_triton` from `eagle_utils.py`, and the UNO path calls it too. The threshold is `bs >= ceil(2^31 / (num_draft_tokens * vocab))`, for example:

| vocab | num_draft_tokens | first crashing batch size |
|---|---|---|
| 152,064 (Qwen) | 4 | 3,532 |
| 152,064 (Qwen) | 8 | 1,766 |
| 262,144 (Gemma) | 8 | 1,025 |

These are large batches, but this is exactly the high-throughput regime where rejection sampling is turned on. The default (non-rejection) path uses the JIT CUDA kernel and is not affected by this bug.

**Proposed fix:** one line, `pid = tl.program_id(0).to(tl.int64)`. With that change, every `pid * stride` product is computed in int64. I can send a PR with a regression test.

### Reproduction

Each row `b` puts all target mass on token `tok[b]`, and every draft has `q = 0`, so every draft is rejected and the first predicted token must be `tok[b]`. `draft_probs` is an `expand`ed view (`stride_b = 0`) so only `target_probs` (~8.6 GB fp32) has to fit.

<details><summary>reject_sampling_overflow.py</summary>

```python
import sys
import torch
from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 3532
V, nd, dev = 152064, 4, "cuda"
print(f"bs={bs} bs*nd*V={bs*nd*V} first overflowing row={(2**31 - 1) // (nd * V) + 1}")

tok = (torch.arange(bs, device=dev) * 37) % (V - 1) + 1
tp = torch.zeros(bs, nd, V, device=dev)
tp[torch.arange(bs, device=dev), :, tok] = 1.0
dp = torch.zeros(1, nd, V, device=dev).expand(bs, nd, V)
cand = torch.zeros(bs, nd, dtype=torch.int64, device=dev)  # q(0) = 0 -> always rejected
ridx = torch.arange(bs * nd, device=dev, dtype=torch.int64).view(bs, nd)
predicts = torch.full((bs * nd,), -1, dtype=torch.int32, device=dev)
acc_idx = torch.full((bs, nd), -1, dtype=torch.int32, device=dev)
acc_n = torch.zeros(bs, dtype=torch.int32, device=dev)

chain_speculative_sampling_triton(
    predicts, acc_idx, acc_n, cand, ridx, None, None,
    torch.rand(bs, nd, device=dev), torch.rand(bs, device=dev), tp, dp, 1.0, 1.0, True,
)
torch.cuda.synchronize()
got = predicts[ridx[:, 0]].long()
print(f"wrong rows: {(got != tok).sum().item()} / {bs}")
```

</details>

Results on H100 80GB (main @ 21fd6ae218; `reject_sampling.py` is unchanged on d0f517168d):

| case | bs | result |
|---|---|---|
| current main | 3,531 (last row just fits) | `wrong rows: 0 / 3531` |
| current main | 3,532 | `CUDA error: operation not supported on global/shared address space` (`CUDA_ERROR_INVALID_ADDRESS_SPACE`) |
| current main | 4,000 | `CUDA error: an illegal memory access was encountered` |
| with `.to(tl.int64)` | 4,000 | `wrong rows: 0 / 4000` |

`compute-sanitizer --tool memcheck`, bs=3532:

```
# current main
========= Invalid __global__ read of size 4 bytes
=========     at speculative_sampling_classic_kernel+0x2d0 in reject_sampling.py:135
=========     Access to 0x2ae43c106000 is out of bounds
=========         Host Frame: chain_speculative_sampling_triton in reject_sampling.py:267

# with .to(tl.int64)
wrong rows: 0 / 3532
========= ERROR SUMMARY: 0 errors
```

Related but different: #42568 (q rounding above 1.0 in the same kernel). #41283 (open) touches this kernel, but its diff keeps the same int32 `pid * stride_*_b` products.

### Environment

sglang from source at `21fd6ae218` (the kernel is pure Triton, no wheel involved), on Modal:

```
Python: 3.12.1
CUDA available: True
GPU 0: NVIDIA H100 80GB HBM3
GPU 0 Compute Capability: 9.0
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 13.0, V13.0.88
CUDA Driver Version: 610.57.04
PyTorch: 2.14.1+cu130
sglang-kernel: 0.4.9
flashinfer_python: 0.7.1
triton: 3.8.0
transformers: 5.19.0
```
