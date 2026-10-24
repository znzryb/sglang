### Checklist

- [x] I searched related issues but found no solution.
- [x] The bug persists in the latest version.
- [x] Issues without environment info and a minimal reproducible demo are hard to resolve and may receive no feedback.
- [x] If this is not a bug report but a general question, please start a discussion at https://github.com/sgl-project/sglang/discussions. Otherwise, it will be closed.
- [x] Please use English. Otherwise, it will be closed.

### Describe the bug

On CUDA, every DeepSeek-family AWQ checkpoint whose `kv_b_proj` is quantized fails during weight loading. The server never starts:

```
ModuleNotFoundError: No module named 'sglang.kernel_api_logging'
```

The CUDA branch of `awq_dequantize_func()` still imports the module from its old location:

https://github.com/sgl-project/sglang/blob/059ba6b666/python/sglang/srt/models/deepseek_common/utils.py#L103-L105

```python
if _is_cuda:
    from sglang.kernel_api_logging import debug_kernel_api   # module no longer exists
```

#35062 moved that module to `sglang.kernels.kernel_api_logging`. The HIP and NPU branches of the same function already import from the new path, a few lines below. The stale CUDA import came in with #32114 (c0b790cf7f, which swapped the AWQ AOT kernel for `sglang.kernels.ops.quantization.awq_dequantize`).

`DeepseekV2WeightLoaderMixin.post_load_weights` calls this function whenever `kv_b_proj` still has a packed `qweight` (`deepseek_weight_loader.py:625-627`), so it is reached at startup with default flags.

Affected checkpoints include `QuantTrio/DeepSeek-V3.1-AWQ`, `QuixiAI/DeepSeek-R1-AWQ`, `QuixiAI/DeepSeek-V3-0324-AWQ` and `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ`. All of them quantize `kv_b_proj`: their `modules_to_not_convert` is empty, lists only some first/last layers (V3.1), or lists only `kv_a_proj_with_mqa` (V3-0324).

Proposed fix (one line): import `debug_kernel_api` from `sglang.kernels.kernel_api_logging`, as the HIP and NPU branches do. I can send a PR.

Related but different: #34102 (open) fixes an earlier AttributeError on the `q_b_proj.weight` probe for DeepSeek-V3.x AWQ, which happens at module construction. Models with `q_lora_rank` would hit that error first, and then this one once #34102 lands. DeepSeek-V2-Lite has no `q_b_proj`, so it hits this one directly.

### Reproduction

```bash
python3 -m sglang.launch_server --model-path TechxGenus/DeepSeek-V2-Lite-Chat-AWQ --trust-remote-code
```

1x H100, main @ bf1a252953 (the line is unchanged on 059ba6b666):

```
  File "sglang/srt/models/deepseek_v2.py", line 3428, in load_weights
    self.do_load_weights(weights, is_nextn)
  File "sglang/srt/models/deepseek_common/deepseek_weight_loader.py", line 550, in do_load_weights
    self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)
  File "sglang/srt/models/deepseek_common/deepseek_weight_loader.py", line 627, in post_load_weights
    awq_dequantize_f = awq_dequantize_func()
  File "sglang/srt/models/deepseek_common/utils.py", line 104, in awq_dequantize_func
    from sglang.kernel_api_logging import debug_kernel_api
ModuleNotFoundError: No module named 'sglang.kernel_api_logging'
```

### Environment

sglang from source at `bf1a252953`, on Modal:

```
Python: 3.12.1
GPU 0: NVIDIA H100 80GB HBM3
GPU 0 Compute Capability: 9.0
CUDA Driver Version: 610.57.04
NVCC: Cuda compilation tools, release 13.0, V13.0.88
PyTorch: 2.14.1+cu130
sglang-kernel: 0.4.9
flashinfer_python: 0.7.1
triton: 3.8.0
transformers: 5.19.0
```
