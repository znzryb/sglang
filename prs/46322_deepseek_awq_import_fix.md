## Motivation

Fixes #46312.

On CUDA, DeepSeek-family AWQ checkpoints whose `kv_b_proj` is quantized fail while loading weights, and the server never starts:

```
  File "sglang/srt/models/deepseek_common/deepseek_weight_loader.py", line 627, in post_load_weights
    awq_dequantize_f = awq_dequantize_func()
  File "sglang/srt/models/deepseek_common/utils.py", line 104, in awq_dequantize_func
    from sglang.kernel_api_logging import debug_kernel_api
ModuleNotFoundError: No module named 'sglang.kernel_api_logging'
```

#35062 moved `kernel_api_logging` into `sglang.kernels`. #32114 later added this CUDA import with the old path. The HIP and NPU branches of the same function already use `sglang.kernels.kernel_api_logging`. Affected checkpoints include `QuantTrio/DeepSeek-V3.1-AWQ`, `QuixiAI/DeepSeek-R1-AWQ`, `QuixiAI/DeepSeek-V3-0324-AWQ` and `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ`.

## Modifications

- `deepseek_common/utils.py`: import `debug_kernel_api` from `sglang.kernels.kernel_api_logging` in the CUDA branch, matching the HIP and NPU branches.
- New CPU unit test `test/registered/unit/models/test_deepseek_awq_dequantize_func.py`: forces the CUDA branch and checks that `awq_dequantize_func()` returns a callable.

## Accuracy Tests

No numerics change; the dequantize kernel itself is covered by `test/registered/kernels/ops/quantization/test_awq_dequantize.py`.

`python3 -m sglang.launch_server --model-path TechxGenus/DeepSeek-V2-Lite-Chat-AWQ --trust-remote-code` (default flags):

| | main (bf1a252953), 1x H100 | this PR, 1x L4 |
|---|---|---|
| weight loading | `ModuleNotFoundError: No module named 'sglang.kernel_api_logging'`, server exits | `Load weight end. elapsed=15.29 s, type=DeepseekV2ForCausalLM, quant=awq, bits=4` |
| CUDA graph capture | not reached | prefill and decode graphs captured |
| `/health` | never up | 200 |
| chat: "What is 17*23? Answer with the number only." (greedy) | n/a | `17 * 23 = 391` |

New unit test:

| | result |
|---|---|
| without the fix | `FAILED (errors=1)` |
| with the fix | `OK` |

## Speed Tests and Profiling

N/A.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
