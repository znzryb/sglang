# SGLang GPU kernel 审计存档

- 基线：`sgl-project/sglang` main @ `21fd6ae218`（2026-10-24），`sglang-kernel==0.4.9`
- 环境：Modal H100 80GB HBM3，driver 610.57.04，torch 2.14.1+cu130，triton 3.8.0，CUDA 13.0.1 `compute-sanitizer`
- 复现：`cd .gpu-audit && uv run --with "modal[api-proxy-support]" modal run gpu_run.py --cmd "bash run_all.sh"`（`run_aot.sh` 同理）。
  `gpu_run.py` 把本地 `python/` 挂进镜像，测的是本地源码，不是 PyPI 版 sglang。
- 原始日志：`logs/`

## 已实测确认的问题

| # | 位置 | 问题类型 | 触发阈值 | 实测结果 | 已有 PR / issue |
|---|------|----------|----------|----------|-----------------|
| 1 | `python/sglang/kernels/ops/speculative/reject_sampling.py:37`（Triton，已提 issue #46299） | `pid * stride_tp_b` 用 int32 计算，溢出 | `bs*nd*V ≥ 2^31`（Qwen V=152064、nd=4 时 bs ≥ 3532） | bs=3531 正常；bs=3532 报 `CUDA_ERROR_INVALID_ADDRESS_SPACE`；bs=4000 报 illegal memory access；sanitizer 报 `Invalid __global__ read` @ `reject_sampling.py:135`。改成 `.to(tl.int64)` 后 bs=4000 时 0/4000 行错误，sanitizer 0 errors | 无。#41283（OPEN）改的是同一个 kernel，但没修这个问题 |
| 2 | `python/sglang/kernels/jit/csrc/moe/topk_sum.cuh:98`（JIT CUDA） | M 放在 `gridDim.y`，没有限制在 65535 以内 | M ≥ 65536 | M=65535 正常；M=65536 报 `CUDA error: invalid argument` @ `topk_sum.cuh:99` | #34822 修过，但已关闭且未合并 |
| 3 | `python/sglang/kernels/aot/csrc/moe/prepare_moe_input.cu`，`apply_shuffle_mul_sum`（AOT） | `src_row * row_stride` 用 int32 计算，溢出 | `m*topk*k ≥ 2^31`（DSV3 k=7168、topk=8 时 m ≥ 37450） | m=37440 正常；m=37500 报 `CUDA_ERROR_INVALID_ADDRESS_SPACE`；sanitizer 报 2601 处错误 | 无。#25495 改了同一个 kernel，但没修这个问题 |
| 4 | `python/sglang/kernels/aot/csrc/moe/moe_sum_reduce.cu`（AOT） | `grid.y` 被截到 65535，但没有按 token 做 stride 循环，尾部 token 不写输出，属于静默错误 | bf16 时 T > 524280；fp16 时 T > 262140 | bf16 T=524296 有 16 行没写；fp16 T=262152 有 12 行没写，首个出错行正好是阈值 | 无 |

### 在 serving 中能否走到

1. **reject_sampling**：`--speculative-use-rejection-sampling`（以及 block verification）走 `eagle_utils.py:941` 的 Triton 路径。需要大 batch、大词表，单份 fp32 target_probs 约 8 GB 时才会触发，属于边缘场景。修复只需改一行，同目录的 `topk1.py` 和 `renorm_triton.py` 已经修过同类问题，可作参照。
2. **moe_topk_sum**：MXFP4 Marlin MoE，见 `fused_marlin_moe.py:416`。单次 MoE forward 超过 65535 个 token 就会崩，比如 DP attention 把 token 聚合起来，或者用很大的 chunked prefill。#34822 的作者在 DeepSeek-V4-Flash 的真实部署中遇到过这个问题。
3. **apply_shuffle_mul_sum**：`--moe-runner-backend cutlass` 加 FP8 blockwise MoE，见 `cutlass_moe.py:348`。这条路径不按 token 分块，DSV3 单次超过约 37K token 就会越界。同一函数还有两个问题：m=0 时 host 端整数除零（SIGFPE）；`16 / sizeof(scalar_type)` 取的是枚举的 sizeof，不是元素大小。这两个还没实测。
4. **moe_sum_reduce**：`fused_moe.py:972` 和 `fused_marlin_moe`。需要单次 forward 超过 26 万或 52 万个 token，场景是关闭 chunked prefill 跑超长 prompt。

> 第 3、4 条在 AOT wheel 里，修复后要发新版 `sglang-kernel`，再 bump 0.4.9 的 pin。

## 读代码发现、未实测的弱候选

- JIT `rmsnorm`、`fused_add_rmsnorm`、`qknorm` 在 num_tokens==0 时以 grid=0 启动。没有确认 serving 能走到。
- JIT `set_mla_kv_buffer`：`loc` 的校验写的是 `with_strides({-1})`，但 kernel 按连续内存读。目前所有调用方传的都是连续张量。
- JIT `TreeSpeculativeSamplingTargetOnly`：`cur_prob_offset` 用 uint32 计算，需要每个 prob 张量超过 16 GB 才会触发。
- AOT `transfer_kv_launcher`：索引为空时 host 端除零。Python 侧大多已经判空。
- Triton `row_argmax`：小 batch 路径遇到全 NaN 的 split 时返回越界索引 N。
- 现成 issue 里的越界问题（#44535 `stop_token_ids`、#43725、#44998）都已经有好几个修复 PR 在排队，不再跟进。
