"""AOT sgl_kernel.apply_shuffle_mul_sum: int32 src_row*row_stride overflow (cutlass FP8 MoE combine)."""
import sys, torch, sgl_kernel
m = int(sys.argv[1]); topk, k = 8, 7168   # DeepSeek-V3 hidden
print(f"m={m} m*topk*k={m*topk*k} (2^31={2**31})")
torch.manual_seed(0)
inp = torch.randn(m * topk, k, dtype=torch.bfloat16, device="cuda")
perm = torch.randperm(m * topk, device="cuda", dtype=torch.int32)
w = torch.rand(m * topk, dtype=torch.bfloat16, device="cuda")
out = torch.full((m, k), float("nan"), dtype=torch.bfloat16, device="cuda")
sgl_kernel.apply_shuffle_mul_sum(inp, out, perm, w)
torch.cuda.synchronize()
bad = 0
for s in range(0, m, 4096):
    e = min(m, s + 4096)
    p = perm[s * topk:e * topk].long()
    ref = (inp[p].float().view(-1, topk, k) * w[s * topk:e * topk].float().view(-1, topk, 1)).sum(1)
    err = (out[s:e].float() - ref).abs().amax(1)
    bad += int(((err > 0.1) | err.isnan()).sum())
print(f"wrong rows: {bad} / {m}")
