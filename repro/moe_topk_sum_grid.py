import sys, torch
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
M = int(sys.argv[1]); T, K = 4, 2880
x = torch.randn(M, T, K, dtype=torch.bfloat16, device="cuda")
out = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
try:
    moe_topk_sum(x, out); torch.cuda.synchronize()
    ref = x.float().sum(1).bfloat16()
    print(f"M={M} ok, max_abs_err={(out.float()-ref.float()).abs().max().item():.4f}")
except Exception as e:
    print(f"M={M} FAILED: {type(e).__name__}: {str(e)[:300]}")
