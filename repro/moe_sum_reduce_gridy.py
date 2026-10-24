"""AOT sgl_kernel.moe_sum_reduce: grid.y clamped to 65535 with no token stride loop -> tail tokens unwritten."""
import sys, torch, sgl_kernel
T = int(sys.argv[1]); dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[sys.argv[2]]
x = torch.randn(T, 4, 16, dtype=dt, device="cuda")
out = torch.full((T, 16), float("nan"), dtype=dt, device="cuda")
sgl_kernel.moe_sum_reduce(x, out, 1.0); torch.cuda.synchronize()
ref = x.float().sum(1)
bad = ((out.float() - ref).abs().amax(1) > 1e-1) | out.isnan().any(1)
idx = bad.nonzero().flatten()
print(f"T={T} {sys.argv[2]} wrong rows: {idx.numel()}" + (f" (first={idx[0].item()})" if idx.numel() else ""))
