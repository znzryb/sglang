"""Repro: speculative_sampling_classic_kernel int32 overflow of pid * stride_tp_b.

Each row b's target distribution puts all mass on token tok[b]; draft probs are 0,
so every draft is rejected and the first predicted token must be tok[b].
usage: python reject_sampling_overflow.py [orig|fixed] [bs] [vocab] [nd]
"""

import importlib.util
import pathlib
import sys

import torch

variant = sys.argv[1] if len(sys.argv) > 1 else "orig"
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 3532
V = int(sys.argv[3]) if len(sys.argv) > 3 else 152064
nd = int(sys.argv[4]) if len(sys.argv) > 4 else 4

import sglang.kernels.ops.speculative.reject_sampling as rs

if variant == "fixed":
    src = pathlib.Path(rs.__file__).read_text()
    patched = src.replace(
        "pid = tl.program_id(0)\n", "pid = tl.program_id(0).to(tl.int64)\n", 1
    )
    assert patched != src
    p = pathlib.Path("/tmp/reject_sampling_fixed.py")
    p.write_text(patched)
    spec = importlib.util.spec_from_file_location("reject_sampling_fixed", p)
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)

dev = "cuda"
print(f"variant={variant} bs={bs} V={V} nd={nd} bs*nd*V={bs*nd*V} (2^31={2**31})")
print(f"first overflowing row: {(2**31 - 1) // (nd * V) + 1}")

tok = (torch.arange(bs, device=dev) * 37) % (V - 1) + 1
tp = torch.zeros(bs, nd, V, device=dev, dtype=torch.float32)
tp[torch.arange(bs, device=dev), :, tok] = 1.0
dp = torch.zeros(1, nd, V, device=dev, dtype=torch.float32).expand(bs, nd, V)
cand = torch.zeros(bs, nd, dtype=torch.int64, device=dev)  # token 0: q=0 -> rejected
ridx = torch.arange(bs * nd, device=dev, dtype=torch.int64).view(bs, nd)
predicts = torch.full((bs * nd,), -1, dtype=torch.int32, device=dev)
acc_idx = torch.full((bs, nd), -1, dtype=torch.int32, device=dev)
acc_n = torch.zeros(bs, dtype=torch.int32, device=dev)
u = torch.rand(bs, nd, device=dev)
uf = torch.rand(bs, device=dev)

rs.chain_speculative_sampling_triton(
    predicts, acc_idx, acc_n, cand, ridx, None, None, u, uf, tp, dp, 1.0, 1.0, True
)
torch.cuda.synchronize()

got = predicts[ridx[:, 0]].long()
bad = (got != tok).nonzero().flatten()
print(f"wrong rows: {bad.numel()} / {bs}")
if bad.numel():
    print(f"first wrong row={bad[0].item()} expected={tok[bad[0]].item()} got={got[bad[0]].item()}")
