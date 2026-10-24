# Running on paid GPUs (Modal, cloud): minimum cost that completes the task

GPU time is paid for with real money. Every remote GPU run must be the cheapest
run that still completes the task -- not the cheapest card in absolute terms.

## Before launching

- **Minimize total cost, not the hourly rate.** Total = per-GPU rate x GPU
  count x wall time, plus setup (image build, weight download) and the risk of
  a failed run. A card too small to finish the job is not cheap.
- **Pick the configuration that fits at the lowest total cost.** For single-GPU
  work try L4 / A10G -> L40S -> A100 -> H100, moving up only when a cheaper
  card cannot fit the model or reproduce the behavior (e.g. needs SM90, FP8
  tensor cores, more memory), and state why.
- **Weigh one big GPU against several small ones.** When a model does not fit
  one card, compare e.g. 1 x H100 vs 2 x L40S vs 4 x L4: more GPUs add TP
  communication overhead, longer startup, more failure points, and may need
  flags the real deployment does not use. Choose the option with the lowest
  total cost that still reproduces the target configuration.
- **Pick the smallest model that reproduces the problem.** Use a tiny or Lite
  checkpoint of the same architecture before a full-size one.
- **Preflight on CPU first.** Run the script's tooling checks in a CPU-only
  container (or locally): every command it calls exists (e.g. `curl`, `hf`,
  `sgl-eval`), the health-check logic works, paths resolve. A GPU must never
  sit idle waiting on a broken script.
- **Report the estimated cost** (GPU type x expected minutes x rate, from
  `modal billing rates`) and get the user's approval before launching.

## While running

- **One job at a time.** No parallel batches unless the user explicitly asks.
- **Hard timeout on every job** (container `timeout` plus a shell `timeout`),
  sized to the expected runtime, not a generous upper bound.
- **Fail fast.** A health check must exit as soon as the server process dies
  or its probe errors, never loop until the timeout.

## After a run, or when it fails

- **Read the log before rerunning.** Find the actual cause; never rerun with a
  longer timeout on a guess.
- **Confirm nothing is left running:** `modal app list` shows no live app from
  this session; stop orphans with `modal app stop -y <id>`.
- **Report the actual cost** with `modal billing report --for today
  --show-resources`, including any wasted spend and its cause.
