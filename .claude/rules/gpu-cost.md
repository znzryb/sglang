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

- **Parallel jobs are fine; unwatched jobs are not.** Every running job must be
  monitored until it ends.
- **Poll the live logs.** `modal run` streams container stdout/stderr live to
  the local process (tee it to a file and read that file), and
  `modal app logs <app-id>` attaches to any running app. Check within the first
  1-2 minutes of each job, then at intervals matched to its next expected
  milestone (image built, weights loaded, server up, eval done). Polling reads
  Modal's control plane, not the container, so it costs no GPU time.
- **Stop on the first anomaly.** A repeated error line, a missing command, or
  no progress past the expected milestone means `modal app stop -y <id>` now,
  not after the timeout. Never wait passively for a completion notification.
- **Hard timeout on every job** (container `timeout` plus a shell `timeout`),
  sized to the expected runtime, not a generous upper bound.
- **Fail fast.** A health check must exit as soon as the server process dies
  or its probe errors, never loop until the timeout.

## Known Modal pitfalls (each one has burned money here)

- **GPU functions get ~1 vCPU by default.** SGLang JIT-compiles CUDA kernels
  (Marlin GEMM/MoE take 4-5 min each on one core) on first use, so a cold
  startup can exceed 15 min with the GPU idle. Set `cpu=` (e.g. 4-8) on GPU
  functions; extra cores are far cheaper than idle GPU minutes.
- **Persist the JIT cache.** SGLang builds kernels under
  `/root/.cache/sglang/jit`; mount a Volume there (or copy to/from one) so a
  rerun does not recompile. A volume mounted at a different path saves nothing.
- **The CUDA base image has no `curl`.** Health checks written with `curl`
  fail forever and look like a slow startup. Use Python `urllib` (and never
  name a helper script `http.py`, which shadows the stdlib `http` package).
- **Stopping the local `modal run` does not always stop the app.** Ephemeral
  apps can keep running; verify with `modal app list` and stop leftovers with
  `modal app stop -y <id>`.
- **A silent log is not necessarily a hang.** Before killing, look inside:
  `modal container list`, then `modal container exec <id> -- ps -eo etime,pcpu,args`
  or `py-spy dump --pid <pid>` (e.g. a stuck "Capturing prefill shape" was
  actually `nvcc`/`cicc` compiling on one core).
- **Credits do not cap spend.** On the Starter plan, usage past the monthly
  credits is charged to the card up to a Modal-set limit that auto-raises.
  Only a user-set spend limit stops it. Credits and `modal billing report`
  lag real usage by up to an hour, so a falling balance after all apps stopped
  is delayed metering, not a running job (confirm with `modal container list`).

## After a run, or when it fails

- **Read the log before rerunning.** Find the actual cause; never rerun with a
  longer timeout on a guess.
- **Confirm nothing is left running:** `modal app list` shows no live app from
  this session; stop orphans with `modal app stop -y <id>`.
- **Report the actual cost** with `modal billing report --for today
  --show-resources`, including any wasted spend and its cause.
