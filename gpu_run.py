"""Run a shell command on a Modal GPU with the local sglang checkout installed.

usage: uv run modal run gpu_run.py --cmd "python /work/repro/x.py" [--gpu H100]
Local dirs: sglang/python -> /sgl/python (editable install), ./repro -> /work/repro
"""

import subprocess

import modal

import os
REPO = os.environ.get("SGL_REPO", "/Users/zzy/Documents/GitHub/sglang")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "libnuma1")
    .pip_install("uv")
    # install deps once from a snapshot of pyproject, then overlay live source
    .add_local_file(f"{REPO}/python/pyproject.toml", "/deps/pyproject.toml", copy=True)
    .run_commands(
        "cd /deps && python -c \"import tomllib;d=tomllib.load(open('pyproject.toml','rb'));"
        "print('\\n'.join(d['project']['dependencies']))\" > req.txt && "
        "uv pip install --system -r req.txt",
    )
    .run_commands("uv venv /opt/sgleval --python 3.12 && VIRTUAL_ENV=/opt/sgleval uv pip install sgl-eval==0.1.3")
    .env({"CUDA_HOME": "/usr/local/cuda", "PYTHONPATH": "/sgl/python"})
    .add_local_dir(f"{REPO}/python", "/sgl/python", ignore=["**/__pycache__", "**/*.so"])
    .add_local_dir("repro", "/work/repro")
)

app = modal.App("sglang-gpu-audit", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
jit_cache = modal.Volume.from_name("sgl-jit-cache", create_if_missing=True)


def _run(cmd: str):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(["bash", "-lc", cmd], cwd="/work/repro")
    print(f"[exit {r.returncode}]")


kw = dict(timeout=3600, volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/sgl": jit_cache})


@app.function(gpu="H100", **kw)
def run_h100(cmd: str):
    _run(cmd)


@app.function(gpu="A100-80GB", **kw)
def run_a100(cmd: str):
    _run(cmd)


@app.function(gpu="H100:2", **kw)
def run_h100x2(cmd: str):
    _run(cmd)


@app.function(gpu="H100:4", **kw)
def run_h100x4(cmd: str):
    _run(cmd)


@app.function(gpu="L4", cpu=4, timeout=1200, volumes=kw["volumes"])
def run_l4(cmd: str):
    _run(cmd)


@app.function(cpu=2, timeout=600, volumes=kw["volumes"])
def run_cpu(cmd: str):
    _run(cmd)


@app.local_entrypoint()
def main(cmd: str = "nvidia-smi", gpu: str = "H100"):
    {"H100": run_h100, "A100": run_a100, "H100x2": run_h100x2, "H100x4": run_h100x4, "L4": run_l4, "CPU": run_cpu}[gpu].remote(cmd)
