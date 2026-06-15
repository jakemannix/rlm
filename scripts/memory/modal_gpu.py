"""Modal GPU runner for the memory-track acceptance ladder.

This keeps the Colab-style commands headless and repeatable while preserving
artifacts on Modal Volumes.

Examples:
    # Local one-time setup:
    #   uv pip install -e ".[modal]"
    #   modal setup
    #
    # Verify CUDA remotely on the default L4:
    #   modal run scripts/memory/modal_gpu.py --job gpu-report
    #
    # Verify Hugging Face auth remotely without printing the token:
    #   modal run scripts/memory/modal_gpu.py --job hf-report
    #
    # Run the full Step 0 -> Step 2c ladder on an H100:
    #   RLM_MODAL_GPU=H100 modal run scripts/memory/modal_gpu.py --job acceptance-ladder
    #
    # Run a custom repo command, with outputs under /vol/rlm:
    #   modal run scripts/memory/modal_gpu.py --job command \
    #       --command "python scripts/memory/train_skill.py --cache /vol/rlm/cache/gemma1b_v1 --out /vol/rlm/runs/skill.pt"
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

REMOTE_REPO_ROOT = "/root/rlm"
ARTIFACT_ROOT = "/vol/rlm"
HF_CACHE_ROOT = "/root/.cache/huggingface"
PYTHON_BIN = "python"

if modal.is_local():
    LOCAL_REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    LOCAL_REPO_ROOT = Path(REMOTE_REPO_ROOT)

APP_NAME = os.environ.get("RLM_MODAL_APP", "rlm-memory-gpu")
ARTIFACT_VOLUME_NAME = os.environ.get("RLM_MODAL_ARTIFACT_VOLUME", "rlm-memory-artifacts")
HF_CACHE_VOLUME_NAME = os.environ.get("RLM_MODAL_HF_CACHE_VOLUME", "rlm-hf-cache")
HF_SECRET_NAME = os.environ.get("RLM_MODAL_HF_SECRET", "")
DEFAULT_GPU_SPEC = os.environ.get("RLM_MODAL_GPU", "L4")
DEFAULT_TIMEOUT = int(os.environ.get("RLM_MODAL_TIMEOUT", str(24 * 60 * 60)))

PYTHON_PACKAGES = [
    "accelerate>=0.30.0",
    "datasets>=2.20.0",
    "hf_transfer>=0.1.9",
    "peft>=0.11.0",
    "protobuf>=4.25.0",
    "requests>=2.32.5",
    "rich>=13.0.0",
    "sentencepiece>=0.2.0",
    "torch>=2.3.0,<3",
    "transformers>=4.56.0,<5",
    # rlm core client deps — importing any rlm.* pulls in rlm.clients, which
    # needs these even when the GPU job only trains/generates.
    "python-dotenv>=1.0.0",
    "openai>=2.0.0",
    "anthropic>=0.40.0",
    "google-genai>=1.0.0",
    "portkey-ai>=2.0.0",
]


def gpu_spec() -> str | list[str]:
    """Parse Modal's GPU spec from RLM_MODAL_GPU.

    Modal accepts strings such as ``L4``, ``A100-80GB``, or ``H100:8``. A
    comma-separated value is treated as an ordered fallback list, e.g.
    ``H100,A100-80GB,L40S``.
    """

    if "," not in DEFAULT_GPU_SPEC:
        return DEFAULT_GPU_SPEC
    return [part.strip() for part in DEFAULT_GPU_SPEC.split(",") if part.strip()]


def secret_mounts() -> list[modal.Secret]:
    if HF_SECRET_NAME:
        return [modal.Secret.from_name(HF_SECRET_NAME)]
    hf_token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or os.environ.get("HF_API_KEY")
    )
    return [
        modal.Secret.from_dict(
            {
                "HF_API_KEY": os.environ.get("HF_API_KEY"),
                "HF_TOKEN": hf_token,
                "HF_HUB_TOKEN": os.environ.get("HF_HUB_TOKEN") or hf_token,
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            }
        )
    ]


artifacts = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(*PYTHON_PACKAGES)
    .env(
        {
            "HF_HOME": HF_CACHE_ROOT,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": REMOTE_REPO_ROOT,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(LOCAL_REPO_ROOT / "rlm"), remote_path=f"{REMOTE_REPO_ROOT}/rlm")
    .add_local_dir(str(LOCAL_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_REPO_ROOT}/scripts")
    .add_local_file(
        str(LOCAL_REPO_ROOT / "pyproject.toml"),
        remote_path=f"{REMOTE_REPO_ROOT}/pyproject.toml",
    )
)

app = modal.App(APP_NAME, image=image)


def python_cmd(script: str, *args: str) -> list[str]:
    return [PYTHON_BIN, script, *args]


def split_extra(extra: str) -> list[str]:
    return shlex.split(extra) if extra.strip() else []


def option_value(argv: list[str], option: str) -> str | None:
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        return None


def check_gate(argv: list[str]) -> None:
    if len(argv) < 2 or not argv[1].startswith("scripts/memory/"):
        return
    out = option_value(argv, "--out")
    if out is None:
        return
    path = Path(out)
    if not path.exists():
        return
    report = json.loads(path.read_text())
    script = Path(argv[1]).name
    if script in {"oracle_injection.py", "oracle_write_read.py"}:
        verdict = report.get("verdict", {})
        if not verdict.get("pass", False):
            raise RuntimeError(f"{script} gate failed: {verdict}")
    if script == "eval_acceptance.py":
        failures = [name for name in ("A1", "A4") if not report.get(name, {}).get("pass", False)]
        if failures:
            raise RuntimeError(f"acceptance gate failed: {', '.join(failures)}")


def commands_for_job(
    *,
    job: str,
    model: str,
    cache: str,
    skill: str,
    runs_dir: str,
    episodes: int,
    cache_batch_size: int,
    shard_size: int,
    steps: int,
    train_batch_size: int,
    d_k: int,
    lambda_kl: float,
    facts: int,
    a2_facts: int,
    a3: str,
    seed: int,
    dtype: str,
    overwrite_cache: bool,
    command: str,
    extra: str,
) -> list[list[str]]:
    extras = split_extra(extra)
    if job == "command":
        if not command.strip():
            raise ValueError("--command is required when --job command")
        return [shlex.split(command)]
    if job == "gpu-report":
        return [
            ["nvidia-smi"],
            [
                PYTHON_BIN,
                "-c",
                (
                    "import json, torch; "
                    "print(json.dumps({"
                    "'cuda_available': torch.cuda.is_available(), "
                    "'device_count': torch.cuda.device_count(), "
                    "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]"
                    "}, indent=2))"
                ),
            ],
        ]
    if job == "hf-report":
        model_literal = json.dumps(model)
        return [
            [
                PYTHON_BIN,
                "-c",
                (
                    "import json, os; "
                    "from huggingface_hub import model_info, whoami; "
                    "token = os.environ.get('HF_TOKEN') or os.environ.get('HF_HUB_TOKEN'); "
                    f"model = {model_literal}; "
                    "info = whoami(token=token) if token else {}; "
                    "meta = model_info(model, token=token) if token else None; "
                    "print(json.dumps({"
                    "'hf_token_present': bool(token), "
                    "'hf_name': info.get('name'), "
                    "'hf_org_count': len(info.get('orgs', [])), "
                    "'model': model, "
                    "'model_accessible': meta is not None, "
                    "'model_sha': getattr(meta, 'sha', None)"
                    "}, indent=2))"
                ),
            ]
        ]
    if job == "oracle-injection":
        return [
            python_cmd(
                "scripts/memory/oracle_injection.py",
                "--model",
                model,
                "--facts",
                str(facts),
                "--seed",
                str(seed),
                "--dtype",
                dtype,
                "--out",
                f"{runs_dir}/oracle_injection.json",
                *extras,
            )
        ]
    if job == "oracle-write-read":
        return [
            python_cmd(
                "scripts/memory/oracle_write_read.py",
                "--model",
                model,
                "--facts",
                str(facts),
                "--d-k",
                str(d_k),
                "--seed",
                str(seed),
                "--dtype",
                dtype,
                "--out",
                f"{runs_dir}/oracle_write_read.json",
                *extras,
            )
        ]
    if job == "build-cache":
        argv = python_cmd(
            "scripts/memory/build_cache.py",
            "--model",
            model,
            "--episodes",
            str(episodes),
            "--batch-size",
            str(cache_batch_size),
            "--shard-size",
            str(shard_size),
            "--seed",
            str(seed),
            "--dtype",
            dtype,
            "--out",
            cache,
            *extras,
        )
        if overwrite_cache:
            argv.append("--overwrite")
        return [argv]
    if job == "train-skill":
        return [
            python_cmd(
                "scripts/memory/train_skill.py",
                "--cache",
                cache,
                "--out",
                skill,
                "--steps",
                str(steps),
                "--batch-size",
                str(train_batch_size),
                "--d-k",
                str(d_k),
                "--lambda-kl",
                str(lambda_kl),
                "--seed",
                str(seed),
                *extras,
            )
        ]
    if job == "eval-acceptance":
        argv = python_cmd(
            "scripts/memory/eval_acceptance.py",
            "--model",
            model,
            "--skill",
            skill,
            "--facts",
            str(facts),
            "--a2-facts",
            str(a2_facts),
            "--seed",
            str(seed + 1234),
            "--dtype",
            dtype,
            "--out",
            f"{runs_dir}/acceptance.json",
            *extras,
        )
        if a3:
            argv.extend(["--a3", a3])
        return [argv]
    if job == "acceptance-ladder":
        ladder: list[list[str]] = []
        for step in (
            "oracle-injection",
            "oracle-write-read",
            "build-cache",
            "train-skill",
            "eval-acceptance",
        ):
            ladder.extend(
                commands_for_job(
                    job=step,
                    model=model,
                    cache=cache,
                    skill=skill,
                    runs_dir=runs_dir,
                    episodes=episodes,
                    cache_batch_size=cache_batch_size,
                    shard_size=shard_size,
                    steps=steps,
                    train_batch_size=train_batch_size,
                    d_k=d_k,
                    lambda_kl=lambda_kl,
                    facts=facts,
                    a2_facts=a2_facts,
                    a3=a3,
                    seed=seed,
                    dtype=dtype,
                    overwrite_cache=overwrite_cache,
                    command="",
                    extra="",
                )
            )
        return ladder
    raise ValueError(f"unknown job {job!r}")


def run_commands(commands: list[list[str]]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_REPO_ROOT
    Path(ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(f"{ARTIFACT_ROOT}/runs").mkdir(parents=True, exist_ok=True)
    Path(f"{ARTIFACT_ROOT}/cache").mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    for argv in commands:
        print(f"\n$ {shlex.join(argv)}", flush=True)
        step_started = time.monotonic()
        result = subprocess.run(argv, cwd=REMOTE_REPO_ROOT, env=env, check=False)
        elapsed = time.monotonic() - step_started
        completed.append(
            {
                "argv": argv,
                "returncode": result.returncode,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        artifacts.commit()
        hf_cache.commit()
        if result.returncode != 0:
            raise RuntimeError(f"command failed with exit code {result.returncode}: {argv}")
        check_gate(argv)

    return {
        "commands": completed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "artifact_volume": ARTIFACT_VOLUME_NAME,
        "artifact_root": ARTIFACT_ROOT,
        "hf_cache_volume": HF_CACHE_VOLUME_NAME,
    }


@app.function(
    gpu=gpu_spec(),
    timeout=DEFAULT_TIMEOUT,
    volumes={ARTIFACT_ROOT: artifacts, HF_CACHE_ROOT: hf_cache},
    secrets=secret_mounts(),
)
def run_remote(commands: list[list[str]]) -> dict[str, Any]:
    return run_commands(commands)


@app.local_entrypoint()
def main(
    job: str = "gpu-report",
    model: str = "google/gemma-3-1b-it",
    cache: str = f"{ARTIFACT_ROOT}/cache/gemma1b_v1",
    skill: str = f"{ARTIFACT_ROOT}/runs/skill_v1.pt",
    runs_dir: str = f"{ARTIFACT_ROOT}/runs",
    episodes: int = 20_000,
    cache_batch_size: int = 16,
    shard_size: int = 1_000,
    steps: int = 4_000,
    train_batch_size: int = 32,
    d_k: int = 512,
    lambda_kl: float = 0.5,
    facts: int = 24,
    a2_facts: int = 12,
    a3: str = "",
    seed: int = 0,
    dtype: str = "bfloat16",
    overwrite_cache: bool = False,
    command: str = "",
    extra: str = "",
) -> None:
    commands = commands_for_job(
        job=job,
        model=model,
        cache=cache,
        skill=skill,
        runs_dir=runs_dir,
        episodes=episodes,
        cache_batch_size=cache_batch_size,
        shard_size=shard_size,
        steps=steps,
        train_batch_size=train_batch_size,
        d_k=d_k,
        lambda_kl=lambda_kl,
        facts=facts,
        a2_facts=a2_facts,
        a3=a3,
        seed=seed,
        dtype=dtype,
        overwrite_cache=overwrite_cache,
        command=command,
        extra=extra,
    )
    print(f"Modal app={APP_NAME} gpu={gpu_spec()} artifact_volume={ARTIFACT_VOLUME_NAME}")
    result = run_remote.remote(commands)
    print(f"\nremote result: {result}")
