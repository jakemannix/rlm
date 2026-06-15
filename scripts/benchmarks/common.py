"""Shared helpers for the Titans / RLM benchmark scripts.

The benchmark scripts target a tiny open-weights model (Gemma 3 1B-IT by
default) and try to be cheap enough to run end-to-end on a single Colab
GPU.  All scripts share the same CLI surface:

    python scripts/benchmarks/<bench>.py \
        --backend titans_hf --model google/gemma-3-1b-it \
        --num-samples 20 --memory-flavor titans

Set ``--backend openai`` (or ``vllm``, ``openrouter``, ...) to compare
against an API-based baseline without the parametric memory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlm import RLM
from rlm.memory.config import MemoryConfig


@dataclass
class BenchmarkResult:
    """One bench-run's aggregated metrics."""

    name: str
    backend: str
    model: str
    n_examples: int
    n_correct: int
    n_errors: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time_seconds: float = 0.0
    per_example: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        if self.n_examples == 0:
            return 0.0
        return self.n_correct / self.n_examples

    def summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "model": self.model,
            "accuracy": self.accuracy,
            "n_examples": self.n_examples,
            "n_correct": self.n_correct,
            "n_errors": self.n_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_time_seconds": self.total_time_seconds,
            "extras": self.extras,
        }

    def save(self, output_dir: str) -> Path:
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = Path(output_dir) / f"{self.name}-{stamp}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the shared CLI surface to a parser."""
    p.add_argument("--backend", default="openai", help="RLM backend name.")
    p.add_argument("--model", default=None, help="Model id passed to the backend (model_name).")
    p.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible APIs (e.g. vLLM, OpenRouter).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of benchmark examples to run.",
    )
    p.add_argument(
        "--memory-flavor",
        choices=["titans", "miras"],
        default="titans",
        help="Memory module flavor (only used for the titans_hf backend).",
    )
    p.add_argument(
        "--retention",
        choices=["l2", "l1", "huber", "kl"],
        default="l2",
        help="Miras retention surrogate (only used when --memory-flavor=miras).",
    )
    p.add_argument(
        "--memory-hidden-dim",
        type=int,
        default=256,
        help="Hidden dim of the Titans memory MLP.",
    )
    p.add_argument(
        "--memory-layers",
        type=int,
        default=2,
        help="Depth of the Titans memory MLP.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="RLM max_iterations (REPL iteration budget).",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="RLM recursion max_depth.",
    )
    p.add_argument(
        "--output-dir",
        default="./bench_results",
        help="Where to drop per-run JSON result files.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for example selection.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Stream the RLM trajectory to the console.",
    )


def build_rlm(args: argparse.Namespace, **extra_rlm_kwargs: Any) -> RLM:
    """Build an RLM from a parsed argparse namespace.

    Extra kwargs (``custom_tools``, ``environment``, ...) are forwarded
    to :class:`rlm.RLM` so individual benches can register tools.
    """
    backend_kwargs: dict[str, Any] = {}
    if args.model is not None:
        backend_kwargs["model_name"] = args.model
    if args.base_url is not None:
        backend_kwargs["base_url"] = args.base_url
    if args.backend == "titans_hf":
        backend_kwargs.update(
            flavor=args.memory_flavor,
            memory_config=MemoryConfig(
                key_dim=0,
                value_dim=0,
                hidden_dim=args.memory_hidden_dim,
                n_layers=args.memory_layers,
                retention=args.retention if args.memory_flavor == "miras" else "l2",
            ),
        )
    return RLM(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        verbose=args.verbose,
        **extra_rlm_kwargs,
    )
