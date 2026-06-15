"""
Code-generation benchmark (HumanEval-style).

Loads a sample of HumanEval problems via the ``datasets`` library when
available; falls back to a small built-in set otherwise.  The RLM is
asked to implement the function and is graded by running the canonical
HumanEval-style ``check(candidate)`` test suite *outside* the RLM's REPL
(in a fresh subprocess for safety).

Usage
-----
    python scripts/benchmarks/bench_codegen.py --backend titans_hf \
        --model google/gemma-3-1b-it --num-samples 5
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any

from rlm.utils.exceptions import (
    BudgetExceededError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from scripts.benchmarks.common import (
    BenchmarkResult,
    add_common_args,
    build_rlm,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class CodeProblem:
    task_id: str
    prompt: str
    canonical: str
    tests: str
    entry_point: str


FALLBACK_PROBLEMS: list[CodeProblem] = [
    CodeProblem(
        task_id="local/sum_to_n",
        prompt=(
            "def sum_to_n(n: int) -> int:\n"
            '    """Return the sum of integers from 1 to n inclusive."""\n'
        ),
        canonical="    return n * (n + 1) // 2\n",
        tests=(
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(5) == 15\n"
            "    assert candidate(100) == 5050\n"
        ),
        entry_point="sum_to_n",
    ),
    CodeProblem(
        task_id="local/is_palindrome",
        prompt=(
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True iff s reads the same forwards and backwards.\n'
            "    Comparison is case-insensitive and ignores non-alphanumeric chars.\n"
            '    """\n'
        ),
        canonical=(
            "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        tests=(
            "def check(candidate):\n"
            "    assert candidate('') is True\n"
            "    assert candidate('a') is True\n"
            "    assert candidate('A man, a plan, a canal: Panama') is True\n"
            "    assert candidate('hello') is False\n"
        ),
        entry_point="is_palindrome",
    ),
    CodeProblem(
        task_id="local/fizzbuzz",
        prompt=(
            "def fizzbuzz(n: int) -> list[str]:\n"
            '    """Return the FizzBuzz output for 1..n inclusive."""\n'
        ),
        canonical=(
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0: out.append('FizzBuzz')\n"
            "        elif i % 3 == 0: out.append('Fizz')\n"
            "        elif i % 5 == 0: out.append('Buzz')\n"
            "        else: out.append(str(i))\n"
            "    return out\n"
        ),
        tests=(
            "def check(candidate):\n"
            "    assert candidate(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
            "    assert candidate(15)[-1] == 'FizzBuzz'\n"
            "    assert len(candidate(100)) == 100\n"
        ),
        entry_point="fizzbuzz",
    ),
    CodeProblem(
        task_id="local/fib",
        prompt=(
            "def fib(n: int) -> int:\n"
            '    """Return the n-th Fibonacci number (fib(0)=0, fib(1)=1)."""\n'
        ),
        canonical=(
            "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
        ),
        tests=(
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(10) == 55\n"
            "    assert candidate(20) == 6765\n"
        ),
        entry_point="fib",
    ),
    CodeProblem(
        task_id="local/longest_word",
        prompt=(
            "def longest_word(words: list[str]) -> str:\n"
            '    """Return the longest word.  Ties broken by earliest occurrence."""\n'
        ),
        canonical=(
            "    best = ''\n"
            "    for w in words:\n"
            "        if len(w) > len(best):\n"
            "            best = w\n"
            "    return best\n"
        ),
        tests=(
            "def check(candidate):\n"
            "    assert candidate(['a', 'bb', 'ccc']) == 'ccc'\n"
            "    assert candidate(['cat', 'dog', 'fox']) == 'cat'\n"
            "    assert candidate([]) == ''\n"
        ),
        entry_point="longest_word",
    ),
]


def load_humaneval(n: int, seed: int) -> list[CodeProblem]:
    try:
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split="test")
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), min(n, len(ds)))
        out = []
        for i in idxs:
            row = ds[i]
            out.append(
                CodeProblem(
                    task_id=row["task_id"],
                    prompt=row["prompt"],
                    canonical=row["canonical_solution"],
                    tests=row["test"],
                    entry_point=row["entry_point"],
                )
            )
        return out
    except Exception:
        rng = random.Random(seed)
        pool = list(FALLBACK_PROBLEMS)
        rng.shuffle(pool)
        return pool[:n]


# ---------------------------------------------------------------------------
# Extracting code from the model response
# ---------------------------------------------------------------------------


_FENCED_BLOCK = re.compile(r"```(?:python)?\s*\n([\s\S]+?)```", re.MULTILINE)


def extract_code(response: str, entry_point: str) -> str:
    """Pull out the most plausible candidate code from the model's response."""
    if not response:
        return ""
    # 1. Fenced ```python blocks - keep the one that mentions entry_point.
    blocks = _FENCED_BLOCK.findall(response)
    for b in blocks:
        if entry_point in b:
            return b
    if blocks:
        return blocks[-1]
    # 2. Fallback: the raw response.
    return response


# ---------------------------------------------------------------------------
# Sandboxed test runner
# ---------------------------------------------------------------------------


def _run_in_subprocess(code: str, entry_point: str, tests: str, q):  # type: ignore[no-untyped-def]
    """Worker for ``multiprocessing.Process`` - executes the candidate and tests."""
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)
        if entry_point not in namespace:
            q.put({"passed": False, "error": f"{entry_point} not defined"})
            return
        exec(tests, namespace)
        namespace["check"](namespace[entry_point])
        q.put({"passed": True, "error": None})
    except Exception as e:
        q.put({"passed": False, "error": f"{type(e).__name__}: {e}"})


def run_tests(code: str, entry_point: str, tests: str, timeout: float = 10.0) -> dict[str, Any]:
    """Run ``tests`` against ``code`` in a separate process with a timeout."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_run_in_subprocess, args=(code, entry_point, tests, q), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        return {"passed": False, "error": "timeout"}
    try:
        return q.get_nowait()
    except Exception:
        return {"passed": False, "error": "no result"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PROMPT_TEMPLATE = (
    "Implement the following Python function.  You may use the REPL to test "
    "ideas, but the final answer must be a single self-contained ```python "
    "code block defining ``{entry_point}``.  Set answer['content'] to that "
    "block and answer['ready'] = True.\n\n{prompt}"
)


def main() -> None:
    p = argparse.ArgumentParser(description="HumanEval-style code-generation benchmark.")
    add_common_args(p)
    p.add_argument(
        "--test-timeout",
        type=float,
        default=10.0,
        help="Seconds before a candidate's test execution is killed.",
    )
    args = p.parse_args()

    problems = load_humaneval(args.num_samples, args.seed)
    rlm = build_rlm(args)

    result = BenchmarkResult(
        name="codegen_humaneval",
        backend=args.backend,
        model=args.model or "",
        n_examples=len(problems),
        n_correct=0,
    )

    start = time.time()
    for i, prob in enumerate(problems):
        t0 = time.time()
        response = ""
        err = None
        completion = None
        try:
            prompt = PROMPT_TEMPLATE.format(prompt=prob.prompt, entry_point=prob.entry_point)
            completion = rlm.completion(prompt)
            response = completion.response or ""
        except (
            BudgetExceededError,
            TimeoutExceededError,
            TokenLimitExceededError,
            ErrorThresholdExceededError,
        ) as e:
            err = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        code = extract_code(response, prob.entry_point)
        test_result = (
            run_tests(code, prob.entry_point, prob.tests, args.test_timeout)
            if err is None
            else {"passed": False, "error": err}
        )
        elapsed = time.time() - t0
        correct = bool(test_result.get("passed"))
        result.per_example.append(
            {
                "task_id": prob.task_id,
                "code": code,
                "correct": correct,
                "test_error": test_result.get("error"),
                "elapsed_seconds": elapsed,
                "error": err,
            }
        )
        if correct:
            result.n_correct += 1
        if err is not None:
            result.n_errors += 1
        if completion is not None and completion.usage_summary is not None:
            result.total_input_tokens += completion.usage_summary.total_input_tokens
            result.total_output_tokens += completion.usage_summary.total_output_tokens
        print(
            f"[{i + 1}/{len(problems)}] {prob.task_id} correct={correct} "
            f"err={test_result.get('error')!r} ({elapsed:.1f}s)"
        )

    result.total_time_seconds = time.time() - start
    print(
        f"\n=== codegen_humaneval pass@1={result.accuracy:.3f} "
        f"({result.n_correct}/{result.n_examples}) "
        f"errors={result.n_errors} ===",
    )
    path = result.save(args.output_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
