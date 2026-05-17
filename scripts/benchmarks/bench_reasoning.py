"""
Mathematical / multi-step reasoning benchmark.

Runs a sample of GSM8K-style word problems through the RLM and checks the
final numeric answer.  GSM8K is loaded via the ``datasets`` library when
available; otherwise a small built-in problem set is used so the script
remains runnable in any environment.

Usage
-----
    python scripts/benchmarks/bench_reasoning.py --backend titans_hf \
        --model google/gemma-3-1b-it --num-samples 20

    python scripts/benchmarks/bench_reasoning.py --backend openai \
        --model gpt-5-nano --num-samples 20
"""

from __future__ import annotations

import argparse
import random
import re
import time
import traceback
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

# Tiny built-in fallback set drawn in the style of GSM8K so the script can
# run without internet access.  All gold answers are integers.
_FALLBACK_PROBLEMS: list[dict[str, Any]] = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold half "
            "as many clips in May. How many clips did Natalia sell altogether in "
            "April and May?"
        ),
        "answer": 72,
    },
    {
        "question": (
            "A train travels 60 miles in the first hour, 80 miles in the second hour "
            "and 100 miles in the third hour. How many miles does it travel in total?"
        ),
        "answer": 240,
    },
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
            "morning and bakes muffins for her friends every day with four. She sells "
            "the remainder at the farmers' market for $2 per fresh duck egg. How much "
            "in dollars does she make every day at the farmers' market?"
        ),
        "answer": 18,
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
            "many bolts in total does it take?"
        ),
        "answer": 3,
    },
    {
        "question": (
            "Josh decides to try flipping a house.  He buys a house for $80,000 and "
            "then puts in $50,000 in repairs.  This increased the value of the house "
            "by 150%.  How much profit did he make?"
        ),
        "answer": 70000,
    },
    {
        "question": (
            "James decides to run 3 sprints 3 times a week.  He runs 60 meters each "
            "sprint.  How many total meters does he run a week?"
        ),
        "answer": 540,
    },
    {
        "question": (
            "Toulouse has twice as many sheep as Charleston. Charleston has 4 times "
            "as many sheep as Seattle. How many sheep do Toulouse, Charleston, and "
            "Seattle have together if Seattle has 20 sheep?"
        ),
        "answer": 260,
    },
    {
        "question": (
            "A box contains 24 chocolates. If each chocolate weighs 15 grams, what is "
            "the total weight of the chocolates in the box, in grams?"
        ),
        "answer": 360,
    },
    {
        "question": (
            "A farmer has 30 cows and 50 chickens. Each cow has 4 legs and each "
            "chicken has 2 legs. How many legs are there in total?"
        ),
        "answer": 220,
    },
    {
        "question": (
            "Maria buys 5 packs of pencils. Each pack contains 12 pencils. She gives "
            "away 17 pencils to her classmates. How many pencils does Maria have left?"
        ),
        "answer": 43,
    },
]


def load_gsm8k(n: int, seed: int) -> list[dict[str, Any]]:
    """Try to load GSM8K from the ``datasets`` library; fallback otherwise."""
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), min(n, len(ds)))
        out = []
        for i in idxs:
            row = ds[i]
            ans = row["answer"].split("####")[-1].strip().replace(",", "")
            try:
                gold = int(ans)
            except ValueError:
                gold = float(ans)
            out.append({"question": row["question"], "answer": gold})
        return out
    except Exception:
        rng = random.Random(seed)
        pool = list(_FALLBACK_PROBLEMS)
        rng.shuffle(pool)
        return pool[:n]


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")


def extract_numeric_answer(response: str) -> float | None:
    """Pull the last number out of the model's response."""
    if not response:
        return None
    # Prefer an explicit "answer = X" or "Final answer: X" pattern.
    for pat in (
        r"final\s+answer\s*[:=]\s*(-?\d+(?:[.,]\d+)?)",
        r"answer\s*[:=]\s*(-?\d+(?:[.,]\d+)?)",
        r"\\boxed\{\s*(-?\d+(?:[.,]\d+)?)\s*\}",
    ):
        m = re.search(pat, response, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    numbers = _NUMBER_RE.findall(response)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", "").replace(" ", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PROMPT_TEMPLATE = (
    "Solve the following grade-school math word problem.  Reason carefully, "
    "use Python in the REPL when helpful, and finish by setting "
    "answer['content'] to the final numeric answer (no units) and "
    "answer['ready'] = True.\n\n"
    "Problem:\n{question}"
)


def main() -> None:
    p = argparse.ArgumentParser(description="GSM8K-style reasoning benchmark.")
    add_common_args(p)
    args = p.parse_args()

    problems = load_gsm8k(args.num_samples, args.seed)
    rlm = build_rlm(args)
    result = BenchmarkResult(
        name="reasoning_gsm8k",
        backend=args.backend,
        model=args.model or "",
        n_examples=len(problems),
        n_correct=0,
    )

    start = time.time()
    for i, problem in enumerate(problems):
        prompt = PROMPT_TEMPLATE.format(question=problem["question"])
        t0 = time.time()
        response = ""
        err = None
        completion = None
        try:
            completion = rlm.completion(prompt)
            response = completion.response or ""
        except (
            BudgetExceededError,
            TimeoutExceededError,
            TokenLimitExceededError,
            ErrorThresholdExceededError,
        ) as e:
            err = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 - report any backend errors
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        elapsed = time.time() - t0
        predicted = extract_numeric_answer(response)
        gold = float(problem["answer"])
        correct = predicted is not None and abs(predicted - gold) < 1e-6
        result.per_example.append(
            {
                "index": i,
                "question": problem["question"],
                "gold": gold,
                "predicted": predicted,
                "correct": correct,
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
            f"[{i + 1}/{len(problems)}] gold={gold} predicted={predicted} "
            f"correct={correct} ({elapsed:.1f}s)"
        )

    result.total_time_seconds = time.time() - start
    print(
        f"\n=== reasoning_gsm8k accuracy={result.accuracy:.3f} "
        f"({result.n_correct}/{result.n_examples}) "
        f"errors={result.n_errors} ===",
    )
    path = result.save(args.output_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
