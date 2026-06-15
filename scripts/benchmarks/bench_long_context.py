"""
Long-context "needle in a haystack" benchmark.

This is the natural showcase for the Titans/Miras parametric memory: a
long passage of distractor text is streamed through the model, and a
single "needle" sentence is hidden at a known position.  The model is
then asked a question whose answer only appears in the needle.

For backends that don't have a long context window, the parametric
memory module should still recover the answer because the needle's
``(key, value)`` pair was stored in the memory MLP.

Even for backends with long context, this benchmark is a sanity check
that the model can attend to distant tokens.
"""

from __future__ import annotations

import argparse
import random
import time
import traceback

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

DISTRACTORS = [
    "The sky was overcast that morning, the kind of grey that promises rain by evening.",
    "She thought about how the river had risen each spring for as long as she could remember.",
    "Outside, the wind picked up, rattling the loose pane in the kitchen window.",
    "He poured the coffee carefully, leaving just enough room at the top for cream.",
    "The bookshop on the corner had been there since the 1920s and smelled of dust and leather.",
    "Around three in the afternoon the cats wandered in from the garden, looking offended.",
    "Generations of fishermen had set out from this harbour, returning before the storms.",
    "The bell from the church across the square struck the hour, slow and uncertain.",
    "She measured the flour by eye, the way her mother had taught her many summers ago.",
    "A small dog barked from somewhere behind the stone wall but no one ever saw it.",
]


SECRETS = [
    ("The magic codeword for project Phoenix is BLUEFOX-77.", "BLUEFOX-77"),
    ("Dr. Tanaka's office is on the 14th floor, room 1408.", "1408"),
    ("The combination to the safe is 04-22-19.", "04-22-19"),
    ("Captain Reyes was promoted to commodore on 7 March 2024.", "7 March 2024"),
    ("The recipe calls for exactly 237 grams of dark chocolate.", "237"),
    ("Annual revenue from the Helios project was $1.42 million.", "$1.42 million"),
    ("The encryption key for vault A is named CASCADE_ALPHA.", "CASCADE_ALPHA"),
    ("Customer #88412 has overdue invoices totalling $5,309.50.", "$5,309.50"),
]


QUESTION_TEMPLATES = {
    "BLUEFOX-77": "What is the magic codeword for project Phoenix?",
    "1408": "Which room is Dr. Tanaka's office in?",
    "04-22-19": "What is the combination to the safe?",
    "7 March 2024": "When was Captain Reyes promoted to commodore?",
    "237": "How many grams of dark chocolate does the recipe call for?",
    "$1.42 million": "What was the annual revenue from the Helios project?",
    "CASCADE_ALPHA": "What is the encryption key for vault A?",
    "$5,309.50": "How much do customer #88412's overdue invoices total?",
}


def build_haystack(rng: random.Random, n_distractors: int, needle: str, position: float) -> str:
    """Build a haystack: many distractor sentences with the needle inserted."""
    insert_at = int(position * n_distractors)
    lines: list[str] = []
    for i in range(n_distractors):
        if i == insert_at:
            lines.append(needle)
        lines.append(rng.choice(DISTRACTORS))
    return " ".join(lines)


PROMPT_TEMPLATE = (
    "You will be given a long passage of text.  Read it carefully and then "
    "answer the question.  Use the REPL if helpful (the passage is "
    "available as the variable ``context``).  Set answer['content'] to "
    "the exact answer (no extra words) and answer['ready'] = True.\n\n"
    "QUESTION: {question}\n\n"
    "PASSAGE START\n{passage}\nPASSAGE END\n"
)


def main() -> None:
    p = argparse.ArgumentParser(description="Long-context needle-in-haystack benchmark.")
    add_common_args(p)
    p.add_argument(
        "--haystack-size",
        type=int,
        default=200,
        help="Number of distractor sentences in the haystack.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    rlm = build_rlm(args)

    secrets = list(SECRETS)
    rng.shuffle(secrets)
    secrets = secrets[: args.num_samples]

    result = BenchmarkResult(
        name="long_context_niah",
        backend=args.backend,
        model=args.model or "",
        n_examples=len(secrets),
        n_correct=0,
        extras={"haystack_size": args.haystack_size},
    )

    start = time.time()
    for i, (needle, gold) in enumerate(secrets):
        position = rng.random()
        passage = build_haystack(rng, args.haystack_size, needle, position)
        question = QUESTION_TEMPLATES[gold]
        prompt = PROMPT_TEMPLATE.format(question=question, passage=passage)

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
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        elapsed = time.time() - t0
        correct = err is None and gold.lower().strip() in (response or "").lower()
        result.per_example.append(
            {
                "index": i,
                "needle_position": position,
                "gold": gold,
                "response": response,
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
            f"[{i + 1}/{len(secrets)}] gold={gold!r} pos={position:.2f} "
            f"correct={correct} ({elapsed:.1f}s)"
        )

    result.total_time_seconds = time.time() - start
    print(
        f"\n=== long_context_niah accuracy={result.accuracy:.3f} "
        f"({result.n_correct}/{result.n_examples}) "
        f"errors={result.n_errors} ===",
    )
    path = result.save(args.output_dir)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
