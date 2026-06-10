"""Acceptance harness — LIVE A1/A2/A4 (and optionally A3) on the real base with
a trained skill, using fresh :class:`MemorySession` objects so the eval regime
is exactly the brief's: ingest → persist → context gone → recall.

    python scripts/memory/eval_acceptance.py --model google/gemma-3-1b-it \
        --skill runs/skill_v1.pt --out runs/acceptance.json
    # capacity curve:
    python scripts/memory/eval_acceptance.py ... --a3 1,2,4,8,16,32,64,128

Gates (design doc §4): A1 mean first-token lift ≥ 3 nats AND top-5 ≥ 0.8 AND
controls |lift| ≤ 0.5; A2 neutral-text KL ≤ 0.02 nats/token and unrelated-query
top-1 unchanged ≥ 0.9; A4 = A1 retained through a disk round-trip into a fresh
session.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from rlm.memory.episodes import EpisodeGenerator, EpisodeText
from rlm.memory.session import MemorySession
from rlm.memory.skill import MemorySkill

NEUTRAL = (
    "The library reading room stayed open until nine on weekdays. Visitors left"
    " their coats on the rack beside the entrance, and the catalog terminals"
    " hummed quietly near the reference desk."
)


def recall_episode(gen: EpisodeGenerator, fact) -> EpisodeText:
    return EpisodeText(
        "recall",
        [gen.session_with_facts([fact])],
        fact.verbatim_prompt,
        fact.answer,
        None,
        [fact],
        fact,
    )


def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def query_of(fact, paraphrase: bool) -> str:
    """A1b (content-addressable): query with the natural question instead of the
    verbatim prefix; falls back to verbatim when the fact carries no paraphrase."""
    if paraphrase and getattr(fact, "paraphrase_prompts", None):
        return fact.paraphrase_prompts[0]
    return fact.verbatim_prompt


def run_a1_a4(
    model, tok, skill, gen: EpisodeGenerator, n_facts: int, device: str, paraphrase: bool = False
) -> dict:
    facts = [gen.make_fact() for _ in range(n_facts)]
    mem_lift, mem_top5, empty_lift, wrong_lift, a4_lift = [], [], [], [], []
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(facts):
            q = query_of(f, paraphrase)  # ingest the statement; query the question (A1b) or prefix
            ingest = MemorySession(model, tok, skill, device=device)
            ingest.ingest_episode(recall_episode(gen, f))
            r_same = ingest.score_answer(q, f.answer)

            path = str(Path(td) / f"s{i}.pt")
            ingest.save(path)
            fresh = MemorySession(model, tok, skill, device=device)  # A4: fresh session
            fresh.load(path)
            r = fresh.score_answer(q, f.answer)
            r0 = fresh.score_answer(q, f.answer, use_store=False)

            wrong = MemorySession(model, tok, skill, device=device)
            wrong.ingest_episode(recall_episode(gen, facts[(i + 1) % n_facts]))
            rw = wrong.score_answer(q, f.answer)

            mem_lift.append(r_same["first_token_lift_nats"])
            a4_lift.append(r["first_token_lift_nats"])
            mem_top5.append(float(r_same["first_token_top5"]))  # A1 = in-session, not post-reload
            empty_lift.append(r0["first_token_lift_nats"])
            wrong_lift.append(rw["first_token_lift_nats"])

    a1 = {
        "mean_first_lift_nats": mean(
            mem_lift
        ),  # in-session lift (A1); A4 measures the reload retention
        "top5_frac": mean(mem_top5),
        "empty_control_lift": mean(empty_lift),
        "wrong_fact_control_lift": mean(wrong_lift),
    }
    a1["pass"] = (
        a1["mean_first_lift_nats"] >= 3.0
        and a1["top5_frac"] >= 0.8
        and abs(a1["empty_control_lift"]) < 1e-6
        and abs(a1["wrong_fact_control_lift"]) <= 0.5
    )
    a4 = {
        "in_session_lift": mean(mem_lift),
        "after_reload_lift": mean(a4_lift),
        "retention_frac": mean(a4_lift) / mem_lift_safe(mean(mem_lift)),
    }
    a4["pass"] = bool(a1["pass"]) and a4["retention_frac"] >= 0.9
    return {"A1": a1, "A4": a4}


def mem_lift_safe(x: float) -> float:
    return x if abs(x) > 1e-9 else 1e-9


def run_a2(
    model, tok, skill, gen: EpisodeGenerator, n: int, device: str, paraphrase: bool = False
) -> dict:
    kls, unchanged = [], []
    for _ in range(n):
        f, other = gen.make_fact(), gen.make_fact()
        sess = MemorySession(model, tok, skill, device=device)
        sess.ingest_episode(recall_episode(gen, f))
        kls.append(sess.neutral_kl(NEUTRAL))
        with torch.no_grad():
            r_mem = sess.score_answer(query_of(other, paraphrase), other.answer)
        unchanged.append(float(abs(r_mem["first_token_lift_nats"]) <= 0.5))
    a2 = {"neutral_kl_nats_per_tok": mean(kls), "unrelated_query_unchanged_frac": mean(unchanged)}
    a2["pass"] = (
        a2["neutral_kl_nats_per_tok"] <= 0.02 and a2["unrelated_query_unchanged_frac"] >= 0.9
    )
    return a2


def run_a3(
    model, tok, skill, gen: EpisodeGenerator, ns: list[int], device: str, paraphrase: bool = False
) -> list[dict]:
    curve = []
    for n in ns:
        facts = [gen.make_fact() for _ in range(n)]
        sess = MemorySession(model, tok, skill, device=device)
        sess.ingest_episode(
            EpisodeText("multifact", [gen.session_with_facts(facts)], "", None, None, facts, None)
        )
        lifts, top5 = [], []
        for f in facts:
            r = sess.score_answer(query_of(f, paraphrase), f.answer)
            lifts.append(r["first_token_lift_nats"])
            top5.append(float(r["first_token_top5"]))
        point = {"n_facts": n, "mean_first_lift_nats": mean(lifts), "top5_frac": mean(top5)}
        curve.append(point)
        print(
            f"A3 n={n:>4}: lift {point['mean_first_lift_nats']:.2f} nats, top5 {point['top5_frac']:.2f}"
        )
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--facts", type=int, default=24)
    ap.add_argument("--a2-facts", type=int, default=12)
    ap.add_argument("--a3", default=None, help="comma-separated fact counts, e.g. 1,2,4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=1234)  # disjoint from training seed
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="runs/acceptance.json")
    ap.add_argument(
        "--real-corpus",
        action="store_true",
        help="milestone-3: evaluate on held-out REAL facts (RealCorpusGenerator)",
    )
    ap.add_argument("--real-split", default="eval", choices=["train", "eval"])
    ap.add_argument("--squad", action="store_true")
    ap.add_argument(
        "--paraphrase",
        action="store_true",
        help="A1b: query with the natural question, not the verbatim prefix (content-addressable)",
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        .to(args.device)
        .eval()
    )
    skill = MemorySkill.load(args.skill, device=args.device)
    if args.real_corpus:
        from rlm.memory.real_corpus import RealCorpusGenerator, load_squad_records

        records = load_squad_records() if args.squad else None
        gen = RealCorpusGenerator(records=records, split=args.real_split, seed=args.seed)
        print(
            f"real-corpus eval: split={args.real_split}, {len(gen.records)} held-out records, "
            f"paraphrase={args.paraphrase}"
        )
    else:
        gen = EpisodeGenerator(seed=args.seed)

    p = args.paraphrase
    report = run_a1_a4(model, tok, skill, gen, args.facts, args.device, paraphrase=p)
    report["A2"] = run_a2(model, tok, skill, gen, args.a2_facts, args.device, paraphrase=p)
    if args.a3:
        report["A3_curve"] = run_a3(
            model, tok, skill, gen, [int(x) for x in args.a3.split(",")], args.device, paraphrase=p
        )

    for name in ("A1", "A2", "A4"):
        print(f"{name}: {'PASS' if report[name]['pass'] else 'FAIL'}  {json.dumps(report[name])}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
