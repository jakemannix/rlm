"""Milestone-3 items 3–5: is the memory *useful* on real text, vs honest baselines,
generatively, and at deployment scale?  Runs on a trained skill + the real base.

All facts come from the HELD-OUT entity split of :class:`RealCorpusGenerator`
(``--squad`` for scale), so every fact is about an entity never seen in meta-training.
A fixed seeded bank is drawn once and reused across every measurement (stats protocol).

Item 3 — baselines on identical inputs (per fact, ingest the statement in its own session):
  floor     : base, no store                         (score_answer use_store=False)
  memory    : base + the persisted store             (score_answer use_store=True)
  in_context: base with the statement IN the prompt  (the strong ceiling)
  retrieval : embed statements + query, top-1 by cosine, in-context with the hit
Item 4 — generative exact-match (generate_greedy, gold ∈ output) + a no-regression
  check (neutral-text KL with the full N-fact store loaded must stay small).
Item 5 — scale + persistence: ingest the bank one fact at a time, save→reload between
  each (the deployment regime), and at checkpoints query ALL facts so far, reporting
  worst-case recall (fraction at top-5) + mean lift as the store grows.

    python scripts/memory/eval_real.py --skill runs/skill_real.pt --squad \
        --scale 8,16,32,64,128 --out runs/real_eval.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from rlm.memory.cache import postnorm_hiddens
from rlm.memory.real_corpus import RealCorpusGenerator, load_squad_records
from rlm.memory.session import MemorySession
from rlm.memory.skill import MemorySkill


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def fact_bank(gen: RealCorpusGenerator, n: int):
    """n facts with DISTINCT entities (so 'ingest all, query each' has no duplicates)."""
    seen, facts = set(), []
    guard = 0
    while len(facts) < n and guard < 100 * n:
        guard += 1
        f = gen.make_fact()
        if f.entity not in seen:
            seen.add(f.entity)
            facts.append(f)
    return facts


def query_of(f, paraphrase):
    return f.paraphrase_prompts[0] if (paraphrase and f.paraphrase_prompts) else f.verbatim_prompt


def mean_pool_embed(model, tok, text, device):
    ids = torch.tensor([tok(text, add_special_tokens=True)["input_ids"]], device=device)
    h = postnorm_hiddens(model, ids).to(torch.float32)
    return F.normalize(h[0].mean(0), dim=-1)


# --------------------------------------------------------------------------
def run_baselines(model, tok, skill, facts, device, paraphrase):
    floor, mem, ctx = [], [], []
    stmt_emb = torch.stack([mean_pool_embed(model, tok, f.statement, device) for f in facts])
    retr_hit, retr_recall = [], []
    for i, f in enumerate(facts):
        q = query_of(f, paraphrase)
        sess = MemorySession(model, tok, skill, device=device)
        sess.ingest(f.statement)
        floor.append(sess.score_answer(q, f.answer, use_store=False)["first_token_lift_nats"])
        mem.append(sess.score_answer(q, f.answer, use_store=True)["first_token_lift_nats"])
        # in-context ceiling: the base sees the statement, no store
        ic = sess.score_answer(f.statement + " " + q, f.answer, use_store=False)
        ctx.append(ic["first_token_lift_nats"])
        # retrieval: top-1 statement by cosine to the query, then in-context with it
        qe = mean_pool_embed(model, tok, q, device)
        hit = int((stmt_emb @ qe).argmax())
        retr_hit.append(float(hit == i))
        rr = sess.score_answer(facts[hit].statement + " " + q, f.answer, use_store=False)
        retr_recall.append(float(rr["first_token_top5"]))
    return {
        "floor_lift_nats": mean(floor),
        "memory_lift_nats": mean(mem),
        "in_context_lift_nats": mean(ctx),
        "retrieval_at1": mean(retr_hit),
        "retrieval_recall_top5": mean(retr_recall),
        "n": len(facts),
    }


def run_generative(model, tok, skill, facts, device, paraphrase):
    em = []
    for f in facts:
        sess = MemorySession(model, tok, skill, device=device)
        sess.ingest(f.statement)
        out = sess.generate_greedy(query_of(f, paraphrase), max_new_tokens=8)
        em.append(float(f.answer.strip().lower() in out.lower()))
    return {"generative_exact_match": mean(em), "n": len(facts)}


def run_scale(model, tok, skill, facts, checkpoints, device, paraphrase, neutral):
    """Ingest the bank one fact at a time, save→reload between each (deployment
    regime), and at each checkpoint query ALL facts so far."""
    curve = []
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "store.pt")
        sess = MemorySession(model, tok, skill, device=device)
        for k, f in enumerate(facts, 1):
            sess.ingest(f.statement)
            sess.save(path)
            sess.load(path)  # round-trip the persisted store every fact
            if k in checkpoints:
                lifts, top5 = [], []
                for g in facts[:k]:
                    r = sess.score_answer(query_of(g, paraphrase), g.answer)
                    lifts.append(r["first_token_lift_nats"])
                    top5.append(float(r["first_token_top5"]))
                kl = sess.neutral_kl(neutral)
                pt = {
                    "n_facts": k,
                    "mean_lift_nats": mean(lifts),
                    "frac_top5": mean(top5),  # worst-case-leaning: every fact must still be retrievable
                    "min_lift_nats": min(lifts),
                    "neutral_kl": kl,
                }
                curve.append(pt)
                print(f"scale n={k:>4}: frac_top5 {pt['frac_top5']:.2f} mean_lift {pt['mean_lift_nats']:.1f} "
                      f"min_lift {pt['min_lift_nats']:.1f} neutral_kl {kl:.3f}")
    return curve


NEUTRAL = (
    "The library reading room stayed open until nine on weekdays. Visitors left their coats"
    " on the rack beside the entrance, and the catalog terminals hummed near the reference desk."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--squad", action="store_true")
    ap.add_argument("--baseline-facts", type=int, default=64)
    ap.add_argument("--scale", default="8,16,32,64,128")
    ap.add_argument("--paraphrase", action="store_true", help="query with questions (content-addressable)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="runs/real_eval.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    skill = MemorySkill.load(args.skill, device=args.device)

    records = load_squad_records() if args.squad else None
    gen = RealCorpusGenerator(records=records, split="eval", seed=args.seed)
    checkpoints = [int(x) for x in args.scale.split(",")]
    bank = fact_bank(gen, max(args.baseline_facts, max(checkpoints)))
    print(f"held-out bank: {len(bank)} distinct-entity real facts, paraphrase={args.paraphrase}")

    report = {
        "n_held_out": len(bank),
        "paraphrase": args.paraphrase,
        "baselines": run_baselines(model, tok, skill, bank[: args.baseline_facts], args.device, args.paraphrase),
        "generative": run_generative(model, tok, skill, bank[: args.baseline_facts], args.device, args.paraphrase),
        "scale": run_scale(model, tok, skill, bank, checkpoints, args.device, args.paraphrase, NEUTRAL),
    }
    b = report["baselines"]
    print(f"\nbaselines (n={b['n']}): floor {b['floor_lift_nats']:.2f} | memory {b['memory_lift_nats']:.2f} | "
          f"in-context {b['in_context_lift_nats']:.2f} | retrieval@1 {b['retrieval_at1']:.2f}")
    print(f"generative EM: {report['generative']['generative_exact_match']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
