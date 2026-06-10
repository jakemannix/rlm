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


def agg(dicts, keys):
    """mean ± sd of each key across per-seed result dicts (stats protocol)."""
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        m = mean(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        out[k] = {"mean": m, "sd": sd}
    return out


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
    """All lifts are in answer-logprob NATS over the no-context/no-store FLOOR
    (base scoring the answer from the query alone), so the conditions are
    comparable: memory = store's contribution at the query; in_context = the base
    with the statement IN the prompt (the ceiling); retrieval = in-context with the
    top-1 retrieved statement.  (Earlier bug: used store-vs-no-store lift, which is
    0 by construction when use_store=False — it never measured the ceiling.)"""
    mem, ctx, retr_hit, retr_ctx = [], [], [], []
    stmt_emb = torch.stack([mean_pool_embed(model, tok, f.statement, device) for f in facts])
    for i, f in enumerate(facts):
        q = query_of(f, paraphrase)
        sess = MemorySession(model, tok, skill, device=device)
        sess.ingest(f.statement)
        r_q = sess.score_answer(q, f.answer, use_store=True)
        floor_abs = r_q["gold_logprob_base"]  # base, query alone, no store
        mem.append(r_q["gold_logprob"] - floor_abs)  # memory's contribution
        ic = sess.score_answer(f.statement + " " + q, f.answer, use_store=False)
        ctx.append(ic["gold_logprob_base"] - floor_abs)  # in-context ceiling over floor
        qe = mean_pool_embed(model, tok, q, device)
        hit = int((stmt_emb @ qe).argmax())
        retr_hit.append(float(hit == i))
        rr = sess.score_answer(facts[hit].statement + " " + q, f.answer, use_store=False)
        retr_ctx.append(rr["gold_logprob_base"] - floor_abs)  # in-context with the retrieved fact
    return {
        "floor_lift_nats": 0.0,  # reference
        "memory_lift_nats": mean(mem),
        "in_context_lift_nats": mean(ctx),
        "retrieval_at1": mean(retr_hit),
        "retrieval_in_context_lift_nats": mean(retr_ctx),
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
    ap.add_argument("--n-seeds", type=int, default=3, help="repeat baselines/generative over N held-out draws -> mean±sd")
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
    checkpoints = [int(x) for x in args.scale.split(",")]

    # baselines + generative over N seeds (different held-out draws) -> mean ± sd
    base_runs, gen_runs = [], []
    for s in range(args.n_seeds):
        gs = RealCorpusGenerator(records=records, split="eval", seed=args.seed + s)
        bank_s = fact_bank(gs, args.baseline_facts)
        base_runs.append(run_baselines(model, tok, skill, bank_s, args.device, args.paraphrase))
        gen_runs.append(run_generative(model, tok, skill, bank_s, args.device, args.paraphrase))
    bkeys = ["floor_lift_nats", "memory_lift_nats", "in_context_lift_nats", "retrieval_at1", "retrieval_in_context_lift_nats"]

    # scale + persistence once on the seed-0 bank (already worst-case over many facts)
    scale_bank = fact_bank(RealCorpusGenerator(records=records, split="eval", seed=args.seed), max(checkpoints))
    report = {
        "n_held_out_entities": len({f.entity for f in scale_bank}),
        "paraphrase": args.paraphrase,
        "n_seeds": args.n_seeds,
        "baselines": agg(base_runs, bkeys),
        "generative": agg(gen_runs, ["generative_exact_match"]),
        "scale": run_scale(model, tok, skill, scale_bank, checkpoints, args.device, args.paraphrase, NEUTRAL),
    }
    b = report["baselines"]
    print(f"\nbaselines (n_seeds={args.n_seeds}, mean±sd): "
          f"floor {b['floor_lift_nats']['mean']:.2f}±{b['floor_lift_nats']['sd']:.2f} | "
          f"memory {b['memory_lift_nats']['mean']:.2f}±{b['memory_lift_nats']['sd']:.2f} | "
          f"in-context {b['in_context_lift_nats']['mean']:.2f} | retrieval@1 {b['retrieval_at1']['mean']:.2f}")
    print(f"generative EM: {report['generative']['generative_exact_match']['mean']:.2f}"
          f"±{report['generative']['generative_exact_match']['sd']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
