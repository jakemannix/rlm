"""Step 0 — oracle injection (design doc §4).

Bypasses the memory entirely: adds λ·rms_norm(e[gold]) to the post-final-norm
hidden at the answer position and sweeps λ, with a matched random-direction
control.  Gate: some stable λ-range where gold enters top-5 with Δlogprob ≥ +3
nats while the random-direction control stays flat.  If no λ works, post-norm
residual injection is dead at this splice regardless of memory quality → switch
to the two-pass soft-token arm before building anything else.

Runtime: minutes on an L4 (forwards only).

    python scripts/memory/oracle_injection.py --model google/gemma-3-1b-it \
        --facts 24 --out runs/oracle_injection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rlm.memory.cache import get_head, postnorm_hiddens
from rlm.memory.episodes import EpisodeGenerator
from rlm.memory.linear_store import rms_norm

LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--facts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="runs/oracle_injection.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    W, softcap = get_head(model)
    W = W.detach().to(torch.float32)
    print(f"model={args.model} d={W.shape[1]} vocab={W.shape[0]} softcap={softcap}")

    def head(h: torch.Tensor) -> torch.Tensor:
        z = h.to(torch.float32) @ W.T.to(h.device)
        if softcap is not None:
            z = softcap * torch.tanh(z / softcap)
        return z

    gen = EpisodeGenerator(seed=args.seed)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []
    for _ in range(args.facts):
        f = gen.make_fact()
        ids = torch.tensor([[tok.bos_token_id] + tok(f.verbatim_prompt, add_special_tokens=False)["input_ids"]], device=args.device)
        gold = tok(f.answer, add_special_tokens=False)["input_ids"][0]
        hn = postnorm_hiddens(model, ids).to(torch.float32)[0, -1]
        e_gold = rms_norm(W[gold]).to(args.device)
        e_rand = rms_norm(torch.randn(W.shape[1], generator=rng)).to(args.device)
        base_logp = F.log_softmax(head(hn), dim=-1)
        row = {"prompt": f.verbatim_prompt, "answer": f.answer,
               "base_gold_logprob": float(base_logp[gold]),
               "base_gold_rank": int((base_logp > base_logp[gold]).sum()), "sweep": {}}
        for lam in LAMBDAS:
            lp_g = F.log_softmax(head(hn + lam * e_gold), dim=-1)
            lp_r = F.log_softmax(head(hn + lam * e_rand), dim=-1)
            row["sweep"][lam] = {
                "gold_lift_nats": float(lp_g[gold] - base_logp[gold]),
                "gold_rank": int((lp_g > lp_g[gold]).sum()),
                "rand_gold_lift_nats": float(lp_r[gold] - base_logp[gold]),
                "rand_kl_nats": float(F.kl_div(base_logp, lp_r, reduction="sum", log_target=True)),
            }
        rows.append(row)

    summary = {}
    print(f"\n{'λ':>6} {'mean Δgold (nats)':>18} {'top5 frac':>10} {'rand Δgold':>11} {'rand KL':>9}")
    for lam in LAMBDAS:
        lifts = [r["sweep"][lam]["gold_lift_nats"] for r in rows]
        top5 = [r["sweep"][lam]["gold_rank"] < 5 for r in rows]
        rnd = [r["sweep"][lam]["rand_gold_lift_nats"] for r in rows]
        rkl = [r["sweep"][lam]["rand_kl_nats"] for r in rows]
        summary[lam] = {
            "mean_gold_lift_nats": sum(lifts) / len(lifts),
            "top5_frac": sum(top5) / len(top5),
            "mean_rand_gold_lift_nats": sum(rnd) / len(rnd),
            "mean_rand_kl_nats": sum(rkl) / len(rkl),
        }
        s = summary[lam]
        print(f"{lam:>6} {s['mean_gold_lift_nats']:>18.2f} {s['top5_frac']:>10.2f} "
              f"{s['mean_rand_gold_lift_nats']:>11.2f} {s['mean_rand_kl_nats']:>9.3f}")

    passing = [lam for lam, s in summary.items()
               if s["mean_gold_lift_nats"] >= 3.0 and s["top5_frac"] >= 0.9
               and abs(s["mean_rand_gold_lift_nats"]) <= 0.5]
    verdict = {"gate": "step0_oracle_injection", "pass": bool(passing), "passing_lambdas": passing}
    print(f"\nVERDICT: {'PASS' if passing else 'FAIL'}  (λ ∈ {passing})")
    if not passing:
        print("→ post-norm injection cannot steer this base; branch to the two-pass soft-token arm.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"verdict": verdict, "summary": summary, "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
