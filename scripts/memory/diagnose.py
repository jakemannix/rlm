"""Capacity diagnostics for the decoupled delta-rule store (runs on CPU, no Colab).

Answers the two questions the design review flagged as gating the whole capacity
direction (docs/handoff_capacity_and_multihead.md §4):

A. ORACLE-KEY CEILING: with *perfect* (orthonormal) keys, does the delta-rule store
   recall N facts, or does it collapse anyway? Collapse here => delta-rule EROSION is
   the ceiling (keys can't fix it). Perfect recall to ~d_k => the store is fine and the
   bottleneck is the KEYS (encoder geometry).
B. REAL-KEY GEOMETRY: effective rank (participation ratio) + pairwise-cosine of Gemma's
   post-final-norm hidden at the answer-onset position across nonce facts, and of a
   random-linear-projection of it (mimicking the untrained key_enc). Low effective rank /
   high cosine => keys collide => the diagnosed effective-dim bottleneck.

    uv run python scripts/memory/diagnose.py --model google/gemma-3-1b-it --facts 128
"""

from __future__ import annotations

import argparse
import os

import torch

from rlm.memory.linear_store import DeltaRuleStore, LinearStoreConfig, rms_norm


def participation_ratio(X: torch.Tensor) -> float:
    """Effective rank: (sum sigma^2)^2 / sum sigma^4 of the centered rows of X."""
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.float())
    e = s.pow(2)
    return float((e.sum() ** 2) / (e.pow(2).sum() + 1e-12))


def mean_pairwise_cosine(X: torch.Tensor) -> tuple[float, float]:
    U = torch.nn.functional.normalize(X.float(), dim=-1)
    C = U @ U.T
    n = C.shape[0]
    off = C[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean()), float(off.abs().mean())


# --------------------------------------------------------------------------
# A. Oracle-key capacity ceiling (no model — pure store math)
# --------------------------------------------------------------------------
def oracle_capacity(d_k: int, d_v: int, ns: list[int], cosine: float, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    store = DeltaRuleStore(LinearStoreConfig(d_k=d_k, d_v=d_v, chunk_size=1))
    store.eval()
    curve = []
    for n in ns:
        # n keys with a controlled mean pairwise cosine ~= `cosine` (cosine=0 -> orthogonal).
        base = torch.randn(n, d_k, generator=g)
        base = torch.nn.functional.normalize(base, dim=-1)
        if cosine > 0:
            shared = torch.nn.functional.normalize(torch.randn(d_k, generator=g), dim=-1)
            a = cosine ** 0.5
            base = torch.nn.functional.normalize(a * shared + (1 - a) ** 0.5 * base, dim=-1)
        keys = base.unsqueeze(0)  # [1, n, d_k]
        vals = torch.nn.functional.normalize(torch.randn(1, n, d_v, generator=g), dim=-1)
        state = store.init_state(1)
        with torch.no_grad():
            state = store.write(keys, vals, state)
            read = store.read(keys, state)  # [1, n, d_v]
        # recall: is read_i closest (max dot) to val_i among all n stored values?
        scores = torch.einsum("nv,mv->nm", read[0], rms_norm(vals[0]))
        rank0 = (scores.argmax(dim=-1) == torch.arange(n)).float().mean().item()
        relmse = ((read[0] - rms_norm(vals[0])).pow(2).mean() / rms_norm(vals[0]).pow(2).mean()).item()
        curve.append({"n": n, "recall@1": round(rank0, 3), "rel_mse": round(relmse, 3)})
    return curve


# --------------------------------------------------------------------------
# B. Real Gemma key geometry
# --------------------------------------------------------------------------
def real_key_geometry(model_id: str, n_facts: int, d_k: int, seed: int) -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    tok_env = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if tok_env:
        os.environ["HF_TOKEN"] = tok_env

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rlm.memory.cache import postnorm_hiddens
    from rlm.memory.episodes import EpisodeGenerator

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    gen = EpisodeGenerator(seed=seed)
    hs = []
    with torch.no_grad():
        for _ in range(n_facts):
            f = gen.make_fact()
            ids = tok(f.verbatim_prompt, return_tensors="pt").input_ids
            ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1)
            hn = postnorm_hiddens(model, ids)[0, -1]  # answer-onset (last prompt) position
            hs.append(hn.float())
    H = torch.stack(hs)  # [N, d_model]
    d_model = H.shape[-1]

    torch.manual_seed(seed)
    Wk = torch.nn.functional.normalize(torch.randn(d_model, d_k), dim=0)  # random linear "key_enc"
    K = rms_norm(H @ Wk)  # mimics untrained key_enc(hn)

    mc_h, mac_h = mean_pairwise_cosine(H)
    mc_k, mac_k = mean_pairwise_cosine(K)
    return {
        "model": model_id, "n_facts": n_facts, "d_model": d_model, "d_k": d_k,
        "hidden_participation_ratio": round(participation_ratio(H), 1),
        "hidden_mean_cos": round(mc_h, 3), "hidden_mean_abscos": round(mac_h, 3),
        "key_participation_ratio": round(participation_ratio(K), 1),
        "key_mean_cos": round(mc_k, 3), "key_mean_abscos": round(mac_k, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--facts", type=int, default=128)
    ap.add_argument("--d-k", type=int, default=512)
    ap.add_argument("--d-v", type=int, default=1152)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-model", action="store_true", help="run only the no-model oracle ceiling")
    args = ap.parse_args()

    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print("=== A. ORACLE-KEY capacity (orthogonal keys; collapse here => delta-rule erosion is the ceiling) ===")
    for cos in (0.0, 0.3, 0.6):
        curve = oracle_capacity(args.d_k, args.d_v, ns, cos, args.seed)
        tag = "orthogonal" if cos == 0 else f"mean-cos~{cos}"
        print(f"  [{tag:14}] " + "  ".join(f"n{c['n']}:{c['recall@1']:.2f}" for c in curve))

    if not args.skip_model:
        print("\n=== B. REAL Gemma key geometry (low rank / high cosine => keys collide) ===")
        geo = real_key_geometry(args.model, args.facts, args.d_k, args.seed)
        for k, v in geo.items():
            print(f"  {k:28}: {v}")


if __name__ == "__main__":
    main()
