"""Key-geometry diagnostics + capacity prediction (design review §1–§2).

Three parts, all cheap (no base model; runs from a cache directory + optionally
a trained skill):

1. **Measure** the addressing population's geometry from the cache: eigen-spectrum,
   participation ratio, r50/r90 (effective rank), and pairwise-cosine structure —
   within-relation vs across-relation when the cache has relation labels (caches
   built after the labels patch).  This is the proof/refutation of the
   effective-dim hypothesis.
2. **Predict** the A3 curve from the measured covariance: synthesise keys with the
   measured spectrum, run the pure delta-rule store against the *real* head
   (``head.pt``), report top-5 vs N — raw vs ZCA-whitened.  If the raw predicted
   curve matches the observed A3 knee and the whitened curve doesn't, whitening is
   the fix and the expected post-fix curve is printed before anyone trains anything.
3. **Oracle-key ceiling**: orthonormal keys + real-embedding values → the store's
   capacity with perfect addressing (separates key collision from delta-rule
   erosion; design-review answer to handoff Q4).

    python scripts/memory/diagnose_keys.py --cache cache/gemma1b_v1 \
        --skill runs/skill_v1.pt --out runs/key_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rlm.memory.skill import MemorySkill, fit_zca


# ---------------------------------------------------------------------------
# Part 1 — measurement
# ---------------------------------------------------------------------------
def collect_addressing_vectors(cache_dir: Path, max_vectors: int) -> tuple[torch.Tensor, list[str | None]]:
    """Query-position post-norm hiddens (one per recall episode) with relation
    labels where the cache provides them.  These are the *query keys'* inputs —
    the population whose geometry bounds capacity."""
    xs: list[torch.Tensor] = []
    labels: list[str | None] = []
    for shard_path in sorted(cache_dir.glob("shard_*.pt")):
        for ep in torch.load(shard_path, weights_only=False):
            q = ep["query"]
            if not bool(q["ce_in_loss"]) or len(q["ce_pos"]) == 0:
                continue
            xs.append(q["hn"][q["ce_pos"][0]].to(torch.float32).unsqueeze(0))
            labels.append(ep.get("relation"))
            if len(xs) >= max_vectors:
                return torch.cat(xs), labels
    if not xs:
        raise ValueError(f"no recall episodes found in {cache_dir}")
    return torch.cat(xs), labels


def spectrum_stats(x: torch.Tensor) -> dict:
    xc = x - x.mean(dim=0)
    cov = xc.T @ xc / max(len(x) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0.0).flip(0)
    total = ev.sum().clamp(min=1e-12)
    frac = ev / total
    cum = torch.cumsum(frac, 0)
    return {
        "n_vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "participation_ratio": float(total.pow(2) / ev.pow(2).sum().clamp(min=1e-12)),
        "r50": int((cum < 0.50).sum()) + 1,
        "r90": int((cum < 0.90).sum()) + 1,
        "top1_var_frac": float(frac[0]),
        "top8_var_frac": float(frac[:8].sum()),
    }


def cosine_stats(x: torch.Tensor, labels: list[str | None], n_pairs: int = 20_000) -> dict:
    xn = F.normalize(x, dim=-1)
    g = torch.Generator().manual_seed(0)
    i = torch.randint(0, len(x), (n_pairs,), generator=g)
    j = torch.randint(0, len(x), (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    cos = (xn[i] * xn[j]).sum(-1)
    out = {"mean_abs_cos": float(cos.abs().mean()), "p95_abs_cos": float(cos.abs().quantile(0.95))}
    if any(lbl is not None for lbl in labels):
        same = torch.tensor(
            [labels[a] == labels[b] and labels[a] is not None
             for a, b in zip(i.tolist(), j.tolist(), strict=True)]
        )
        if same.any() and (~same).any():
            out["mean_cos_same_relation"] = float(cos[same].mean())
            out["mean_cos_cross_relation"] = float(cos[~same].mean())
    else:
        out["note"] = "no relation labels in this cache — rebuild with the labels patch for within/across-template cosines"
    return out


def paired_write_query_cosines(cache_dir: Path, skill, max_pairs: int = 2000) -> dict | None:
    """D1a (design review): how well does the key written DURING the streamed
    session (at the answer-binding position, in session context) match the key
    the standalone QUERY produces for the same fact?  Same-fact vs cross-fact
    cosine; the gap is the addressing margin that survives streaming.  Needs a
    cache built with answer_mask (post answer-segment patch); returns None on
    older caches."""
    wk, qk = [], []
    for shard_path in sorted(cache_dir.glob("shard_*.pt")):
        for ep in torch.load(shard_path, weights_only=False):
            q = ep["query"]
            if ep.get("type") != "recall" or not bool(q["ce_in_loss"]) or len(q["ce_pos"]) == 0:
                continue
            sess = ep["sessions"][0]
            am = sess.get("answer_mask")
            if am is None:
                return None
            nz = am.nonzero()
            if len(nz) == 0:
                continue
            wk.append(sess["hn"][int(nz[0])].to(torch.float32).unsqueeze(0))
            qk.append(q["hn"][int(q["ce_pos"][0])].to(torch.float32).unsqueeze(0))
            if len(wk) >= max_pairs:
                break
        if len(wk) >= max_pairs:
            break
    if len(wk) < 8:
        return None
    w, q_ = torch.cat(wk), torch.cat(qk)
    if skill is not None:
        with torch.no_grad():
            w = skill.key_enc(skill.whiten(w))
            q_ = skill.query_enc(skill.whiten(q_))
    w, q_ = F.normalize(w, dim=-1), F.normalize(q_, dim=-1)
    same = (w * q_).sum(-1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(w), generator=g)
    cross = (w * q_[perm]).sum(-1)
    return {
        "n_pairs": int(len(w)),
        "same_fact_cos_mean": float(same.mean()),
        "same_fact_cos_p10": float(same.quantile(0.10)),
        "cross_fact_cos_mean": float(cross.mean()),
        "margin": float(same.mean() - cross.mean()),
    }


# ---------------------------------------------------------------------------
# Parts 2 & 3 — prediction
# ---------------------------------------------------------------------------
def delta_write(K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Pure delta rule, θ=1, sequential single pass.  K [m, d_k], V [m, d_v]."""
    M = torch.zeros(V.shape[1], K.shape[1])
    for k, v in zip(K, V, strict=True):
        M = M + torch.outer(v - M @ k, k) / K.shape[1]
    return M


def store_top5(K: torch.Tensor, Q: torch.Tensor, gold: torch.Tensor, W: torch.Tensor) -> float:
    V = F.normalize(W[gold], dim=-1) * (W.shape[1] ** 0.5)
    M = delta_write(K, V)
    logits = (M @ Q.T).T @ W.T
    rank = (logits > logits.gather(1, gold.unsqueeze(1))).sum(1)
    return float((rank < 5).float().mean())


def predicted_capacity_curve(
    x: torch.Tensor, W: torch.Tensor, d_k: int, ns: list[int],
    whiten_eps: float, q_noise: float, trials: int, seed: int = 0,
) -> dict:
    """Keys drawn with the *measured* covariance (x's), mapped by a random
    orthogonal encoder to d_k — raw vs whitened — against the real head W."""
    mean_w, T_w = fit_zca(x, eps_frac=whiten_eps)
    xc = x - x.mean(dim=0)
    cov = xc.T @ xc / max(len(x) - 1, 1)
    S, U = torch.linalg.eigh(cov)
    half = U @ torch.diag(S.clamp(min=0).sqrt()) @ U.T  # Σ^{1/2}
    enc = torch.linalg.qr(torch.randn(x.shape[1], d_k, generator=torch.Generator().manual_seed(seed)))[0]

    out: dict = {"raw": {}, "whitened": {}}
    for n in ns:
        for arm in ("raw", "whitened"):
            accs = []
            for t in range(trials):
                g = torch.Generator().manual_seed(1000 * t + n)
                h = torch.randn(n, x.shape[1], generator=g) @ half + x.mean(dim=0)
                hq = h + q_noise * h.std() * torch.randn(n, x.shape[1], generator=g)
                if arm == "whitened":
                    h, hq = (h - mean_w) @ T_w.T, (hq - mean_w) @ T_w.T
                K = F.normalize(h @ enc, dim=-1) * (d_k ** 0.5)
                Q = F.normalize(hq @ enc, dim=-1) * (d_k ** 0.5)
                gold = torch.randint(0, W.shape[0], (n,), generator=g)
                accs.append(store_top5(K, Q, gold, W))
            out[arm][n] = sum(accs) / len(accs)
    return out


def oracle_key_ceiling(W: torch.Tensor, d_k: int, ns: list[int], trials: int) -> dict:
    out = {}
    for n in ns:
        accs = []
        for t in range(trials):
            g = torch.Generator().manual_seed(t)
            K = torch.randn(max(n, d_k), d_k, generator=g)
            K = torch.linalg.qr(K.T)[0].T[:n] * (d_k ** 0.5) if n <= d_k else F.normalize(torch.randn(n, d_k, generator=g), dim=-1) * (d_k ** 0.5)
            gold = torch.randint(0, W.shape[0], (n,), generator=g)
            accs.append(store_top5(K, K, gold, W))
        out[n] = sum(accs) / len(accs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--skill", default=None, help="optional trained skill — also reports trained-key geometry")
    ap.add_argument("--d-k", type=int, default=512)
    ap.add_argument("--max-vectors", type=int, default=8_000)
    ap.add_argument("--ns", default="4,8,16,32,64,128")
    ap.add_argument("--whiten-eps", type=float, default=1e-3)
    ap.add_argument("--q-noise", type=float, default=0.1)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="runs/key_diagnostics.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    head = torch.load(cache / "head.pt", weights_only=False)
    W = head["W"].to(torch.float32)
    ns = [int(v) for v in args.ns.split(",")]

    x, labels = collect_addressing_vectors(cache, args.max_vectors)
    report: dict = {"raw_hidden": {**spectrum_stats(x), **cosine_stats(x, labels)}}
    print(f"raw addressing hiddens: {json.dumps(report['raw_hidden'], indent=2)}")

    if args.skill:
        skill = MemorySkill.load(args.skill)
        with torch.no_grad():
            k = skill.key_enc(skill.whiten(x))
        report["trained_keys"] = {**spectrum_stats(k), **cosine_stats(k, labels)}
        print(f"trained keys: {json.dumps(report['trained_keys'], indent=2)}")

    paired = paired_write_query_cosines(cache, MemorySkill.load(args.skill) if args.skill else None)
    if paired is not None:
        report["write_query_pairing"] = paired
        print(f"\nD1a write-vs-query key pairing: {json.dumps(paired, indent=2)}")
        print("→ same-fact cos ≈ cross-fact cos means streaming destroys addressing"
              " (context-mismatch); a healthy gap means the bottleneck is erosion/selectivity.")
    else:
        print("\nD1a pairing: cache lacks answer_mask — rebuild cache for the write/query diagnostic")

    print("\npredicted A3 ceiling from measured geometry (top5 vs N):")
    pred = predicted_capacity_curve(x, W, args.d_k, ns, args.whiten_eps, args.q_noise, args.trials)
    report["predicted_capacity"] = pred
    print(f"{'N':>6} | {'raw':>6} {'whitened':>9}")
    for n in ns:
        print(f"{n:>6} | {pred['raw'][n]:>6.2f} {pred['whitened'][n]:>9.2f}")

    report["oracle_key_ceiling"] = oracle_key_ceiling(W, args.d_k, ns, args.trials)
    print(f"\noracle orthonormal-key ceiling: {report['oracle_key_ceiling']}")
    print("→ if 'raw' matches the observed A3 knee while 'oracle' is clean, the collapse is key")
    print("  geometry (not erosion); the 'whitened' column is the expected post-fix curve.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
