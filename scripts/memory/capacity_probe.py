"""Capacity probes for the memory store (pure store math, CPU, no base model).

Backs the claims in docs/memory_capacity.md.  Four probes, each isolating one
factor in the store's fact capacity:

  A. value-dimension scaling  — does capacity grow with d_v? (curse-of-dimensionality)
  B. update rule              — Hebbian (no erosion) vs the delta rule (our store)
  C. key rank                 — capacity vs the effective rank of the key population
  D. readout nonlinearity     — linear (M·q) vs softmax-over-kept-kv (Hopfield/attention)

Run:  uv run python scripts/memory/capacity_probe.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlm.memory.linear_store import DeltaRuleStore, LinearStoreConfig, rms_norm

G = torch.Generator().manual_seed(0)


def _kv(n: int, d_k: int, d_v: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    basis = F.normalize(torch.randn(rank, d_k, generator=G), dim=-1)
    k = F.normalize(torch.randn(n, rank, generator=G) @ basis, dim=-1)
    v = F.normalize(torch.randn(n, d_v, generator=G), dim=-1)
    return k, v


def _recall1(read: torch.Tensor, v: torch.Tensor) -> float:
    return float((read @ v.T).argmax(-1).eq(torch.arange(len(v))).float().mean())


def hebbian(n: int, d_k: int, d_v: int, rank: int | None = None) -> float:
    """M = Σ v kᵀ (no error correction); read = M q."""
    k, v = _kv(n, d_k, d_v, rank or d_k)
    return _recall1((k @ k.T / d_k) @ v, v)


def delta(n: int, d_k: int, d_v: int, rank: int | None = None, theta: float = 1.0) -> float:
    """The actual error-correcting delta-rule store (sequential writes) at init lr
    ``theta`` (set via max_lr=2·theta, since the lr gate inits at sigmoid(0)·max_lr)."""
    k, v = _kv(n, d_k, d_v, rank or d_k)
    store = DeltaRuleStore(LinearStoreConfig(d_k=d_k, d_v=d_v, chunk_size=1, max_lr=2 * theta)).eval()
    st = store.init_state(1)
    with torch.no_grad():
        st = store.write(k.unsqueeze(0), v.unsqueeze(0), st)
        read = store.read(k.unsqueeze(0), st)[0]
    return _recall1(read, rms_norm(v))


def softmax_readout(n: int, d_k: int, d_v: int, beta: float = 16.0, rank: int | None = None) -> float:
    """Keep the kv pairs; read = softmax(β q·K) V (modern Hopfield / attention / kNN)."""
    k, v = _kv(n, d_k, d_v, rank or d_k)
    return _recall1(F.softmax(beta * (k @ k.T), -1) @ v, v)


def fullvocab_arm(head_path: str) -> None:
    """D0 (design review of memory_capacity.md): re-score probes A–C against the
    REAL tied head with full-vocabulary argmax — values are actual embedding
    rows, competitors are all |V| tokens, anisotropy and clustering included.
    The among-N random-unit protocol above is the geometric upper bound; this is
    the deployed readout."""
    head = torch.load(head_path, weights_only=False)
    W = head["W"].to(torch.float32)
    print(f"\n== FULL-VOCAB arm (real head: |V|={W.shape[0]}, d_v={W.shape[1]}) ==")

    def score(read: torch.Tensor, ids: torch.Tensor) -> tuple[float, float]:
        logits = read @ W.T
        rank = (logits > logits.gather(1, ids.unsqueeze(1))).sum(1)
        return float((rank == 0).float().mean()), float((rank < 5).float().mean())

    def kv_real(n: int, d_k: int, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = F.normalize(torch.randn(rank, d_k, generator=G), dim=-1)
        k = F.normalize(torch.randn(n, rank, generator=G) @ basis, dim=-1)
        ids = torch.randint(0, W.shape[0], (n,), generator=G)
        return k, F.normalize(W[ids], dim=-1), ids

    for name, d_k, rank, update in (
        ("hebbian d_k=256 full-rank", 256, 256, "heb"),
        ("delta   d_k=256 full-rank", 256, 256, "delta"),
        ("hebbian d_k=4096 rank-46 ", 4096, 46, "heb"),
        ("hebbian d_k=4096 rank-256", 4096, 256, "heb"),
    ):
        for n in (256, 1024, 2000):
            k, v, ids = kv_real(n, d_k, rank)
            if update == "heb":
                read = (k @ k.T / d_k) @ v
            else:
                store = DeltaRuleStore(LinearStoreConfig(d_k=d_k, d_v=W.shape[1], chunk_size=1)).eval()
                st = store.init_state(1)
                with torch.no_grad():
                    st = store.write(k.unsqueeze(0), v.unsqueeze(0), st)
                    read = store.read(k.unsqueeze(0), st)[0]
            t1, t5 = score(read, ids)
            print(f"  {name}  N={n:>5}: top1 {t1:.2f}  top5 {t5:.2f}")


def main() -> None:
    print("A. VALUE-DIM scaling — Hebbian recall@1, N=2000, d_k=256, full-rank keys")
    for d_v in (64, 256, 1152, 4096):
        print(f"     d_v={d_v:>5}: {hebbian(2000, 256, d_v):.2f}")

    print("\nB. UPDATE RULE — N=2000, d_k=256, d_v=1152, full-rank keys")
    print(f"     hebbian (no erosion) {hebbian(2000, 256, 1152):.2f}   delta (our store) {delta(2000, 256, 1152):.2f}")

    print("\nC. KEY RANK — Hebbian, huge store d_k=4096 d_v=1152, N=2000, keys confined to rank r")
    for r in (46, 256, 2000):
        print(f"     rank {r:>4}: {hebbian(2000, 4096, 1152, r):.2f}")

    print("\nD. READOUT — d_k=256, d_v=1152, full-rank keys, recall@1 vs N")
    print(f"     {'N':>6} {'×d_k':>5} | {'linear M·q':>10} | {'softmax kv':>10}")
    for n in (256, 1024, 4096, 8192):
        print(f"     {n:>6} {n // 256:>4}x | {hebbian(n, 256, 1152):>10.2f} | {softmax_readout(n, 256, 1152):>10.2f}")

    print("\nE. UPDATE RULE × KEY RANK — d_k=512, d_v=1152, recall@1 vs N")
    print("   (lower lr is a ~10x lever EVEN at rank 46; rank is a SOFT degrader, not a 46-cap)")
    NS = (64, 256, 1024, 2048)
    for rank in (46, 512):
        tag = "46 (≈ real frozen-base key rank)" if rank == 46 else "512 (= d_k, full rank)"
        print(f"   -- key rank {tag} --")
        print("     " + "rule".ljust(12) + " ".join(f"N={n:<5}" for n in NS))
        for theta in (None, 1.0, 0.1, 0.03):
            nm = "hebbian" if theta is None else f"delta θ={theta}"
            vals = [hebbian(n, 512, 1152, rank) if theta is None else delta(n, 512, 1152, rank, theta) for n in NS]
            print("     " + nm.ljust(12) + " ".join(f"{x:<7.2f}" for x in vals))


if __name__ == "__main__":
    import sys

    main()
    if len(sys.argv) > 1:
        fullvocab_arm(sys.argv[1])  # usage: capacity_probe.py cache/<dir>/head.pt
