"""How does the DENSE store M ∈ R^{d_v × d_k} scale with module size d_k and key
quality?  Pure store math (no base model), so it isolates the MODULE's information
capacity from the encoder/streaming bottlenecks.

For each (d_k, key-cosine, N): write N (value, key) pairs into a fresh DeltaRuleStore
at lr=1 and measure recall@1 (is read_i nearest its own stored value among the N?).
Keys are drawn with a controlled mean pairwise cosine ρ (ρ=0 ⇒ orthonormal = the best
case the encoder could ever hope to produce).

    uv run python scripts/memory/capacity_scaling.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlm.memory.linear_store import DeltaRuleStore, LinearStoreConfig, rms_norm


def capacity(d_k: int, n: int, cosine: float, d_v: int = 1152, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    store = DeltaRuleStore(LinearStoreConfig(d_k=d_k, d_v=d_v, chunk_size=1)).eval()
    keys = F.normalize(torch.randn(n, d_k, generator=g), dim=-1)
    if cosine > 0:  # inject a shared component to raise mean pairwise cosine ~ρ
        shared = F.normalize(torch.randn(d_k, generator=g), dim=-1)
        a = cosine ** 0.5
        keys = F.normalize(a * shared + (1 - a) ** 0.5 * keys, dim=-1)
    vals = F.normalize(torch.randn(1, n, d_v, generator=g), dim=-1)
    state = store.init_state(1)
    with torch.no_grad():
        state = store.write(keys.unsqueeze(0), vals, state)
        read = store.read(keys.unsqueeze(0), state)[0]
    scores = read @ rms_norm(vals[0]).T
    return float((scores.argmax(-1) == torch.arange(n)).float().mean())


def main() -> None:
    NS = [16, 64, 256, 1024, 4096]
    print("Dense store capacity — recall@1 vs N facts (rows: d_k × key-cosine ρ)\n")
    print(f"{'d_k':>6} {'ρ':>5} | " + " ".join(f"N={n:<5}" for n in NS))
    for d_k in (512, 2048, 8192):
        for cos in (0.0, 0.1, 0.3):
            row = []
            for n in NS:
                row.append(f"{capacity(d_k, n, cos):.2f}" if n <= 2 * d_k else "  -- ")
            print(f"{d_k:>6} {cos:>5.1f} | " + " ".join(f"{c:<7}" for c in row))
        print()


if __name__ == "__main__":
    main()
