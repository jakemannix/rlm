# Memory fact-capacity: theory, measurements, and open R&D

> Design notes for review (Fable). Captures what we've learned about how many facts the
> parametric store can hold, where the real ceilings are, and the open questions worth
> R&D. All numbers are from `scripts/memory/{capacity_probe,capacity_scaling,diagnose_keys}.py`
> (pure store math, CPU, no base model) plus the real-text runs in `runs/REPORT.md`.
>
> **Two corrections to earlier claims are baked in here, flagged ⚠️ — the "capacity ≈ d_k,
> 10⁴ off the table" framing was wrong.**

## 0. The object
The store is a single dense matrix `M ∈ R^{d_v × d_k}` (d_v = 1152 = Gemma embedding dim, fixed
by the tied-embedding readout; d_k = key dim, default 512). Updated by the analytic **delta rule**
(error-correcting: `M ← (1-α)M + ηS - G`, `G = Σ_t θ_t (M k_t - v_t) k_tᵀ / d_k`), RMS-normed keys,
value `v` = the answer-shifted tied token embedding. A meta-trained-then-frozen "skill" (key/query
encoders on post-final-norm hiddens, gates, ZCA whitening, out_proj) produces the keys; `M` is the
mutable, persisted, cross-session content. Read = `M q`, injected post-final-norm. Code:
`rlm/memory/{linear_store,skill,session}.py`.

## 1. TL;DR for the reviewer
- ⚠️ **The store's geometric capacity is NOT `d_k`.** A linear associative memory holds **far more
  than `d_k` facts** because near-orthogonal value vectors pack densely in the high-d value space
  (Johnson–Lindenstrauss / "curse of dimensionality") and their cross-talk cancels at readout.
  Probe: a Hebbian store at `d_k=256, d_v=1152` recalls **2000 facts (≈8× d_k) at 1.00**, and the
  capacity **grows with d_v** (d_v=64→0.07, 256→0.64, 1152→1.00 at N=2000). So "capacity ≈ d_k" was
  the *eroded* number, not the geometric one.
- ⚠️ **Our delta rule (at lr θ=1) is the DOMINANT capacity killer — and lowering the lr recovers most
  of it, even at low key rank.** Same keys/values, N=2000, full rank: Hebbian **1.00** vs delta θ=1
  **0.13**; and at the *real* key rank 46, delta θ=1 caps ~64 facts while **delta θ=0.03 holds 0.60@1024
  and Hebbian holds 0.98@1024**. The error-correcting projector erodes prior content whenever keys
  overlap — catastrophic at θ=1, ~gone as θ→0. **A `max_lr` drop (2.0→~0.06; design rule θ* ≈ 0.3·d_eff/N_lifetime) is predicted
  to be worth ~10× — but only WITH retraining (or at minimum a re-fit read gain): the probe metric is
  scale-invariant in θ while deployment lift ∝ θ·gain, so an eval-only drop on the existing skill will
  collapse recall and must not be read as refuting the lever (review §8-A1).**
- ⚠️ **Key *rank* is a SOFT degrader, NOT a hard cap (an earlier "≈46-cap" claim here was wrong).**
  Isotropic rank-46 keys recall **~1000 facts** (the high-d value space disambiguates overlapping keys).
  Rank is PR ≈10–22 (answer-prefix hiddens) / ≈46 (whitened-trained keys). So the real ~8–16 ceiling is
  **delta-θ=1 erosion + real-key *anisotropy* (template-clustered, cos 0.45–0.77, worse than isotropic
  rank-46) + the streamed write↔query mismatch** — not a 46-fact wall.
- **Therefore the module size and the value dimension are *ample*, not the constraint.** Scaling to
  10³–10⁴ is gated by **(a) the update rule, (b) key rank/addressing, (c) the streamed-write dynamics**
  — all addressing/dynamics problems with concrete *untested* levers. None of them is "make M bigger."

## 2. Measurements (`scripts/memory/capacity_probe.py`)
```
A. VALUE-DIM scaling (Hebbian, N=2000, d_k=256, full-rank keys):
     d_v=64 0.07 | 256 0.64 | 1152 1.00 | 4096 1.00        ← capacity grows with d_v
B. UPDATE RULE (N=2000, d_k=256, d_v=1152, full-rank keys):
     hebbian 1.00   vs   delta-rule (our store) 0.13        ← the delta rule is the capacity killer
C. KEY RANK (huge store d_k=4096 d_v=1152, N=2000):
     rank 46 0.79 | 256 1.00 | 2000 1.00                    ← addressing rank is the binding cap
D. READOUT nonlinearity (d_k=256, d_v=1152, recall@1 vs N):
     N=  256(1x) 1024(4x) 4096(16x) 8192(32x)
     linear M·q   1.00    1.00     0.96      0.58
     softmax kv   1.00    1.00     1.00      1.00            ← softmax(=Hopfield/attention) > linear
E. UPDATE RULE × KEY RANK (d_k=512, d_v=1152, recall@1 vs N) — lower lr is ~10× EVEN at rank 46:
     key rank 46 (≈ real):     N=64   256    1024   2048
       hebbian                 1.00   1.00   0.98   0.73
       delta θ=1.0 (today)     0.84   0.23   0.05   0.03    ← the lr we run = the capacity killer
       delta θ=0.03            1.00   1.00   0.60   0.30    ← just lowering lr ≈ recovers it
     key rank 512 (full):  delta θ=1.0 → 0.51@1024 ; delta θ=0.03 → 1.00@1024, 0.92@2048
```
Corroborating (`capacity_scaling.py`, the *delta* store with orthonormal keys): recall ≈ 1.0 up to
N ≈ d_k then decays (d_k=512→1.0@256/0.78@1024; 8192→0.97@4096); realistic key-cosine ρ=0.3 collapses
even d_k=8192 to 0.54@1024. Real deployment (`eval_real.py` item 5, one fact/session + save/reload):
frac_top5 0.75@8 → 0.28@64 at d_k=512, min_lift goes negative — caps at ~8–16 on real text.

## 3. The three ceilings, and which binds
| Ceiling | Set by | Status | Lever |
|---|---|---|---|
| **Module / store** | ~ key-crowding bound, ≫ d_k; grows with d_v | **ample** (≥ tens × d_k; M is ~1–150 MB) | trivial: widen d_k / d_v is fixed |
| **Update rule (DOMINANT)** | delta-rule erosion at θ=1; ≈ vanishes as θ→0 | **the main killer; ~10× recoverable even at rank 46 by lowering lr** | drop `max_lr` 2.0→~0.06; capacity-pressure training; novelty-gated lr |
| **Key rank (addressing)** | rank of `key_enc(post-norm hidden)`, PR ≈10–46 | **soft degrader, NOT a 46-cap** (rank-46 → ~1000 with gentle write); real-key *anisotropy* is the bite | whitening (have it); earlier-layer keys; multi-token keys |
| **Streaming dynamics** | in-session-write key ≠ standalone-query key; filler erosion; whole-statement writes | caps real recall ~8–16 below static prediction | per-token lr selectivity; align write/query key geometry |

At every N we've measured, the **binding** ceiling is the **delta-rule lr (dominant)**, then real-key
*anisotropy* + the streamed write↔query mismatch — *never* the module size or the value dimension, and
*not* a hard key-rank wall (rank-46 holds ~1000 with a gentle update).

## 4. The deep fork: linear readout vs softmax (why "it becomes retrieval at scale")
`M q = Σ_i (k_i·q) v_i` is **linear attention** — capacity is bounded by superposition cross-talk and
key crowding. `softmax(β q·K) V` over *kept* k/v pairs (modern Hopfield / transformer attention /
kNN / RAG) suppresses cross-talk exponentially and has far higher capacity (probe D). The crux:
**to exploit the full high-d geometry you need the softmax nonlinearity, which means keeping the
individual k/v pairs** — i.e. slot/retrieval memory. The dense parametric store's distinctive
property (no text at query time, content persisted in weights, cross-session) is exactly what you'd
trade away. So the strategic question is: how far can a *parametric* (collapsed-M) store be pushed
before slot/product-key/retrieval is simply the right tool?

## 5. Open R&D questions (for review)
1. **Update rule for capacity (highest priority — CPU-confirmed ~10× lever).** Probe E: at the real key
   rank 46, dropping init lr θ from 1.0→0.03 moves the static ceiling from ~64 to ~1000 facts (Hebbian
   ~1000); the error-correction is the dominant capacity killer at *every* rank. The lr gate sits at θ≈1
   only because training never demanded >6 facts. Open: the right capacity/interference trade — a θ
   schedule (bold when empty, gentle when full)? novelty-gated lr (full θ to *add*, ~0 to not erode,
   gated by a `qᵀM₂q` match score)? Hebbian + forget-gate cleanup? **Fastest informative GPU experiment:
   train with `max_lr≈0.06` (+ whitening) and re-run `eval_real --scale` — does the ~10× survive real
   anisotropic keys + the streamed write↔query mismatch? (Probe E can't see those.)**
2. **Key rank lifting.** Capacity ∝ effective key rank, capped at ~46 by the frozen base's answer-prefix
   geometry. A linear encoder can't exceed the input rank; a 2-layer GELU encoder lifts PR 20→35 in a
   toy probe. How high can rank go via (a) deeper/nonlinear encoders, (b) **earlier-layer keys**
   (pre-collapse, higher-rank — untested, likely highest-leverage), (c) multi-token/positional keys,
   (d) explicit decorrelation penalties? Is there a hard rank ceiling intrinsic to a frozen base?
3. **Value-side ceiling.** d_v=1152 is fixed by tied embeddings. Probes show it's not binding ≤ ~10²,
   but at 10³–10⁴ distinct answers must stay separable in the 1152-dim logit readout. Does it bind?
   Does an untied / wider value space (decoupled from the embedding) help, at what cost to the
   near-closed-form readout?
4. **Streaming/deployment geometry.** The write key (fact in session context) and query key (standalone
   prompt) differ for the same fact, and `ingest()` writes every token of a statement. How much of the
   ~128(static)→~8–16(streamed) gap is this, and does single-fact-span writing + a write/query
   consistency objective close it?
5. **Where does parametric beat retrieval?** Given (4)'s honest small-N regime, what is the actual
   value proposition vs embed+kNN — and is it cross-session persistence, no-query-time-context,
   editability, or composition? Define the regime where the parametric store wins before scaling it.
6. **Architecture path by scale.** dense + better-update + higher-rank-keys (10²–10³?) vs **product-key
   memory** (Lample et al.; capacity = product of sub-key spaces, the only candidate that grows
   multiplicatively) vs slot/kNN (= learned RAG, unbounded but loses the parametric property). At what N
   does each become correct?

## 6. Proposed experiment: 3-layer attribution ladder
A failure at N=10^k must be assignable to exactly one ceiling. Three nested probes, same store/head/
corpus, differing by one degradation, swept over `d_k × {update rule} × {whiten} × {ingest regime} ×
corpus-scale (SQuAD→NaturalQuestions for 10⁴ distinct entities)`:
- **L0 module** — oracle orthonormal keys, batched write (`capacity_scaling.py` / `capacity_probe.py`).
- **L1 key-quality** — real encoder keys, batched write (`diagnose_keys.py` raw-vs-whitened + measured PR/cos).
- **L2 streaming** — real keys, one-fact-per-session + save/reload (`eval_real.py --scale`).
Each pairwise drop L0→L1→L2 attributes the loss. Gates per decade on **worst-case** frac_top5 +
min_lift>0 + neutral_kl<0.2, ≥3 held-out seeds. The single highest-value first run is **the update-rule
swap** (probe B says it may be worth ~10×) — cheap, no retraining of structure.

## 7. Honest framing
Mechanism is proven on real held-out text at small N (verbatim recall ≈ 81% of the in-context ceiling
with context destroyed + persisted; generative EM 0.45; A4 retention 1.0). **Scale to 10³–10⁴ is open,
not closed** — gated by the update rule, key-addressing rank, and streamed-write dynamics, with the
store size and value dimension ample. If none of those levers breaks past ~10², the honest conclusion
is that this is a small-N parametric-memory technique and retrieval is the right tool at scale — but
we have not yet run the experiments that would decide it.
