"""
Episode tokenisation + post-norm activation caching against a real HF model.

Two layers, deliberately separated:

- :func:`tokenize_episode` is **pure** (episode text + tokenizer → ids, masks,
  CE/probe positions).  It is tested offline with a mock tokenizer
  (``tests/test_episodes.py``); the alignment rules live in exactly one place,
  shared by the cache builder and the live :class:`rlm.memory.session.MemorySession`.
- :func:`build_cache` runs the frozen base **once** over the tokenised episodes
  and stores post-final-norm hiddens + next-token **ids** (the trainer gathers
  embedding rows from ``head.pt`` per batch — halves the cache), after which
  meta-training never touches the base (design doc §3).  Session tensors keep
  only positions 0..T−2: position t binds to token t+1, so the final position
  has nothing to write and is dropped at the source.

Alignment convention: every segment starts with a leading space (the generator
guarantees this), and segments are tokenised independently with
``add_special_tokens=False`` then concatenated after BOS.  SentencePiece-style
tokenizers (Gemma) are stable under this; ``tokenize_episode`` verifies the
round-trip decode and fails fast on mismatch (AGENTS.md: fail loud).

CE positions are *predictor* positions: position p's logits predict token p+1,
so for a prompt of length P followed by answer tokens a₁..a_n, CE positions are
P−1, …, P+n−2 with targets a₁..a_n.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rlm.memory.episodes import EpisodeGenerator, EpisodeText


@dataclass
class TokenizedEpisode:
    episode_type: str
    relation_id: str | None
    sessions: list[dict]  # {"ids": Long[T], "fact_mask": Float[T]}
    query_ids: Tensor  # Long[Tq]  (prompt + answer + probe-answer-free, see below)
    ce_pos: Tensor  # Long[P]
    ce_tgt: Tensor  # Long[P]
    ce_in_loss: bool
    kl_mask: Tensor  # Float[Tq]
    probe_pos: Tensor
    probe_tgt: Tensor


def encode_segments(tok, segments, bos_id: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """Concatenate per-segment encodings after BOS.

    Returns ``(ids, fact_mask, answer_mask)`` — fact_mask covers the whole fact
    statement (fact + answer roles, preserving the original semantics);
    answer_mask covers answer-role tokens only.  NB masks are *position*-aligned;
    the write-selectivity mask wants *target*-aligned (position t writes the
    binding for token t+1), which the cache builder and session derive as
    ``answer_mask[1:]``."""
    ids: list[int] = [] if bos_id is None else [bos_id]
    fact: list[float] = [] if bos_id is None else [0.0]
    ans: list[float] = [] if bos_id is None else [0.0]
    for seg in segments:
        seg_ids = tok(seg.text, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        is_ans = seg.role == "answer"
        fact.extend([1.0 if (seg.role == "fact" or is_ans) else 0.0] * len(seg_ids))
        ans.extend([1.0 if is_ans else 0.0] * len(seg_ids))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(fact), torch.tensor(ans)


def tokenize_episode(ep: EpisodeText, tok, bos_id: int | None) -> TokenizedEpisode:
    sessions = []
    for segs in ep.sessions:
        ids, fact_mask, answer_mask = encode_segments(tok, segs, bos_id)
        decoded = tok.decode(ids[1:] if bos_id is not None else ids)
        joined = "".join(s.text for s in segs)
        if decoded.strip() != joined.strip():
            raise ValueError(
                f"segment tokenisation round-trip mismatch:\n {decoded!r}\n vs {joined!r}"
            )
        sessions.append({"ids": ids, "fact_mask": fact_mask, "answer_mask": answer_mask})

    prompt_ids = tok(ep.query_prompt, add_special_tokens=False)["input_ids"]
    prompt = ([bos_id] if bos_id is not None else []) + prompt_ids
    P = len(prompt)

    def span(answer: str | None) -> tuple[list[int], Tensor, Tensor]:
        if answer is None:
            return [], torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        a = tok(answer, add_special_tokens=False)["input_ids"]
        pos = torch.arange(P - 1, P - 1 + len(a), dtype=torch.long)
        return a, pos, torch.tensor(a, dtype=torch.long)

    # An episode has either a CE answer (recall-type) or a probe answer
    # (control) or neither (abstain); the answer tokens are appended for
    # teacher forcing in all cases that have one.
    ans_ids, ce_pos, ce_tgt = span(ep.answer)
    probe_ids, probe_pos, probe_tgt = span(ep.probe_answer)
    tail = ans_ids if ans_ids else probe_ids
    query_ids = torch.tensor(prompt + tail, dtype=torch.long)

    kl_mask = torch.ones(len(query_ids))
    kl_mask[ce_pos] = 0.0  # CE owns these; KL everywhere else (incl. probe pos)
    if len(query_ids) > 0:
        kl_mask[-1] = 0.0  # last position predicts beyond the sequence

    return TokenizedEpisode(
        ep.episode_type,
        ep.queried_fact.relation_id if ep.queried_fact is not None else None,
        sessions,
        query_ids,
        ce_pos,
        ce_tgt,
        ep.answer is not None,
        kl_mask,
        probe_pos,
        probe_tgt,
    )


# ---------------------------------------------------------------------------
# Model-side helpers (shared with session.py)
# ---------------------------------------------------------------------------
def get_decoder(model) -> torch.nn.Module:
    """The pre-head decoder whose ``last_hidden_state`` is post-final-norm.

    HF Llama/Gemma-style ``*Model.forward`` applies the final RMSNorm before
    returning, so ``decoder(...).last_hidden_state`` is exactly the post-norm
    stream the design splices at.  Fail fast if the layout is unfamiliar.
    """
    for attr in ("model", "language_model"):
        dec = getattr(model, attr, None)
        if dec is not None and hasattr(dec, "norm") and hasattr(dec, "embed_tokens"):
            return dec
        if dec is not None and hasattr(dec, "model"):  # nested (multimodal wrappers)
            inner = dec.model
            if hasattr(inner, "norm") and hasattr(inner, "embed_tokens"):
                return inner
    raise ValueError(f"cannot locate a norm+embed_tokens decoder on {type(model).__name__}")


def get_head(model) -> tuple[Tensor, float | None]:
    """(tied head weight [V, d], final logit softcap or None) — verified tied."""
    W_in = model.get_input_embeddings().weight
    W_out = model.get_output_embeddings().weight
    if W_in.data_ptr() != W_out.data_ptr() and not torch.equal(W_in, W_out):
        raise ValueError(
            "input/output embeddings are not tied; the value-codebook design "
            "(design doc §0) assumes tied embeddings — re-derive out_proj for this base."
        )
    softcap = getattr(model.config, "final_logit_softcapping", None)
    return W_in, softcap


@torch.no_grad()
def postnorm_hiddens(model, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    dec = get_decoder(model)
    out = dec(input_ids=input_ids, attention_mask=attention_mask)
    hn = out.last_hidden_state
    if not torch.isfinite(hn).all():
        raise FloatingPointError("non-finite post-norm hiddens from the base")
    return hn


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_texts(model, rows: list[Tensor], device, batch_size: int = 16) -> list[Tensor]:
    """Forward ragged id-rows through the decoder; return per-row [T, d] hiddens."""
    outs: list[Tensor] = [None] * len(rows)  # type: ignore[list-item]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))  # length-bucketed
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        tmax = max(len(rows[i]) for i in idx)
        ids = torch.zeros(len(idx), tmax, dtype=torch.long, device=device)
        mask = torch.zeros(len(idx), tmax, dtype=torch.long, device=device)
        for r, i in enumerate(idx):
            ids[r, : len(rows[i])] = rows[i].to(device)
            mask[r, : len(rows[i])] = 1
        hn = postnorm_hiddens(model, ids, mask)
        for r, i in enumerate(idx):
            outs[i] = hn[r, : len(rows[i])].to("cpu")
    return outs


def build_cache(
    model,
    tok,
    out_dir: str,
    n_episodes: int = 20_000,
    mix: dict[str, float] | None = None,
    generator: EpisodeGenerator | None = None,
    seed: int = 0,
    paraphrase_prob: float = 0.0,
    batch_size: int = 16,
    shard_size: int = 1_000,
    dtype: torch.dtype = torch.float16,
    overwrite: bool = False,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resume-by-completion / anti-clobber: a matching finished cache is skipped;
    # a stale/mismatched one must be explicitly overwritten (else a re-run with
    # fewer episodes would leave stale shards that train_skill silently mixes in).
    existing = sorted(out.glob("shard_*.pt"))
    meta_path = out / "meta.json"
    if existing and meta_path.exists():
        try:
            m = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            m = {}
        matches = (
            m.get("n_episodes") == n_episodes
            and m.get("seed") == seed
            and m.get("paraphrase_prob") == paraphrase_prob
            and m.get("mix") == mix
        )
        if matches and (out / "head.pt").exists():
            print(
                f"cache already complete at {out} ({m.get('n_episodes')} episodes); "
                f"skipping (pass overwrite=True to rebuild)",
                flush=True,
            )
            return
    if (existing or meta_path.exists()) and not overwrite:
        raise FileExistsError(
            f"{out} already contains cache artifacts that don't match this config; pass "
            f"--overwrite to rebuild (deletes stale shards/head.pt/meta.json) or choose a fresh --out"
        )
    if overwrite:
        for p in [*existing, out / "head.pt", meta_path]:
            p.unlink(missing_ok=True)

    device = next(model.parameters()).device
    W, softcap = get_head(model)
    torch.save(
        {
            "W": W.detach().to(dtype).cpu(),
            "softcap": softcap,
            "d_model": W.shape[1],
            "model_name": getattr(model.config, "name_or_path", "?"),
        },
        out / "head.pt",
    )

    gen = generator if generator is not None else EpisodeGenerator(seed=seed, paraphrase_prob=paraphrase_prob)
    bos = tok.bos_token_id
    counts: dict[str, int] = {}
    shard: list[dict] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard, shard_idx
        if shard:
            torch.save(shard, out / f"shard_{shard_idx:04d}.pt")
            shard_idx += 1
            shard = []

    pending: list[TokenizedEpisode] = []

    def process(batch: list[TokenizedEpisode]) -> None:
        rows = [s["ids"] for te in batch for s in te.sessions] + [te.query_ids for te in batch]
        hiddens = encode_texts(model, rows, device, batch_size)
        i = 0
        sess_h: list[list[Tensor]] = []
        for te in batch:
            sess_h.append(hiddens[i : i + len(te.sessions)])
            i += len(te.sessions)
        q_h = hiddens[i:]
        for te, sh, qh in zip(batch, sess_h, q_h, strict=True):
            # Write tensors keep only positions 0..T-2: position t binds to the
            # *next* token, so the last position has no value and is dropped at
            # the source rather than masked downstream.
            sessions = [
                {
                    "hn": h[:-1].to(dtype),
                    "e_next_ids": sess["ids"][1:].clone(),
                    "fact_mask": sess["fact_mask"][:-1].clone(),
                    # target-aligned: position t's write binds token t+1, so the
                    # answer-binding positions are those whose TARGET is an
                    # answer token (includes the prompt-final position).
                    "answer_mask": sess["answer_mask"][1:].clone(),
                }
                for sess, h in zip(te.sessions, sh, strict=True)
            ]
            shard.append(
                {
                    "type": te.episode_type,
                    "relation": te.relation_id,
                    "sessions": sessions,
                    "query": {
                        "hn": qh.to(dtype),
                        "ce_pos": te.ce_pos,
                        "ce_tgt": te.ce_tgt,
                        "ce_in_loss": te.ce_in_loss,
                        "kl_mask": te.kl_mask,
                        "probe_pos": te.probe_pos,
                        "probe_tgt": te.probe_tgt,
                    },
                }
            )
            counts[te.episode_type] = counts.get(te.episode_type, 0) + 1
        while len(shard) >= shard_size:
            keep = shard[shard_size:]
            del shard[shard_size:]
            flush()
            shard.extend(keep)

    skipped = 0
    for n in range(n_episodes):
        # Real-corpus answers (arbitrary spans) can fail the tokenizer round-trip
        # guard; skip those episodes rather than abort a long build.
        try:
            pending.append(tokenize_episode(gen.sample_mix(mix), tok, bos))
        except ValueError:
            skipped += 1
            continue
        if len(pending) >= 64:
            process(pending)
            pending = []
        if (n + 1) % 1000 == 0:
            print(f"cached {n + 1}/{n_episodes} episodes ({skipped} skipped)", flush=True)
    if pending:
        process(pending)
    flush()
    (out / "meta.json").write_text(
        json.dumps(
            {
                "n_episodes": n_episodes,
                "counts": counts,
                "seed": seed,
                "paraphrase_prob": paraphrase_prob,
                "mix": mix,
            },
            indent=2,
        )
    )
    print(f"done: {n_episodes} episodes → {out}", flush=True)
