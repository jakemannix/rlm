"""
Live cross-session memory on a frozen HF base: the deployment counterpart of the
cached-activation trainer, sharing the same :class:`MemorySkill` so there is no
train/deploy skew (design doc §5.8).

Modes are asymmetric by construction (design doc §2): ``ingest`` is write-only
(no injection while writing — the write loop never sees its own outputs) and
recall is read-only (writing the query would delta-rule-overwrite the stored
association at the same key with a not-yet-known value).

A4 surface: :meth:`save` / :meth:`load` round-trip the store matrix only;
momentum never crosses a session boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from rlm.memory.cache import encode_segments, get_head, postnorm_hiddens
from rlm.memory.episodes import EpisodeText
from rlm.memory.linear_store import StoreState
from rlm.memory.skill import MemorySkill


class MemorySession:
    def __init__(self, model, tok, skill: MemorySkill, device: str | torch.device | None = None):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = tok
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.skill = skill.to(self.device).eval()
        W, softcap = get_head(model)
        self.W = W.detach().to(self.device, torch.float32)
        self.softcap = softcap
        self.state: StoreState = self.skill.init_state(1, device=self.device)

    # ------------------------------------------------------------------
    def logits(self, hidden: Tensor) -> Tensor:
        z = hidden.to(torch.float32) @ self.W.T
        if self.softcap is not None:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def ids_for(self, text: str, with_bos: bool = True) -> Tensor:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if with_bos and self.tok.bos_token_id is not None:
            ids = [self.tok.bos_token_id] + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Ingest (write-only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ingest_ids(self, ids: Tensor, write_mask: Tensor | None = None) -> None:
        """Write a token sequence: keys from post-norm h at 0..T−2, values from
        the embedding rows of tokens 1..T−1 (the answer-shifted binding)."""
        hn = postnorm_hiddens(self.model, ids).to(torch.float32)
        e_next = self.W[ids[0, 1:]].unsqueeze(0)
        wm = write_mask[:, : ids.shape[1] - 1] if write_mask is not None else None
        self.state = self.skill.write(hn[:, :-1], e_next, self.state, write_mask=wm).detach()

    @torch.no_grad()
    def ingest(self, text: str) -> None:
        self.ingest_ids(self.ids_for(text))

    @torch.no_grad()
    def ingest_episode(self, ep: EpisodeText, fact_only: bool = False) -> None:
        """Ingest every session of an :class:`EpisodeText`, optionally writing
        only fact-span tokens (the oracle Step-1 configuration)."""
        bos = self.tok.bos_token_id
        for segs in ep.sessions:
            ids, fact_mask = encode_segments(self.tok, segs, bos)
            ids = ids.unsqueeze(0).to(self.device)
            wm = fact_mask.unsqueeze(0).to(self.device) if fact_only else None
            self.ingest_ids(ids, write_mask=wm)

    # ------------------------------------------------------------------
    # Recall (read-only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_answer(
        self,
        prompt: str,
        answer: str,
        read_scale: float = 1.0,
        use_store: bool = True,
    ) -> dict:
        """Teacher-forced gold logprobs for ``answer`` after ``prompt`` with the
        current store vs the empty store (the A1 metric and its control)."""
        p_ids = self.ids_for(prompt)
        a_ids = self.ids_for(answer, with_bos=False)
        full = torch.cat([p_ids, a_ids], dim=1)
        hn = postnorm_hiddens(self.model, full).to(torch.float32)

        state = self.state if use_store else self.state.zeroed()
        delta, aux = self.skill.read_delta(hn, state)
        logp_mem = F.log_softmax(self.logits(hn + read_scale * delta), dim=-1)
        logp_base = F.log_softmax(self.logits(hn), dim=-1)

        P, n = p_ids.shape[1], a_ids.shape[1]
        pos = torch.arange(P - 1, P - 1 + n, device=self.device)
        tgt = a_ids[0]
        gold_mem = logp_mem[0, pos].gather(1, tgt.unsqueeze(-1)).squeeze(-1)
        gold_base = logp_base[0, pos].gather(1, tgt.unsqueeze(-1)).squeeze(-1)
        first_logits = (hn + read_scale * delta)[0, pos[0]] @ self.W.T
        rank = int((first_logits > first_logits[tgt[0]]).sum())
        return {
            "gold_logprob": float(gold_mem.sum()),
            "gold_logprob_base": float(gold_base.sum()),
            "first_token_lift_nats": float(gold_mem[0] - gold_base[0]),
            "lift_nats": float((gold_mem - gold_base).sum()),
            "first_token_rank": rank,
            "first_token_top5": rank < 5,
            "read_gate_mean": float(aux["read_gate"][0, pos].mean()),
        }

    @torch.no_grad()
    def generate_greedy(self, prompt: str, max_new_tokens: int = 12, read_scale: float = 1.0) -> str:
        ids = self.ids_for(prompt)
        for _ in range(max_new_tokens):
            hn = postnorm_hiddens(self.model, ids).to(torch.float32)
            delta, _ = self.skill.read_delta(hn[:, -1:], self.state)
            logits = self.logits(hn[:, -1:] + read_scale * delta)
            nxt = logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        return self.tok.decode(ids[0, self.ids_for(prompt).shape[1] :])

    # ------------------------------------------------------------------
    # Distribution-shift probe (A2): how much does the loaded store perturb the
    # base on text the memory knows nothing about?
    # ------------------------------------------------------------------
    @torch.no_grad()
    def neutral_kl(self, text: str) -> float:
        ids = self.ids_for(text)
        hn = postnorm_hiddens(self.model, ids).to(torch.float32)
        delta, _ = self.skill.read_delta(hn, self.state)
        logp_aug = F.log_softmax(self.logits(hn + delta), dim=-1)
        logp_base = F.log_softmax(self.logits(hn), dim=-1)
        kl = F.kl_div(logp_base, logp_aug, reduction="none", log_target=True).sum(-1)
        return float(kl[0, :-1].mean())

    # ------------------------------------------------------------------
    # Persistence (A4)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        self.skill.store.save_state(self.state, path)

    def load(self, path: str) -> None:
        self.state = self.skill.store.load_state(path, device=self.device)

    def reset(self) -> None:
        self.state = self.skill.init_state(1, device=self.device)
