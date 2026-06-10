"""
Standalone meta-trainer for the memory skill, running entirely on **cached
post-norm activations** — the base model is never in the training loop.

Design doc §3.  With the post-final-norm splice, the gradient path is::

    CE/KL → tied head (a single matmul against the frozen embedding table)
          → (hn + delta) → read_gate/out_proj → store read
          → BPTT through the analytic delta-rule writes
          → key/query encoders and gates

and never enters a transformer block.  Each training step is a handful of GEMMs
over cached tensors, so the ~1–10 M-parameter skill trains at toy-model speed; on
an L4 the wall-clock is dominated by building the cache once.

Objective (design doc §1/Q5)::

    L = CE(answer tokens of recall-type episodes)
      + λ_KL · KL( p_aug ‖ p_base )  over positions where memory is irrelevant

The KL term is the explicit A2/abstention pressure: it trains the read gate (and
the lr gate's selectivity) toward "defer to the base" wherever the store holds
nothing useful — including the *non-answer* positions of recall episodes.

Batch schema (one padded batch; produced by ``collate`` from per-episode dicts —
the cache builder and the toy test both emit this schema):

    sessions:  list over session-index of dicts with
        hn        [B, T, d]   post-norm hiddens (zeros where padded)
        e_next    [B, T, d]   embedding rows of the next token
        mask      [B, T]      1 = real token (padding excluded from writes)
        fact_mask [B, T]      1 = token inside a fact statement (diagnostics /
                              optional fact-only-write curriculum)
    query:
        hn        [B, Tq, d]
        mask      [B, Tq]
        ce_pos    [B, P]      positions whose logits predict answer tokens (−1 pad)
        ce_tgt    [B, P]      gold token ids (−1 pad)
        ce_in_loss[B]         bool: CE counted (recall/multifact/multisession)
        kl_mask   [B, Tq]     positions for the KL-to-base term
        probe_pos / probe_tgt [B, P]  eval-only golds (control episodes), −1 pad
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rlm.memory.skill import MemorySkill


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    steps: int = 2000
    batch_size: int = 32
    lambda_kl: float = 0.5
    grad_clip: float = 1.0
    # Curriculum: write only fact-span tokens for the first N steps (lets the
    # binding form before the lr gate must learn selectivity on filler).
    fact_only_write_steps: int = 0
    log_every: int = 25
    eval_every: int = 200
    device: str = "cpu"
    seed: int = 0


def pad_stack(tensors: list[Tensor], pad_value: float = 0.0) -> Tensor:
    """Stack variable-length [T, ...] tensors into [B, Tmax, ...]."""
    tmax = max(t.shape[0] for t in tensors)
    out = tensors[0].new_full((len(tensors), tmax, *tensors[0].shape[1:]), pad_value)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out


def collate(episodes: list[dict], device: str | torch.device, dtype=torch.float32) -> dict:
    """Episodes (dicts of per-episode tensors as documented above, unpadded) →
    one padded batch.  Episodes may have different numbers of sessions; missing
    sessions become all-padding."""
    n_sessions = max(len(ep["sessions"]) for ep in episodes)
    d = episodes[0]["sessions"][0]["hn"].shape[-1]
    sessions = []
    for s in range(n_sessions):
        hn, en, mask, fmask = [], [], [], []
        for ep in episodes:
            if s < len(ep["sessions"]):
                sess = ep["sessions"][s]
                hn.append(sess["hn"].to(dtype))
                en.append(sess["e_next"].to(dtype))
                mask.append(torch.ones(sess["hn"].shape[0]))
                fmask.append(sess["fact_mask"].to(torch.float32))
            else:
                hn.append(torch.zeros(1, d))
                en.append(torch.zeros(1, d))
                mask.append(torch.zeros(1))
                fmask.append(torch.zeros(1))
        sessions.append(
            {
                "hn": pad_stack(hn).to(device),
                "e_next": pad_stack(en).to(device),
                "mask": pad_stack([m.unsqueeze(-1) for m in mask]).squeeze(-1).to(device),
                "fact_mask": pad_stack([m.unsqueeze(-1) for m in fmask]).squeeze(-1).to(device),
            }
        )
    q_hn = pad_stack([ep["query"]["hn"].to(dtype) for ep in episodes]).to(device)
    q_mask = pad_stack(
        [torch.ones(ep["query"]["hn"].shape[0], 1) for ep in episodes]
    ).squeeze(-1).to(device)
    pmax = max(max(ep["query"]["ce_pos"].shape[0], ep["query"]["probe_pos"].shape[0], 1) for ep in episodes)

    def pad_idx(key: str) -> Tensor:
        rows = []
        for ep in episodes:
            t = ep["query"][key].to(torch.long)
            rows.append(F.pad(t, (0, pmax - t.shape[0]), value=-1))
        return torch.stack(rows).to(device)

    return {
        "sessions": sessions,
        "query": {
            "hn": q_hn,
            "mask": q_mask,
            "ce_pos": pad_idx("ce_pos"),
            "ce_tgt": pad_idx("ce_tgt"),
            "ce_in_loss": torch.tensor([bool(ep["query"]["ce_in_loss"]) for ep in episodes], device=device),
            "kl_mask": pad_stack([ep["query"]["kl_mask"].to(torch.float32).unsqueeze(-1) for ep in episodes]).squeeze(-1).to(device),
            "probe_pos": pad_idx("probe_pos"),
            "probe_tgt": pad_idx("probe_tgt"),
        },
    }


class EpisodeTrainer:
    """Owns the skill, the frozen tied head, the optimizer, and the loop."""

    def __init__(
        self,
        skill: MemorySkill,
        head_weight: Tensor,  # [V, d_model] — the (tied) embedding table, frozen
        cfg: TrainerConfig,
        logit_softcap: float | None = None,
    ):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.skill = skill.to(cfg.device)
        self.W = head_weight.detach().to(cfg.device, torch.float32)
        self.W.requires_grad_(False)
        self.softcap = logit_softcap
        self.opt = torch.optim.AdamW(self.skill.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.step_idx = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def logits(self, hidden: Tensor) -> Tensor:
        z = hidden @ self.W.T
        if self.softcap is not None:
            z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def run_memory(self, batch: dict, fact_only_write: bool, zero_store: bool = False):
        """Write all sessions in order (state carried across), read on the query."""
        B = batch["query"]["hn"].shape[0]
        state = self.skill.init_state(B, device=self.cfg.device)
        if not zero_store:
            for sess in batch["sessions"]:
                wm = sess["mask"] * (sess["fact_mask"] if fact_only_write else 1.0)
                state = self.skill.write(sess["hn"], sess["e_next"], state, write_mask=wm)
        delta, aux = self.skill.read_delta(batch["query"]["hn"], state)
        return self.logits(batch["query"]["hn"] + delta), aux, state

    # ------------------------------------------------------------------
    @staticmethod
    def gather_logprobs(logp: Tensor, pos: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
        """logp [B,T,V], pos/tgt [B,P] with −1 padding → (per-slot logprob, valid mask)."""
        valid = (pos >= 0) & (tgt >= 0)
        pos_c = pos.clamp(min=0)
        tgt_c = tgt.clamp(min=0)
        rows = logp.gather(1, pos_c.unsqueeze(-1).expand(-1, -1, logp.shape[-1]))  # [B,P,V]
        lp = rows.gather(2, tgt_c.unsqueeze(-1)).squeeze(-1)  # [B,P]
        return lp, valid

    def loss_on_batch(self, batch: dict) -> tuple[Tensor, dict]:
        cfg = self.cfg
        fact_only = self.step_idx < cfg.fact_only_write_steps
        logits_aug, aux, _ = self.run_memory(batch, fact_only_write=fact_only)
        with torch.no_grad():
            logits_base = self.logits(batch["query"]["hn"])

        logp_aug = F.log_softmax(logits_aug, dim=-1)
        q = batch["query"]

        # --- CE on answer tokens (recall-type episodes only) -------------
        lp, valid = self.gather_logprobs(logp_aug, q["ce_pos"], q["ce_tgt"])
        valid = valid & q["ce_in_loss"].unsqueeze(-1)
        n_ce = valid.sum().clamp(min=1)
        ce = -(lp * valid).sum() / n_ce

        # --- KL(p_aug ‖ p_base) on memory-irrelevant positions ------------
        kl_mask = q["kl_mask"] * q["mask"]
        log_p_base = F.log_softmax(logits_base, dim=-1)
        kl_tok = F.kl_div(log_p_base, logp_aug, reduction="none", log_target=True).sum(-1)
        kl = (kl_tok * kl_mask).sum() / kl_mask.sum().clamp(min=1.0)

        loss = ce + cfg.lambda_kl * kl

        # --- diagnostics ----------------------------------------------------
        with torch.no_grad():
            gate = aux["read_gate"]
            metrics = {
                "loss": float(loss),
                "ce": float(ce),
                "kl": float(kl),
                "read_gate_mean": float((gate * q["mask"]).sum() / q["mask"].sum().clamp(min=1)),
                "fact_only_write": bool(fact_only),
            }
        return loss, metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, batch: dict) -> dict:
        """A1-style lift on a held-out batch: gold logprob with the written store
        vs the zeroed store, top-k hit rate, and control-probe lift."""
        self.skill.eval()
        logits_mem, _, _ = self.run_memory(batch, fact_only_write=False)
        logits_empty, _, _ = self.run_memory(batch, fact_only_write=False, zero_store=True)
        lp_mem = F.log_softmax(logits_mem, dim=-1)
        lp_empty = F.log_softmax(logits_empty, dim=-1)
        q = batch["query"]

        out: dict = {}
        for name, pos_key, tgt_key, gate_key in (
            ("recall", "ce_pos", "ce_tgt", "ce_in_loss"),
            ("control_probe", "probe_pos", "probe_tgt", None),
        ):
            a, valid = self.gather_logprobs(lp_mem, q[pos_key], q[tgt_key])
            b, _ = self.gather_logprobs(lp_empty, q[pos_key], q[tgt_key])
            if gate_key is not None:
                valid = valid & q[gate_key].unsqueeze(-1)
            n = valid.sum()
            if int(n) == 0:
                continue
            lift = ((a - b) * valid).sum() / n
            # first-token top-5: rank of gold among logits at the first valid slot
            first = valid.float().argmax(dim=1)
            rows = torch.arange(valid.shape[0], device=valid.device)
            has = valid.any(dim=1)
            pos0 = q[pos_key][rows, first].clamp(min=0)
            tgt0 = q[tgt_key][rows, first].clamp(min=0)
            row_logits = logits_mem[rows, pos0]
            rank = (row_logits > row_logits.gather(1, tgt0.unsqueeze(-1))).sum(dim=1)
            top5 = ((rank < 5) & has).float().sum() / has.float().sum().clamp(min=1)
            out[f"{name}_lift_nats"] = float(lift)
            out[f"{name}_top5"] = float(top5)
        self.skill.train()
        return out

    # ------------------------------------------------------------------
    def train(self, sample_batch, eval_batch: dict | None = None, log_path: str | None = None) -> list[dict]:
        """``sample_batch`` is a callable () → collated batch (so the caller owns
        data loading / mixing / curriculum)."""
        cfg = self.cfg
        self.skill.train()
        t0 = time.time()
        log_f = open(log_path, "a") if log_path else None
        try:
            for _ in range(cfg.steps):
                batch = sample_batch()
                loss, metrics = self.loss_on_batch(batch)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.skill.parameters(), cfg.grad_clip)
                self.opt.step()
                self.step_idx += 1
                if self.step_idx % cfg.log_every == 0 or self.step_idx == 1:
                    metrics.update(step=self.step_idx, wall_s=round(time.time() - t0, 1))
                    if eval_batch is not None and (
                        self.step_idx % cfg.eval_every == 0 or self.step_idx == 1
                    ):
                        metrics.update(self.evaluate(eval_batch))
                    self.history.append(metrics)
                    line = json.dumps(metrics)
                    print(line, flush=True)
                    if log_f:
                        log_f.write(line + "\n")
                        log_f.flush()
        finally:
            if log_f:
                log_f.close()
        return self.history


def trainer_config_to_json(cfg: TrainerConfig) -> str:
    return json.dumps(asdict(cfg), indent=2, default=str)


def nats_to_bits(x: float) -> float:
    return x / math.log(2.0)
