"""End-to-end meta-training on a synthetic 'base model' world (CPU, seconds).

This closes the whole v1 loop with no GPU and no HF model: cached post-norm
hiddens → key encoding → gated delta-rule writes (BPTT) → read → post-norm
injection → tied head → CE + KL.  It is the strongest de-risking test the repo
can run offline: if this fails, nothing downstream matters; if it passes, the
remaining risk on real Gemma is *representation* quality (oracle scripts), not
the learning machinery.

Toy world: a random embedding table E [V, d] plays the tied head.  Each fact i
has a latent 'question representation' u_i; ingest sequences contain u_i at a
known position with e_next = E[answer_i] (the answer-shifted value), surrounded
by junk positions; query sequences end in a noisy copy of u_i.  A skill with
orthogonal-init encoders must learn to bind and recall across the (simulated)
session boundary, while controls stay flat.
"""

import pytest

torch = pytest.importorskip("torch")

from rlm.memory.linear_store import LinearStoreConfig  # noqa: E402
from rlm.memory.skill import MemorySkill, SkillConfig  # noqa: E402
from rlm.memory.trainer import EpisodeTrainer, TrainerConfig, collate  # noqa: E402

D_MODEL, D_K, VOCAB, N_FACTS = 32, 16, 64, 24
T_INGEST, T_QUERY = 8, 5


class ToyWorld:
    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.functional.normalize(torch.randn(VOCAB, D_MODEL, generator=g), dim=-1) * (D_MODEL ** 0.5) * 0.5
        self.u = torch.nn.functional.normalize(torch.randn(N_FACTS, D_MODEL, generator=g), dim=-1) * (D_MODEL ** 0.5)
        self.answers = torch.randint(0, VOCAB, (N_FACTS,), generator=g)
        self.g = g

    def junk(self, t: int) -> torch.Tensor:
        return torch.randn(t, D_MODEL, generator=self.g)

    def episode(self, i: int | None = None, episode_type: str = "recall") -> dict:
        """Emit the trainer's per-episode schema (see trainer.collate)."""
        i = int(torch.randint(0, N_FACTS, (1,), generator=self.g)) if i is None else i
        fact_pos = int(torch.randint(1, T_INGEST - 1, (1,), generator=self.g))
        hn = self.junk(T_INGEST)
        hn[fact_pos] = self.u[i] + 0.05 * torch.randn(D_MODEL, generator=self.g)
        e_next = self.E[torch.randint(0, VOCAB, (T_INGEST,), generator=self.g)].clone()
        e_next[fact_pos] = self.E[self.answers[i]]
        fact_mask = torch.zeros(T_INGEST)
        fact_mask[fact_pos] = 1.0

        q_hn = self.junk(T_QUERY)
        # control: query a *different* fact than was ingested
        j = i
        if episode_type == "control":
            while j == i:
                j = int(torch.randint(0, N_FACTS, (1,), generator=self.g))
        q_hn[-1] = self.u[j] + 0.1 * torch.randn(D_MODEL, generator=self.g)

        is_recall = episode_type == "recall"
        kl_mask = torch.ones(T_QUERY)
        kl_mask[-1] = 0.0 if is_recall else 1.0
        empty = torch.zeros(0, dtype=torch.long)
        return {
            "sessions": [{"hn": hn, "e_next": e_next, "fact_mask": fact_mask}],
            "query": {
                "hn": q_hn,
                "ce_pos": torch.tensor([T_QUERY - 1]) if is_recall else empty,
                "ce_tgt": self.answers[i : i + 1] if is_recall else empty,
                "ce_in_loss": is_recall,
                "kl_mask": kl_mask,
                "probe_pos": empty if is_recall else torch.tensor([T_QUERY - 1]),
                "probe_tgt": empty if is_recall else self.answers[j : j + 1],
            },
        }


@pytest.fixture(scope="module")
def trained() -> tuple[EpisodeTrainer, dict]:
    torch.manual_seed(0)
    world = ToyWorld()
    skill = MemorySkill(
        SkillConfig(d_model=D_MODEL, d_k=D_K, store=LinearStoreConfig(chunk_size=2))
    )
    cfg = TrainerConfig(steps=300, batch_size=24, lr=3e-3, lambda_kl=0.3, log_every=100, eval_every=10_000, seed=0)
    trainer = EpisodeTrainer(skill, world.E, cfg)

    def sample():
        eps = [world.episode(episode_type="recall") for _ in range(20)]
        eps += [world.episode(episode_type="control") for _ in range(4)]
        return collate(eps, cfg.device)

    eval_batch = collate(
        [world.episode(i=k % N_FACTS, episode_type="recall") for k in range(32)]
        + [world.episode(episode_type="control") for _ in range(16)],
        cfg.device,
    )
    trainer.train(sample, eval_batch=None)
    return trainer, eval_batch


def test_loss_decreases(trained):
    trainer, _ = trained
    first = trainer.history[0]["loss"]
    last = trainer.history[-1]["loss"]
    assert last < first * 0.7, f"loss {first:.3f} → {last:.3f}"


def test_cross_session_recall_lift(trained):
    """A1 in miniature: gold logprob with the written store must beat the zeroed
    store by ≥ 2 nats with top-5 recall — and the empty store is the control."""
    trainer, eval_batch = trained
    m = trainer.evaluate(eval_batch)
    assert m["recall_lift_nats"] > 2.0, f"lift only {m['recall_lift_nats']:.2f} nats"
    assert m["recall_top5"] > 0.8, f"top5 only {m['recall_top5']:.2f}"


def test_wrong_fact_control_stays_flat(trained):
    """A1 control (ii): ingesting fact A must not lift fact B's gold."""
    trainer, eval_batch = trained
    m = trainer.evaluate(eval_batch)
    assert abs(m["control_probe_lift_nats"]) < 0.5, (
        f"control lift {m['control_probe_lift_nats']:.2f} nats — confidence smearing"
    )


def test_empty_store_is_noop_injection(trained):
    """Zero store ⇒ delta ≡ 0 ⇒ augmented logits == base logits exactly."""
    trainer, eval_batch = trained
    logits_empty, _, _ = trainer.run_memory(eval_batch, fact_only_write=False, zero_store=True)
    base = trainer.logits(eval_batch["query"]["hn"])
    assert torch.allclose(logits_empty, base, atol=1e-5)
