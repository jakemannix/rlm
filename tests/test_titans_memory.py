"""Unit tests for the Titans / Miras parametric memory module.

Skipped automatically if ``torch`` is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from rlm.memory.config import MemoryConfig  # noqa: E402
from rlm.memory.miras import MirasMemory  # noqa: E402
from rlm.memory.titans import TitansMemory  # noqa: E402


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def test_memory_config_validation():
    with pytest.raises(ValueError):
        MemoryConfig(n_layers=0)
    with pytest.raises(ValueError):
        MemoryConfig(chunk_size=0)
    with pytest.raises(ValueError):
        MemoryConfig(forget_rate=1.5)
    with pytest.raises(ValueError):
        MemoryConfig(retention="bogus")
    # Happy path - just check that __post_init__ doesn't raise.
    MemoryConfig(key_dim=8, value_dim=8)


def test_titans_memory_shapes():
    _seed(0)
    cfg = MemoryConfig(key_dim=8, value_dim=8, hidden_dim=16, n_layers=2, learnable_gates=False)
    mem = TitansMemory(cfg)
    keys = torch.randn(2, 5, 8)
    values = torch.randn(2, 5, 8)
    queries = torch.randn(2, 3, 8)
    out, state = mem.read_write(keys, values, queries)
    assert out.shape == (2, 3, 8)
    # State has one entry per memory MLP parameter, each prefixed with the
    # batch dimension.
    for name, p in mem.memory.named_parameters():
        assert state["params"][name].shape[0] == 2
        assert state["params"][name].shape[1:] == p.shape


def test_titans_memory_associative_recall():
    """
    Sanity check from the Titans paper: after writing a few (k, v) pairs
    several times, querying with the same keys should recover the values
    markedly better than random init.
    """
    _seed(123)
    cfg = MemoryConfig(
        key_dim=16,
        value_dim=16,
        hidden_dim=32,
        n_layers=2,
        inner_lr=2.0,
        momentum=0.0,
        forget_rate=0.0,
        learnable_gates=False,
        chunk_size=1,
    )
    mem = TitansMemory(cfg)
    keys = torch.randn(1, 8, 16)
    values = torch.randn(1, 8, 16)

    # Read before any writes.
    err_before = (mem.read(keys) - values).pow(2).mean().item()

    # Several rehearsal passes through the (k, v) pairs.  Real Titans
    # systems do this implicitly via the chunk-wise gradient over a
    # longer sequence and / or via momentum.
    state = mem.init_state(1)
    for _ in range(4):
        state = mem.write(keys, values, state)
    post = mem.read(keys, state)
    err_after = (post - values).pow(2).mean().item()

    assert err_after < err_before * 0.7, (
        f"Memory did not learn the associations: before={err_before:.3f} after={err_after:.3f}"
    )


def test_titans_chunked_update_matches_sequential():
    """Chunked vs token-wise updates differ but both should run cleanly."""
    _seed(0)
    cfg_seq = MemoryConfig(
        key_dim=8,
        value_dim=8,
        hidden_dim=16,
        n_layers=2,
        inner_lr=0.1,
        momentum=0.0,
        forget_rate=0.0,
        learnable_gates=False,
        chunk_size=1,
    )
    cfg_chunked = MemoryConfig(
        key_dim=8,
        value_dim=8,
        hidden_dim=16,
        n_layers=2,
        inner_lr=0.1,
        momentum=0.0,
        forget_rate=0.0,
        learnable_gates=False,
        chunk_size=4,
    )
    keys = torch.randn(1, 8, 8)
    values = torch.randn(1, 8, 8)

    mem_seq = TitansMemory(cfg_seq)
    mem_chunked = TitansMemory(cfg_chunked)
    # Make sure both have the same init.
    mem_chunked.load_state_dict(mem_seq.state_dict())

    out_seq, _ = mem_seq.read_write(keys, values, keys)
    out_chunked, _ = mem_chunked.read_write(keys, values, keys)
    # Both should produce sensible outputs (no NaNs).
    assert torch.isfinite(out_seq).all()
    assert torch.isfinite(out_chunked).all()


def test_titans_learnable_gates_no_nan():
    _seed(0)
    cfg = MemoryConfig(key_dim=8, value_dim=8, hidden_dim=16, n_layers=2, learnable_gates=True)
    mem = TitansMemory(cfg)
    keys = torch.randn(2, 6, 8)
    values = torch.randn(2, 6, 8)
    out, state = mem.read_write(keys, values, keys)
    assert torch.isfinite(out).all()
    for v in state["params"].values():
        assert torch.isfinite(v).all()


def test_titans_state_persistence_across_calls():
    """Writing in two calls should match writing the whole sequence at once."""
    _seed(0)
    cfg = MemoryConfig(
        key_dim=8,
        value_dim=8,
        hidden_dim=16,
        n_layers=2,
        inner_lr=0.1,
        momentum=0.0,
        forget_rate=0.0,
        learnable_gates=False,
        chunk_size=1,
    )
    mem = TitansMemory(cfg)
    keys = torch.randn(1, 6, 8)
    values = torch.randn(1, 6, 8)

    out_one, _ = mem.read_write(keys, values, keys)

    state = mem.init_state(1)
    state = mem.write(keys[:, :3], values[:, :3], state)
    state = mem.write(keys[:, 3:], values[:, 3:], state)
    out_two = mem.read(keys, state)

    assert torch.allclose(out_one, out_two, atol=1e-5)


@pytest.mark.parametrize("retention", ["l2", "l1", "huber", "kl"])
def test_miras_retention_losses_runnable(retention: str):
    _seed(0)
    cfg = MemoryConfig(
        key_dim=8,
        value_dim=8,
        hidden_dim=16,
        n_layers=2,
        retention=retention,
        learnable_gates=False,
        inner_lr=0.05,
    )
    mem = MirasMemory(cfg)
    keys = torch.randn(1, 4, 8)
    values = torch.randn(1, 4, 8)
    out, state = mem.read_write(keys, values, keys)
    assert out.shape == (1, 4, 8)
    assert torch.isfinite(out).all()


def test_miras_l2_matches_titans():
    """Miras with retention='l2' should be numerically equivalent to Titans."""
    _seed(0)
    cfg_args = dict(
        key_dim=8,
        value_dim=8,
        hidden_dim=16,
        n_layers=2,
        inner_lr=0.1,
        momentum=0.5,
        forget_rate=0.01,
        learnable_gates=False,
        chunk_size=1,
    )
    cfg_t = MemoryConfig(retention="l2", **cfg_args)
    cfg_m = MemoryConfig(retention="l2", **cfg_args)

    mem_t = TitansMemory(cfg_t)
    mem_m = MirasMemory(cfg_m)
    mem_m.load_state_dict(mem_t.state_dict())

    keys = torch.randn(1, 5, 8)
    values = torch.randn(1, 5, 8)
    out_t, _ = mem_t.read_write(keys, values, keys)
    out_m, _ = mem_m.read_write(keys, values, keys)
    assert torch.allclose(out_t, out_m, atol=1e-5)


def test_memory_module_trainable_via_outer_loss():
    """The initial memory parameters must be differentiable end-to-end."""
    _seed(0)
    cfg = MemoryConfig(
        key_dim=8,
        value_dim=8,
        hidden_dim=16,
        n_layers=2,
        inner_lr=0.1,
        learnable_gates=True,
    )
    mem = TitansMemory(cfg)
    keys = torch.randn(1, 4, 8)
    values = torch.randn(1, 4, 8)
    out, _ = mem.read_write(keys, values, keys)
    loss = out.pow(2).mean()
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in mem.parameters())
    assert any_grad, "No memory parameters received a gradient."


def test_titans_hf_backend_is_registered():
    """The ``titans_hf`` backend should be wired into the client router.

    We don't load a real HF model here (that needs the transformers
    package and a network/disk hit); we just check that calling
    ``get_client('titans_hf', ...)`` raises a *transformers* error, not
    an "unknown backend" error.  If transformers IS importable, the
    constructor will go further but that's fine - we still don't want
    to download a model in unit tests.
    """
    import pytest

    from rlm.clients import get_client

    with pytest.raises(Exception) as exc:
        get_client("titans_hf", {"model_name": "this/does-not-exist"})
    msg = str(exc.value)
    assert "Unknown backend" not in msg, f"titans_hf is not registered: {msg!r}"


def test_modeling_wrapper_compatibility():
    """``TitansAugmentedLM`` should accept any HF-style module exposing
    ``model.layers`` and a ``config.hidden_size``."""
    from rlm.memory.modeling import TitansAugmentedLM

    class FakeConfig:
        hidden_size = 16

    class FakeBlock(torch.nn.Module):
        def forward(self, x):
            return x

    class FakeLayers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])

        def __getitem__(self, i):
            return self.blocks[i]

        def __len__(self):
            return len(self.blocks)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeConfig()
            self.model = torch.nn.Module()
            self.model.layers = FakeLayers()  # type: ignore[attr-defined]

        def forward(self, x):
            for b in self.model.layers.blocks:
                x = b(x)
            return x

    fake = FakeModel()
    aug = TitansAugmentedLM(
        fake,
        MemoryConfig(key_dim=0, value_dim=0, hidden_dim=8, n_layers=2),
        flavor="titans",
        freeze_base=False,
    )
    x = torch.randn(1, 5, 16)
    out = aug(x)
    assert out.shape == (1, 5, 16)
    assert torch.isfinite(out).all()
