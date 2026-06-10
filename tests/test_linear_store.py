"""Property tests for the analytic delta-rule store (rlm/memory/linear_store.py).

These encode the math the design relies on (design doc §1/Q1 and §5.1):
one-shot writes at lr=1, √(m/d) interference scaling, gradient flow through the
inner loop to gates *and* upstream encoders, exact-zero reads from an empty
store, and the persist-params/zero-momentum A4 contract.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from rlm.memory.linear_store import (  # noqa: E402
    DeltaRuleStore,
    LinearStoreConfig,
    rms_norm,
)


def make_store(d_k=64, d_v=48, **kw) -> DeltaRuleStore:
    cfg = LinearStoreConfig(d_k=d_k, d_v=d_v, **kw)
    torch.manual_seed(0)
    return DeltaRuleStore(cfg)


def test_empty_store_reads_exact_zero():
    store = make_store()
    state = store.init_state(2)
    q = torch.randn(2, 5, store.cfg.d_k)
    assert torch.all(store.read(q, state) == 0)


def test_one_shot_write_recalls_value():
    """θ=1 (the gate init), fresh key ⇒ M k = v exactly (up to forget α₀ ≈ 1e-3
    and the value's RMS normalisation)."""
    store = make_store(chunk_size=1)
    state = store.init_state(1)
    k = torch.randn(1, 1, store.cfg.d_k)
    v = torch.randn(1, 1, store.cfg.d_v)
    new = store.write(k, v, state)
    out = store.read(k, new)
    target = rms_norm(v)
    rel = (out - target).norm() / target.norm()
    assert rel < 0.01, f"one-shot recall rel err {rel:.4f}"


def test_orthogonal_keys_do_not_interfere():
    """m ≤ d_k orthonormal-direction keys: each value recalled near-exactly."""
    d_k, m = 64, 32
    store = make_store(d_k=d_k, d_v=32, chunk_size=1)
    state = store.init_state(1)
    K = torch.linalg.qr(torch.randn(d_k, d_k))[0][:m] * math.sqrt(d_k)  # rms-normed rows
    V = torch.randn(m, 32)
    new = store.write(K.unsqueeze(0), V.unsqueeze(0), state)
    out = store.read(K.unsqueeze(0), new)[0]
    rel = (out - rms_norm(V)).norm(dim=-1) / rms_norm(V).norm(dim=-1)
    assert rel.max() < 0.05, f"orthogonal interference: max rel {rel.max():.3f}"


def test_random_key_interference_scales_like_sqrt_m_over_d():
    """Random keys: recall noise grows ~√(m/d) — the A3 planning heuristic."""
    torch.manual_seed(1)
    d_k = 256
    errs = {}
    for m in (8, 128):
        store = make_store(d_k=d_k, d_v=32, chunk_size=1)
        state = store.init_state(1)
        K = torch.randn(1, m, d_k)
        V = torch.randn(1, m, 32)
        new = store.write(K, V, state)
        out = store.read(K, new)
        errs[m] = float((out - rms_norm(V)).norm() / rms_norm(V).norm())
    # 16× more facts ⇒ ≈4× the noise; allow slack for delta-rule error correction.
    ratio = errs[128] / max(errs[8], 1e-9)
    assert 2.0 < ratio < 8.0, f"interference scaling off: {errs}"
    assert errs[8] < 0.25, f"m≪d should be near-clean: {errs}"


def test_write_mask_blocks_writes():
    store = make_store(chunk_size=2)
    state = store.init_state(1)
    k = torch.randn(1, 4, store.cfg.d_k)
    v = torch.randn(1, 4, store.cfg.d_v)
    new = store.write(k, v, state, write_mask=torch.zeros(1, 4))
    assert torch.all(new.M.abs() < 1e-6)


def test_gradients_reach_gates_and_upstream_encoder():
    """BPTT through write→read reaches the lr gate AND a Linear encoder feeding
    the keys — the property the whole meta-training design rests on."""
    store = make_store(d_k=32, d_v=16, chunk_size=2)
    enc = torch.nn.Linear(24, 32)
    h = torch.randn(2, 6, 24)
    v = torch.randn(2, 6, 16)
    state = store.init_state(2)
    new = store.write(enc(h), v, state)
    out = store.read(enc(h[:, -1:]), new)
    out.pow(2).sum().backward()
    assert store.lr_gate.weight.grad is not None and store.lr_gate.weight.grad.abs().sum() > 0
    assert store.forget_gate.weight.grad is not None
    assert enc.weight.grad is not None and enc.weight.grad.abs().sum() > 0


def test_unselective_writes_erode_like_exp_m_theta_over_d():
    """Delta-rule erosion: m random writes at lr θ retain ≈ exp(−mθ/d_k)·(1−α)^m
    of an earlier association — and a write_mask (selectivity) prevents it.
    This is why the lr gate matters for persistence (design doc §5)."""
    import math

    torch.manual_seed(0)
    d_k, m = 128, 100
    store = make_store(d_k=d_k, d_v=16, chunk_size=1)
    state = store.init_state(1)
    k0 = torch.randn(1, 1, d_k)
    v0 = torch.randn(1, 1, 16)
    state = store.write(k0, v0, state)
    baseline = store.read(k0, state)
    keys, vals = torch.randn(1, m, d_k), torch.randn(1, m, 16)

    eroded = store.write(keys, vals, state.clone())
    retention = float((store.read(k0, eroded) * baseline).sum() / baseline.pow(2).sum())
    alpha0 = float(torch.sigmoid(torch.tensor(-5.0))) * store.cfg.max_forget
    predicted = math.exp(-m * 1.0 / d_k) * (1 - alpha0) ** m
    assert abs(retention - predicted) < 0.25, (
        f"retention {retention:.2f} vs predicted {predicted:.2f}"
    )

    protected = store.write(keys, vals, state.clone(), write_mask=torch.zeros(1, m))
    retention_sel = float((store.read(k0, protected) * baseline).sum() / baseline.pow(2).sum())
    assert retention_sel > 0.95, f"masked writes still eroded: {retention_sel:.2f}"


def test_persistence_roundtrip_zeroes_momentum(tmp_path):
    store = make_store()
    state = store.init_state(1)
    state = store.write(torch.randn(1, 3, store.cfg.d_k), torch.randn(1, 3, store.cfg.d_v), state)
    assert state.S.abs().sum() > 0
    p = tmp_path / "mem.pt"
    store.save_state(state, str(p))
    loaded = store.load_state(str(p))
    assert torch.allclose(loaded.M, state.M.to(torch.float32), atol=1e-6)
    assert torch.all(loaded.S == 0)


def test_detach_and_zeroed_views():
    store = make_store()
    state = store.init_state(1)
    state = store.write(torch.randn(1, 2, store.cfg.d_k), torch.randn(1, 2, store.cfg.d_v), state)
    z = state.zeroed()
    assert torch.all(z.M == 0) and state.M.abs().sum() > 0
    d = state.detach()
    assert not d.M.requires_grad


def test_masked_filler_does_not_erode_stored_fact():
    """Regression: write_mask must gate the WHOLE chunk update, not just lr.

    A fact is written, then masked filler is streamed through with a strong
    forget gate; the fact must be retained because a fully-masked chunk is an
    exact no-op. Before the fix the per-chunk forget decay applied on every
    chunk regardless of the mask, eroding cross-session memory on filler.
    """
    store = make_store(d_k=64, d_v=64, chunk_size=4, max_forget=0.5)
    with torch.no_grad():
        store.forget_gate.bias.fill_(8.0)  # sigmoid(8)~1 -> alpha~=max_forget on any ACTIVE chunk
    state = store.init_state(1)
    k0, v0 = torch.randn(1, 1, 64), torch.randn(1, 1, 64)
    state = store.write(k0, v0, state, write_mask=torch.ones(1, 1))
    target = store.read(k0, state)

    state = store.write(
        torch.randn(1, 40, 64), torch.randn(1, 40, 64), state, write_mask=torch.zeros(1, 40)
    )  # 10 fully-masked chunks -> must be a no-op
    after = store.read(k0, state)
    retention = (after.norm() / target.norm().clamp(min=1e-9)).item()
    assert retention > 0.9, f"masked filler eroded the fact (retention {retention:.3f})"
    assert torch.allclose(after, target, atol=1e-4)


def test_write_stats_surface_gate_means():
    """write() populates an optional stats dict with the write-side gate means."""
    store = make_store(d_k=32, d_v=32, chunk_size=2)
    state = store.init_state(1)
    stats: dict = {}
    store.write(
        torch.randn(1, 6, 32),
        torch.randn(1, 6, 32),
        state,
        write_mask=torch.ones(1, 6),
        stats=stats,
    )
    assert {"lr_gate_mean", "momentum_mean", "forget_mean"} <= set(stats)
    assert 0.0 <= stats["lr_gate_mean"] <= store.cfg.max_lr
