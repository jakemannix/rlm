"""End-to-end regression for the Titans HF integration on a real model.

Loads the smallest Gemma (``google/gemma-3-270m``) on CPU and checks that
wrapping it with the Titans memory survives a real forward pass (no NaN from the
model's massive activations) and a few meta-training steps. This is the test
that would have caught the divergence bug the original ``torch.randn`` unit
tests missed.

Skipped unless ``transformers`` is installed *and* an HF token is available
(``HF_TOKEN`` or ``HF_API_KEY``, incl. a local ``.env``) on a Gemma-licensed
account — both true on CI configured with such a token.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if not _TOKEN:
    pytest.skip("no HF token (HF_TOKEN / HF_API_KEY) available", allow_module_level=True)
os.environ["HF_TOKEN"] = _TOKEN

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from rlm.memory import MemoryConfig, TitansAugmentedLM  # noqa: E402

MODEL_ID = "google/gemma-3-270m"


@pytest.fixture(scope="module")
def gemma():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    return tok, model


def test_augmented_forward_is_finite(gemma):
    """Wrapping real Gemma must not NaN: the memory's internal normalization
    bounds the model's O(1e4) activations."""
    tok, model = gemma
    aug = TitansAugmentedLM(
        model, MemoryConfig(hidden_dim=64, chunk_size=8), freeze_base=True
    ).eval()
    try:
        # Long enough to span several memory chunks — the unnormalized update
        # only diverges after ~2 chunks, so a short prompt wouldn't guard the bug.
        text = (
            "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. "
            "It extends from the Arctic in the north to the Southern Ocean, and is bounded "
            "by Asia and Australia in the west and the Americas in the east. Photosynthesis "
            "converts sunlight, water, and carbon dioxide into glucose and oxygen."
        )
        ids = tok(text, return_tensors="pt").input_ids
        out = aug(input_ids=ids, labels=ids)
        assert torch.isfinite(out.loss)
    finally:
        aug.remove_hook()


def test_meta_training_steps_stay_finite(gemma):
    """A few meta-training steps on the frozen-base + memory must stay finite."""
    tok, model = gemma
    aug = TitansAugmentedLM(model, MemoryConfig(hidden_dim=64, chunk_size=8), freeze_base=True)
    aug.train()
    try:
        ids = tok(
            "Recursion solves a problem using smaller instances of itself.", return_tensors="pt"
        ).input_ids
        opt = torch.optim.Adam([p for p in aug.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(3):
            out = aug(input_ids=ids, labels=ids)
            assert torch.isfinite(out.loss)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
    finally:
        aug.remove_hook()
