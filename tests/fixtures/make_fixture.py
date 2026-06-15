"""Generate a tiny real-data fixture for CPU regression tests.

Captures last-decoder-layer hidden states from ``google/gemma-3-270m`` (the
smallest Gemma) on a handful of prompts, so the memory unit tests can exercise
the module on REAL, non-gaussian, large-magnitude representations without a GPU
or a model download at test time.

Why this exists: the original Titans/Miras unit tests only fed ``torch.randn``
(~unit-scale) tensors, which hid three bugs that only surface on real models
(meta-training gradient severed by ``detach``; gate dtype clash under mixed
precision; and online-update divergence on the large activations real LMs
produce). This fixture lets Tier-1 CPU tests reproduce the real-scale regime
deterministically.

Requires an HF token (``HF_TOKEN`` env var, or ``HF_API_KEY`` in a local
``.env``) on an account that has accepted the Gemma license. Regenerate with::

    python tests/fixtures/make_fixture.py
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Token: prefer HF_TOKEN, fall back to HF_API_KEY (incl. a local .env).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass
_tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if _tok:
    os.environ["HF_TOKEN"] = _tok

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "google/gemma-3-270m"
PROMPTS = [
    "The capital of France is Paris, a city on the Seine.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "In 1969, Apollo 11 landed the first humans on the Moon.",
    "A binary search halves the search interval each step, giving O(log n) time.",
    "Recursion solves a problem by reducing it to smaller instances of itself.",
    "The Pacific is the largest and deepest of Earth's oceanic divisions.",
]


def _find_layers(model: torch.nn.Module):
    obj = model
    for part in ("model", "layers"):
        obj = getattr(obj, part)
    return obj


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append((out[0] if isinstance(out, tuple) else out).detach())

    handle = _find_layers(model)[-1].register_forward_hook(
        hook
    )  # matches the wrapper's insertion point
    states = []
    with torch.no_grad():
        for prompt in PROMPTS:
            captured.clear()
            ids = tok(prompt, return_tensors="pt").input_ids
            model(input_ids=ids)
            states.append(captured[0][0])  # (seq, hidden)
    handle.remove()

    # Keep float32: gemma-3-270m's last-layer "massive activations" exceed
    # float16's max (~65504), so a float16 cast overflows to inf. The large
    # magnitudes are the whole point of this fixture.
    H = torch.cat(states, dim=0).float()  # (N_tokens, hidden)
    out_path = Path(__file__).parent / "gemma270m_states.pt"
    torch.save(
        {
            "model": MODEL_ID,
            "layer": "model.layers[-1]",
            "hidden_size": H.shape[-1],
            "states": H,
            "abs_max": float(H.float().abs().max()),
        },
        out_path,
    )
    print(f"saved {out_path}  states={tuple(H.shape)}  abs-max={H.float().abs().max():.1f}")


if __name__ == "__main__":
    main()
