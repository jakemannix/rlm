"""
Generate the three Colab notebooks that accompany the Titans / Miras
memory module.

Run with ``python notebooks/build_notebooks.py`` to regenerate the .ipynb
files from the cell definitions below.  Keeping the source-of-truth as
Python lets us regenerate notebooks reproducibly and lint them like
normal code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _cell_id(text: str) -> str:
    return "cell-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(text),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(text),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Notebook 1: Standalone Titans / Miras memory demo
# ---------------------------------------------------------------------------

NB1 = make_notebook(
    [
        md(
            """# Titans / Miras Parametric Memory — Standalone Demo

This notebook walks through the **neural long-term memory** module that
lives in `rlm.memory`, with no large LM in the loop.  We show:

1. Building a `TitansMemory` module.
2. Streaming `(key, value)` pairs through it and reading the memory.
3. Switching to a `MirasMemory` with different retention losses
   (`l2`, `l1`, `huber`, `kl`).
4. A synthetic associative-recall task that exercises the memory
   the way the long-context "needle in haystack" benchmarks do.

The whole notebook runs in <30s on CPU.
"""
        ),
        code(
            """# === Install ============================================================
# Colab usually has torch.  We install rlms from this repo (editable) so the
# notebook uses the same memory module that the rest of the library uses.
%pip install --quiet torch
import os
if not os.path.exists("/content/rlm"):
    !git clone --depth 1 https://github.com/alexzhang13/rlm /content/rlm 2>/dev/null || true
%pip install --quiet -e /content/rlm 2>/dev/null || %pip install --quiet -e .
"""
        ),
        code(
            """import torch
from rlm.memory import MemoryConfig, TitansMemory, MirasMemory

torch.manual_seed(0)
print("torch:", torch.__version__)
"""
        ),
        md(
            """## 1. Build a Titans memory module

`MemoryConfig` controls the geometry and update rule.  Here we use a
small MLP (16 → 32 → 16) and disable the learnable gates so the inner
update is plain SGD."""
        ),
        code(
            """cfg = MemoryConfig(
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
print(mem)
print("parameter count:", sum(p.numel() for p in mem.parameters()))
"""
        ),
        md(
            """## 2. Associative recall

We generate 8 random `(k, v)` pairs and stream them through the memory.
After enough rehearsal passes the memory MLP "stores" the associations:
querying with `k` recovers `v`."""
        ),
        code(
            """keys = torch.randn(1, 8, 16)
values = torch.randn(1, 8, 16)

state = mem.init_state(1)
err_before = (mem.read(keys, state) - values).pow(2).mean().item()
print(f"MSE before any writes: {err_before:.3f}")

errors = []
for epoch in range(6):
    state = mem.write(keys, values, state)
    err = (mem.read(keys, state) - values).pow(2).mean().item()
    errors.append(err)
    print(f"  epoch {epoch+1:>2}  MSE = {err:.3f}")
"""
        ),
        code(
            """# Quick matplotlib plot of the convergence curve.
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.plot([err_before] + errors, marker="o")
plt.xlabel("rehearsal pass")
plt.ylabel("reconstruction MSE")
plt.title("Titans memory learns the (k,v) associations online")
plt.grid(alpha=0.3)
plt.show()
"""
        ),
        md(
            """## 3. Miras: swap the retention surrogate

`MirasMemory` is a drop-in subclass that exposes the retention loss as
a config knob.  We can compare `l2` (= Titans), `l1`, `huber`, `kl`."""
        ),
        code(
            """def run(retention: str, epochs: int = 6):
    torch.manual_seed(0)
    cfg = MemoryConfig(
        key_dim=16, value_dim=16, hidden_dim=32, n_layers=2,
        inner_lr=2.0, momentum=0.0, forget_rate=0.0,
        learnable_gates=False, chunk_size=1, retention=retention,
    )
    m = MirasMemory(cfg)
    keys = torch.randn(1, 8, 16)
    values = torch.randn(1, 8, 16)
    state = m.init_state(1)
    out = [(m.read(keys, state) - values).pow(2).mean().item()]
    for _ in range(epochs):
        state = m.write(keys, values, state)
        out.append((m.read(keys, state) - values).pow(2).mean().item())
    return out

curves = {r: run(r) for r in ["l2", "l1", "huber", "kl"]}

plt.figure(figsize=(5, 3.5))
for name, c in curves.items():
    plt.plot(c, marker="o", label=name)
plt.legend()
plt.xlabel("rehearsal pass")
plt.ylabel("reconstruction MSE")
plt.title("Miras retention surrogates")
plt.grid(alpha=0.3)
plt.show()
"""
        ),
        md(
            """## 4. Learnable gates + momentum

The realistic Titans configuration uses data-dependent `(lr, momentum,
forget)` gates and momentum surprise.  The gates are gradients-trained
end-to-end with the rest of the model in a meta-learning loop; here we
just check that the forward pass runs cleanly with them on."""
        ),
        code(
            """cfg_full = MemoryConfig(
    key_dim=16, value_dim=16, hidden_dim=32, n_layers=2,
    inner_lr=1.0, momentum=0.9, forget_rate=0.05,
    learnable_gates=True, chunk_size=2,
)
m_full = TitansMemory(cfg_full)
keys = torch.randn(1, 16, 16)
values = torch.randn(1, 16, 16)
out, _ = m_full.read_write(keys, values, keys)
print("output:", out.shape, "finite?", torch.isfinite(out).all().item())

# Outer-loop differentiability check: gradient flows to the initial memory
# parameters even though the memory was updated *during* the forward pass.
loss = out.pow(2).mean()
loss.backward()
any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m_full.parameters())
print("memory params receive outer-loop gradient:", any_grad)
"""
        ),
        md(
            """### What to do next

Continue with the next notebook, `titans_gemma_demo.ipynb`, where we
wrap the same memory module around a tiny **Gemma 3 1B-IT** to build a
memory-augmented LM.  Then `titans_rlm_benchmark.ipynb` runs the four
benchmark scripts (reasoning, tool use, code gen, long context) and
plots the results."""
        ),
    ]
)


# ---------------------------------------------------------------------------
# Notebook 2: Gemma + Titans
# ---------------------------------------------------------------------------

NB2 = make_notebook(
    [
        md(
            """# Wrapping Gemma with Titans Memory

This notebook installs `rlms[titans]`, loads a small **Gemma 3 1B-IT**
model, and splices a Titans memory module into the final transformer
block via `TitansAugmentedLM`.  We then run a short generation and a
synthetic "memorise then recall" test."""
        ),
        code(
            """%pip install --quiet "torch>=2.3.0" "transformers>=4.45.0" "accelerate>=0.30.0"
import os
if not os.path.exists("/content/rlm"):
    !git clone --depth 1 https://github.com/alexzhang13/rlm /content/rlm 2>/dev/null || true
%pip install --quiet -e /content/rlm 2>/dev/null || %pip install --quiet -e .
"""
        ),
        code(
            """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rlm.memory import MemoryConfig, TitansAugmentedLM

# Gemma 3 1B-IT is a tiny chat-tuned Gemma that fits on a free Colab GPU.
MODEL = "google/gemma-3-1b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)
print("base loaded;", sum(p.numel() for p in base.parameters()) / 1e6, "M params")
"""
        ),
        md(
            """## Build the memory-augmented LM

`TitansAugmentedLM` registers a forward hook on the last transformer
block.  At each forward pass it projects the block's hidden states into
the memory's `(key, value, query)` space, updates the memory with the
keys/values, reads at the queries, and adds the result back into the
hidden stream."""
        ),
        code(
            """cfg = MemoryConfig(
    key_dim=0,        # 0 = inherit the model hidden size
    value_dim=0,
    hidden_dim=512,
    n_layers=2,
    inner_lr=0.5,
    momentum=0.9,
    forget_rate=0.01,
    learnable_gates=True,
    chunk_size=8,
)
lm = TitansAugmentedLM(base, cfg, flavor="titans", insert_layer=-1).to(device)
print("memory params:", sum(p.numel() for p in lm.memory.parameters()) / 1e6, "M")
"""
        ),
        md(
            """## Smoke-test generation

A vanilla generation request with the memory in the loop."""
        ),
        code(
            """prompt = "Q: What does the Titans memory module store?\\nA:"
inputs = tok(prompt, return_tensors="pt").to(device)
lm.reset_memory(batch_size=1, device=device, dtype=dtype)
with torch.no_grad():
    out = lm.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
"""
        ),
        md(
            """## Memorise-and-recall

Stream a long passage through the model so the memory module learns
the hidden association, then ask the model to recall it.  This is the
showcase for the parametric memory: the answer is **not** in the
context window of the final question, but it *is* in the memory's
learned weights."""
        ),
        code(
            """passage = (
    "Please remember the following important fact for later: "
    "The vault passcode for project Titans is GRIFFIN-2049. "
) * 8  # repeat so the memory has many rehearsal steps

# Stream the passage so the memory absorbs it.
lm.reset_memory(batch_size=1, device=device, dtype=dtype)
with torch.no_grad():
    enc = tok(passage, return_tensors="pt").to(device)
    _ = lm(**enc)

# Now query.  We *don't* include the passage in the query - only the memory
# is carrying the information.
query = "Question: What is the vault passcode for project Titans?\\nAnswer:"
enc_q = tok(query, return_tensors="pt").to(device)
with torch.no_grad():
    out = lm.generate(**enc_q, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][enc_q.input_ids.shape[1]:], skip_special_tokens=True))
"""
        ),
        md(
            """### Caveats

The Titans memory module is initialised randomly here, so a single
rehearsal isn't enough for it to reliably store arbitrary associations.
For real downstream use, you'd:

1. **Meta-train** the memory: keep the base LM frozen and train the
   memory module on a corpus of (long-context, query) pairs so the
   initial gates and projection layers learn good inductive biases.
2. **Mix memory-as-context with attention**: keep a small amount of the
   recent context in attention, and offload the long tail to memory.

See the next notebook (`titans_rlm_benchmark.ipynb`) for an end-to-end
benchmark of the system as an RLM backend."""
        ),
    ]
)


# ---------------------------------------------------------------------------
# Notebook 3: RLM benchmark
# ---------------------------------------------------------------------------

NB3 = make_notebook(
    [
        md(
            """# Benchmarking Titans-augmented Gemma inside RLM

This notebook runs the four bench scripts shipped in
`scripts/benchmarks/`:

| Bench | Capability tested |
|-------|-------------------|
| `bench_reasoning.py`    | multi-step math reasoning (GSM8K-style)   |
| `bench_tool_use.py`     | agentic tool calling                       |
| `bench_codegen.py`      | code generation (HumanEval-style)          |
| `bench_long_context.py` | long-context needle-in-haystack            |

We compare two backends:

1. The vanilla API baseline (e.g. `gpt-5-nano`).
2. Local Gemma 3 1B-IT augmented with a Titans memory module.

A Colab T4 is enough; expect ~10 minutes for `--num-samples 5`."""
        ),
        code(
            """%pip install --quiet "rlms[titans]" || %pip install --quiet -e /content/rlm
import os
if not os.path.exists("/content/rlm"):
    !git clone --depth 1 https://github.com/alexzhang13/rlm /content/rlm 2>/dev/null || true
"""
        ),
        code(
            """# Optional - set this if you also want to run the OpenAI baseline:
# os.environ["OPENAI_API_KEY"] = "sk-..."
"""
        ),
        md("""## Run all four benchmarks against Gemma + Titans"""),
        code(
            """!cd /content/rlm && python -m scripts.benchmarks.run_all \\
    --backend titans_hf --model google/gemma-3-1b-it \\
    --num-samples 3 --max-iterations 6 --output-dir bench_results/titans
"""
        ),
        md("""## Run the same benches against an API baseline (optional)"""),
        code(
            """# !cd /content/rlm && python -m scripts.benchmarks.run_all \\
#     --backend openai --model gpt-5-nano \\
#     --num-samples 3 --max-iterations 6 --output-dir bench_results/openai
"""
        ),
        md(
            """## Sweep retention surrogates (Miras)

`MirasMemory` exposes the inner-loop retention loss as a config knob.
Each variant captures a different attention/memory style:

* `l2` — vanilla Titans
* `l1` — sign-SGD style updates (mamba-like)
* `huber` — smooth interpolation
* `kl` — softmax / Bregman retention (closer to attention)
"""
        ),
        code(
            """!for r in l2 l1 huber kl; do \\
    cd /content/rlm && python -m scripts.benchmarks.run_all \\
        --backend titans_hf --memory-flavor miras --retention $r \\
        --model google/gemma-3-1b-it \\
        --num-samples 2 --max-iterations 4 \\
        --skip codegen --output-dir bench_results/miras-$r ;\\
done
"""
        ),
        md("""## Aggregate & plot"""),
        code(
            """import json, glob
import pandas as pd
rows = []
for d in glob.glob("/content/rlm/bench_results/**/summary.json", recursive=True):
    with open(d) as f:
        s = json.load(f)
    for bench, val in s.get("benches", {}).items():
        if "error" in val: continue
        rows.append(
            {
                "config": d.split("/")[-2],
                "bench": bench,
                "accuracy": val["accuracy"],
                "n_examples": val["n_examples"],
                "n_errors": val["n_errors"],
                "time_s": val["total_time_seconds"],
            }
        )
df = pd.DataFrame(rows)
df.pivot_table(index="config", columns="bench", values="accuracy")
"""
        ),
        md(
            """### Reading the table

You should see Titans-augmented Gemma score competitively with the API
baseline on long-context needle-in-haystack (the memory's strong suit)
while being below the baseline on reasoning / codegen (where Gemma 1B
alone isn't strong).  Sweeping the Miras retention surrogate gives you
a sense of which inner-loop optimisation problem fits each task best.

For longer experiments, raise `--num-samples` and run on a bigger
backbone (`google/gemma-2-9b-it`)."""
        ),
    ]
)


def main() -> None:
    out_dir = Path(__file__).parent
    for name, nb in [
        ("titans_memory_demo.ipynb", NB1),
        ("titans_gemma_demo.ipynb", NB2),
        ("titans_rlm_benchmark.ipynb", NB3),
    ]:
        path = out_dir / name
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
