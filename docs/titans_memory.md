> **Scope:** part of the *parametric-memory* track (frozen-base + meta-trained adapter + mutable store) — a SEPARATE line from the active sleep-time / lesson-centric behavioral-memory work (see rlm/sleep/README.md, docs/memory_experiment_status.md).

# Titans / Miras Parametric Memory

This page documents the parametric long-term memory module that lives in
`rlm.memory`.  Together with the `TitansHFClient` LM backend
(`backend="titans_hf"`), it lets an RLM run on top of a small
open-weights model (Gemma 3 1B-IT by default) that learns at test time.

## What it implements

| Class | Reference | Notes |
|-------|-----------|-------|
| `TitansMemory` | [Titans, Behrouz et al., 2025](https://arxiv.org/abs/2501.00663) | L2 retention surrogate, momentum surprise, adaptive forgetting. |
| `MirasMemory` | [Miras, Behrouz et al., 2025](https://arxiv.org/abs/2504.13173) | Same recurrence as Titans, but the retention surrogate is pluggable (`l2`, `l1`, `huber`, `kl`). |
| `TitansAugmentedLM` | — | Wraps a HuggingFace causal LM and splices a memory module into the final transformer block via a forward hook. |
| `TitansHFClient` | — | A `BaseLM` subclass that exposes the augmented LM as an RLM backend. |

## The update rule

Each token (or chunk of tokens) is projected into a key `k_t` and value
`v_t` via learnable linear maps.  We then run **one online gradient
step** on the memory MLP's parameters:

```
grad_t = ∇_θ L_retention(M_θ(k_t), v_t)        # "surprise"
S_t    = η_t · S_{t-1} − lr_t · grad_t          # momentum surprise
θ_t    = (1 − α_t) · θ_{t-1} + S_t              # forget + write
```

`L_retention` defaults to ½‖M(k) − v‖²_2 (Titans).  In Miras you can
swap it for L1, Huber or KL.  `lr_t`, `η_t`, `α_t` are scalar gates
either from the config or, when `learnable_gates=True`, produced by
small linear heads on the key.

We use `torch.func.functional_call` + `torch.func.grad` so the *outer
loop* gradient (to the initial memory parameters and gate heads) still
flows correctly even though the *inner loop* re-applies the MLP with
per-batch updated parameters.

## Quick start

```python
import torch
from rlm.memory import MemoryConfig, TitansMemory

cfg = MemoryConfig(key_dim=32, value_dim=32, hidden_dim=64)
mem = TitansMemory(cfg)

# 1. Bring up an empty memory state.
state = mem.init_state(batch_size=1)

# 2. Stream (key, value) pairs through it.
keys   = torch.randn(1, 128, 32)
values = torch.randn(1, 128, 32)
state  = mem.write(keys, values, state)

# 3. Query the memory at some new keys.
queries = torch.randn(1, 16, 32)
out = mem.read(queries, state)
```

## Wrapping a HuggingFace model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from rlm.memory import MemoryConfig, TitansAugmentedLM

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
lm = TitansAugmentedLM(
    base,
    MemoryConfig(key_dim=0, value_dim=0, hidden_dim=256, n_layers=2),
    flavor="titans",   # or "miras"
    insert_layer=-1,   # splice into the last transformer block
    freeze_base=True,  # only train the memory
)
```

`TitansAugmentedLM` registers a `register_forward_hook` on the chosen
layer.  At each forward pass the layer's hidden state is projected into
`(k, v, q)`, the memory is updated with `(k, v)`, and the read at `q` is
added back into the stream.  `reset_memory()` wipes the per-conversation
state when starting a new dialogue.

## Using it as an RLM backend

`rlms[titans]` registers a `titans_hf` backend that you can pass to
`RLM`:

```python
from rlm import RLM
from rlm.memory import MemoryConfig

rlm = RLM(
    backend="titans_hf",
    backend_kwargs={
        "model_name": "google/gemma-3-1b-it",
        "memory_config": MemoryConfig(hidden_dim=512, n_layers=2),
        "flavor": "titans",
        "device": "cuda",
        "torch_dtype": "bfloat16",
    },
    max_iterations=8,
)
print(rlm.completion("Write a haiku about parametric memory.").response)
```

The persistent memory state is kept across `completion()` calls on the
same client by default; call `client.reset_memory()` to wipe it.

## Benchmarks & notebooks

* `scripts/benchmarks/` — four bench scripts (reasoning, tool use, code
  gen, long-context needle-in-haystack) plus a `run_all.py` runner.
* `notebooks/` — three Colab notebooks: a standalone memory demo, a
  Gemma+Titans walkthrough, and an end-to-end benchmark run.

See [`scripts/benchmarks/README.md`](../scripts/benchmarks/README.md) for
the CLI surface.

## Limitations & caveats

* `TitansAugmentedLM` is intentionally minimal.  In particular, it
  splices in **one** memory module at one block — not the multi-scale
  memory described in the Titans paper.  For research integration with a
  specific architecture, hook the memory directly into your attention
  layers.
* The initial memory parameters need to be **meta-trained** for real
  results.  Out of the box, random init plus a single forward pass
  isn't enough to reliably memorise arbitrary associations.
* The chunked update keeps the memory state per-batch and runs the
  update in float32 for numerical stability — this is correct but slow
  for very long sequences.  For production use, switch to a custom CUDA
  kernel or accumulate the momentum surprise across attention layers.

## Citing

If you use this implementation, please cite the Titans / Miras papers
in addition to the RLM paper:

```bibtex
@misc{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  year={2025},
  eprint={2501.00663},
  archivePrefix={arXiv},
}
@misc{behrouz2025miras,
  title={It's All Connected: A Journey Through Test-Time Memorization in
         the Attention Surrogate of Modern Architectures},
  author={Behrouz, Ali and others},
  year={2025},
  eprint={2504.13173},
  archivePrefix={arXiv},
}
```
