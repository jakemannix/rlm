"""Generate the sleep-consolidation Colab notebook.

Run with ``python notebooks/build_sleep_notebook.py`` to regenerate
``sleep_consolidation_sweep.ipynb`` from the cell definitions below,
mirroring ``build_notebooks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_notebooks import code, make_notebook, md

NB = make_notebook(
    [
        md(
            """# Sleep-Time Consolidation — Nightly LoRA from Agentic Traces

Proof-of-concept of the loop motivated in `docs/learning_signal.md`:

1. **Gate** — cheaply score a "day" of agent episodes for *learnworthiness*
   (errors, retries, failures, unusual length).
2. **Judge** — spend high test-time compute (LLM-as-a-Judge, self-consistency
   + a verification pass) reflecting on the gated episodes, distilling
   verified SFT examples.
3. **Train** — LoRA the night's examples (+ general replay) into an adapter.
4. **Eval** — next-day NLL (forward transfer), held-out test NLL
   (generalization), retention NLL (forgetting guard).
5. **Sweep** — grid over gate budget / LoRA rank / learning rate.

Runtime: T4/L4 GPU. Policy model defaults to `Qwen/Qwen2.5-1.5B-Instruct`
(any 1–4B instruct model fits Colab; 7–8B works on A100).
"""
        ),
        code(
            """# 1. Install (repo + the `sleep` extra: torch/transformers/peft/datasets)
!pip install -q "rlms[sleep] @ git+https://github.com/jakemannix/rlm.git@claude/clever-hopper-b6orfo"
"""
        ),
        code(
            """# 2. Load agentic traces from HuggingFace and make deterministic splits.
#    AgentInstruct domains: os, db, alfworld, webshop, kg, mind2web.
from rlm.sleep import partition_days, split_episodes
from rlm.sleep.traces import load_hf_traces

episodes = load_hf_traces("THUDM/AgentInstruct", "os")
train, val, test = split_episodes(episodes)
days = partition_days(train, episodes_per_day=64)
print(f"{len(episodes)} episodes -> {len(train)} train / {len(val)} val / {len(test)} test")
print(f"{len(days)} simulated days of {64} episodes")
print("\\nSample episode:\\n" + days[0][0].transcript(max_chars=600))
"""
        ),
        code(
            """# 3. Configure. The judge needs an API key (or use the mock to smoke-test).
import os
from getpass import getpass

from rlm.clients.openai import OpenAIClient
from rlm.sleep import SleepConfig

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key (for the judge): ")

judge_lm = OpenAIClient(model_name="gpt-4o")
config = SleepConfig(
    policy_model="Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
    episodes_per_day=64,
)
config.judge.n_samples = 3   # self-consistency: more samples = more judge TTC
config.gate.budget_fraction = 0.25
print(config)
"""
        ),
        code(
            """# 4. One nightly cycle: gate -> judge -> LoRA -> eval (~10-20 min on L4).
from rlm.sleep.loop import run_night

result = run_night(
    day_index=0,
    day_episodes=days[0],
    next_day_episodes=days[1][:24],   # cap eval size for notebook speed
    test_episodes=test[:24],
    config=config,
    judge_lm=judge_lm,
    out_dir="runs/sleep/colab/day_000",
)
print(f"gated {result.n_selected}/{result.n_episodes}, "
      f"learn verdicts {result.n_learn_verdicts}, examples {result.n_examples}")
base, adapted = result.base_report, result.adapted_report
print(f"next-day NLL {base.next_day_nll:.4f} -> {adapted.next_day_nll:.4f}")
print(f"test NLL     {base.test_nll:.4f} -> {adapted.test_nll:.4f}")
print(f"retention    {base.retention_nll:.4f} -> {adapted.retention_nll:.4f}  (watch this!)")
"""
        ),
        md(
            """**Reading the numbers** — `next_day` and `test` NLL going **down** means
tonight's lessons transferred; `retention` going **up** means the adapter is
eroding general ability (raise `adapter.replay_ratio`, lower `adapter.lr`,
or tighten the judge's verification). The verification gate is load-bearing:
only judge-verified examples earned a gradient.
"""
        ),
        code(
            """# 5. Inspect what the judge decided to learn.
import json

outputs = json.loads(open("runs/sleep/colab/day_000/judge_outputs.json").read())
for o in outputs[:5]:
    print(f"[{o['verdict']} conf={o['confidence']:.2f}] {o['lesson']}")
    for ex in o["examples"]:
        print(f"   Q: {ex['prompt'][:90]}")
        print(f"   A: {ex['response'][:90]}")
"""
        ),
        code(
            """# 6. Hyperparameter sweep (sequential; size the grid to your GPU budget).
from rlm.sleep.sweep import run_sweep

csv_path = run_sweep(
    grid={
        "adapter.lr": [1e-4, 2e-4, 5e-4],
        "adapter.rank": [8, 16],
        "gate.budget_fraction": [0.15, 0.3],
    },
    base_config=config,
    day_episodes=days[0],
    next_day_episodes=days[1][:24],
    test_episodes=test[:24],
    judge_lm=judge_lm,
    out_dir="runs/sleep/colab/sweep",
)
"""
        ),
        code(
            """# 7. Plot the sweep: forward transfer vs forgetting.
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(csv_path)
fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(df["retention_delta"], df["next_day_delta"],
                     c=df["adapter.lr"], s=60, cmap="viridis")
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("retention delta (forgetting; lower-left is better)")
ax.set_ylabel("next-day delta (transfer)")
ax.set_title("Sleep-consolidation sweep: transfer vs forgetting")
fig.colorbar(scatter, label="adapter.lr")
plt.show()
df.sort_values("next_day_delta").head(10)
"""
        ),
        md(
            """## Next steps

* **More days**: loop `run_night` over `days[i]`, re-evaluating each adapter
  on the *fixed* test split to plot learning curves over a week.
* **Stronger judge**: bump `config.judge.n_samples`, or point `judge_lm` at a
  reasoning model — the whole premise is that judge TTC is cheap at 3am.
* **Real surprise gate**: pass a `SurpriseGate` (policy-model NLL) to
  `select_for_reflection` and sweep `gate.use_surprise`.
* **Other trace sets**: any HF dataset with `conversations` or `messages`
  fields normalizes via `rlm.sleep.traces.normalize_record`.
"""
        ),
    ]
)


if __name__ == "__main__":
    out = Path(__file__).parent / "sleep_consolidation_sweep.ipynb"
    out.write_text(json.dumps(NB, indent=1) + "\n")
    print(f"wrote {out}")
