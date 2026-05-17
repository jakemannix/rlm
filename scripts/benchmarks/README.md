# Titans / Miras Benchmark Scripts

Four lightweight benchmark scripts exercising the parametric memory
module on top of an RLM, plus a runner that executes them all.

| Bench | Capability | Dataset | Grading |
|-------|-----------|---------|---------|
| `bench_reasoning.py` | Multi-step reasoning | GSM8K (subset) | Last numeric answer |
| `bench_tool_use.py` | Agentic tool use | In-house mock APIs | Final-answer string / set / number |
| `bench_codegen.py` | Code generation | HumanEval (subset) | Pass-rate of HumanEval-style `check(...)` |
| `bench_long_context.py` | Long-context recall | Synthetic needle-in-haystack | Substring match |

If the `datasets` package is not available, the scripts fall back to
small built-in problem sets so they always run.

## Quick start

```bash
# Compare API baseline vs Titans-augmented Gemma:
python scripts/benchmarks/run_all.py --backend openai --model gpt-5-nano --num-samples 10
python scripts/benchmarks/run_all.py --backend titans_hf --model google/gemma-3-1b-it --num-samples 10

# Sweep retention surrogates (Miras):
for r in l2 l1 huber kl; do
  python scripts/benchmarks/run_all.py \
    --backend titans_hf --memory-flavor miras --retention $r \
    --model google/gemma-3-1b-it --num-samples 5 \
    --output-dir ./bench_results/$r
done
```

## CLI surface

All four scripts share these flags via `common.add_common_args`:

```
--backend {openai,vllm,portkey,openrouter,vercel,anthropic,gemini,titans_hf}
--model <model_id>
--base-url <url>
--num-samples N
--memory-flavor {titans,miras}    # only used by titans_hf backend
--retention {l2,l1,huber,kl}      # Miras surrogate
--memory-hidden-dim D
--memory-layers L
--max-iterations N
--max-depth N
--output-dir DIR
--seed N
--verbose
```

Results are written as `<bench>-<timestamp>.json` and aggregated by
`run_all.py` into `<output-dir>/summary.json`.

## Adding a new benchmark

1. Drop a new script in `scripts/benchmarks/` that calls
   `add_common_args` and `build_rlm` from `common.py`.
2. Use `BenchmarkResult` for accumulation and call `.save(...)` at the
   end.
3. Add an entry to `BENCHES` in `run_all.py` and a prefix mapping for
   the result-file discovery.
