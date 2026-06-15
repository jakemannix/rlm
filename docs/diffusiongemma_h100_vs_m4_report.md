> **Scope:** One-off infrastructure benchmark, unrelated to the active sleep-lesson / parametric-memory research tracks.

# DiffusionGemma local M4 Max vs Modal H100 benchmark

Date: 2026-06-13

## Summary

We compared the local M4 Max / Metal GGUF DiffusionGemma path against a Modal H100 / CUDA path using the same modified C++ OpenAI-compatible DiffusionGemma HTTP server and Hermes Agent integration.

The H100 result was dramatically faster, including for the actual Hermes tool loop. The earlier multi-minute Hermes turns on the Mac were not inherent to DiffusionGemma or Hermes tool use; they were primarily the local M4 Max / Metal GGUF backend under Hermes-sized prompts.

## Local M4 Max / Metal results

Measured earlier with the local server at `http://127.0.0.1:18081/v1`:

| Case | Elapsed |
| --- | ---: |
| Direct short prompt: `Say OK only.` | ~5s |
| Direct cats sentence | ~9.4s |
| Hermes terminal tool loop | 167s / 2m47s |
| Actual terminal command inside loop | ~0.8s |

The terminal command itself was trivial, so the 167s loop was model/backend latency, not shell execution.

## Modal H100 / CUDA results

Completed Modal run:

`https://modal.com/apps/jakemannix/main/ap-9b8D5jRniE4BYaFncFoYWa`

Command:

```bash
cd /Users/jake/src/open_src/rlm
RLM_MODAL_GPU=H100 modal run scripts/diffusiongemma_modal_hermes_bench.py
```

Hardware:

- GPU: NVIDIA H100 80GB HBM3
- Backend: llama.cpp CUDA build from DiffusionGemma PR path plus our modified `diffusion-gemma-http` server
- Model: `diffusiongemma-26B-A4B-it-Q4_K_M`
- Context: `-c 4096`
- GPU offload: 31/31 layers
- CUDA model buffer: 16013.20 MiB
- CPU mapped model buffer: 577.50 MiB
- CUDA compute buffer: 4140.00 MiB
- Canvas: 256
- Entropy-bound steps: 48

Health check returned:

```json
{
  "canvas": 256,
  "model": "diffusiongemma-26B-A4B-it-Q4_K_M",
  "n_vocab": 262144,
  "server": "diffusion-gemma-http",
  "status": "ok"
}
```

Direct H100 timings:

| Case | Elapsed | Prompt tokens | Completion tokens | Result |
| --- | ---: | ---: | ---: | --- |
| Direct OK | 1.198s | 13 | 5 | `OK` |
| Direct cats | 0.449s | 16 | 15 | `Cats are graceful companions who spend many hours napping.` |
| Direct structured tool-call | 0.245s | 74 | 19 | OpenAI-compatible `tool_calls`, `get_weather({"city":"Paris"})` |

Hermes-on-H100 timings:

| Case | Elapsed | Result |
| --- | ---: | --- |
| `hermes chat -Q --yolo -t safe -q 'Say OK only.'` | 6.156s | `OK` |
| `hermes chat --yolo -t terminal -q 'Use the terminal tool to run: printf DG_TOOL_OK. Then answer with only the exact output.'` | 10.972s | `DG_TOOL_OK` |

Hermes reported the terminal tool loop as a 9s session with 4 messages and 2 tool calls. The actual `printf DG_TOOL_OK` tool execution took ~0.1s.

## H100 vs M4 Max speedup

| Case | M4 Max / Metal | H100 / CUDA | Approx speedup |
| --- | ---: | ---: | ---: |
| Direct OK | ~5s | 1.198s | ~4.2x |
| Direct cats | ~9.4s | 0.449s | ~20.9x |
| Hermes terminal tool loop | 167s | 10.972s | ~15.2x |

## Interpretation

The H100 result shows that DiffusionGemma can be practical for Hermes Agent loops when running on a strong NVIDIA/CUDA backend. The same terminal-tool workflow went from 167s locally to 10.972s on H100.

The dominant factors appear to be:

1. long-context / prefill cost under Hermes-sized prompts, and
2. the local M4 Max / Metal GGUF path versus the H100 / CUDA path.

The direct OpenAI-compatible structured tool-call case was only 0.245s on H100, so the C++ HTTP server and tool-call parser are not the bottleneck.

## Setup overhead in the one-off Modal runner

The completed run still used a temporary one-off script that compiled and installed software during the GPU job. These setup timings should not be counted as inference latency, but they show why a proper baked image is needed:

| Setup step | Elapsed |
| --- | ---: |
| `git clone --depth 1 https://github.com/ggml-org/llama.cpp /work/llama.cpp` | 2.744s |
| Fetch DiffusionGemma PR | 2.399s |
| CMake configure | 4.963s |
| CUDA build of `llama-diffusion-gemma-http` | 240.723s |
| HF GGUF download | 74.368s |

Total Modal run elapsed was 1338.98s, including image setup, pip installing Hermes, model/server startup, and benchmark execution.

## Next step

Build a proper Modal setup:

- Docker/Modal image contains all software:
  - CUDA base image
  - build tools
  - DiffusionGemma llama.cpp PR checkout
  - our modified `diffusion-gemma-http` and `diffusion-server` code
  - compiled `llama-diffusion-gemma-http` binary
  - Hermes Agent and benchmark harness
- Modal Volume contains large mutable data:
  - GGUF model file(s)
  - benchmark JSON/log outputs
- GPU function does only:
  - attach model Volume
  - start server
  - wait for `/health`
  - run direct and Hermes benchmarks
  - write results

This will make repeated H100/A100 sweeps cheaper, cleaner, and more reproducible.
