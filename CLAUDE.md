# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a self-paced CUDA learning repository consisting of 9 Jupyter notebooks. Each notebook is standalone and covers a specific topic in GPU programming, targeted at an **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)** running **CUDA 12.8 / PyTorch 2.11.0+cu128**.

## Running Notebooks

```bash
jupyter notebook          # open notebook server
jupyter nbconvert --to notebook --execute <notebook>.ipynb  # run headless
```

To run a single cell or test a snippet interactively:
```bash
jupyter console           # REPL with GPU access
```

## Key Environment Details

- GPU: RTX 3050 Laptop, 4GB VRAM, 16 SMs, 2048 CUDA cores
- CUDA: 12.8, PyTorch: 2.11.0+cu128
- Primary libraries: `torch`, `numba.cuda`, `cupy`, `transformers`, `bitsandbytes`

## Notebook Sequence & Topics

| Notebook | Topic |
|----------|-------|
| `01_intro_to_cuda.ipynb` | GPU vs CPU, CUDA hierarchy (grid/block/thread), first PyTorch GPU ops |
| `02_pytorch_tensors_on_gpu.ipynb` | Tensor creation, float16/float32, pinned memory, CPU↔GPU transfers |
| `03_numba_cuda_kernels.ipynb` | Writing kernels with `@cuda.jit`, thread indexing, shared memory, reductions |
| `04_cuda_memory_management.ipynb` | VRAM hierarchy, AMP (`autocast`+`GradScaler`), OOM handling, gradient accumulation |
| `05_cupy_numpy_gpu.ipynb` | GPU-accelerated NumPy via CuPy |
| `06_deep_learning_gpu.ipynb` | Training neural networks on GPU |
| `07_cuda_profiling.ipynb` | Profiling GPU kernels and identifying bottlenecks |
| `08_cuda_parallel_patterns.ipynb` | Parallel reduction, scan, and other patterns |
| `09_llm_inference_gpu.ipynb` | Running LLMs (4-bit quantization via `bitsandbytes`, batch inference) |

## Commits & PRs

Rules are in `.claude/commit_pr.md`. Key points:

- **Commit format:** `<type>(<scope>): <summary>` — e.g. `add(nb05): CuPy FFT benchmark`
- **Types:** `add` / `fix` / `update` / `refactor` / `docs` / `chore`
- **Scopes:** notebook number (`nb01`–`nb09`), or `memory` / `kernels` / `llm` / `docs` / `global`
- Single-notebook fixes → commit directly to `main`; multi-notebook or new notebooks → open a PR
- All cells must run top-to-bottom cleanly before committing or opening a PR
- Adding a notebook requires updating the sequence table in this file

## Architecture Patterns Used

**Memory management** — notebooks consistently use `torch.cuda.memory_allocated()` / `torch.cuda.empty_cache()` and the `mem_report()` helper pattern to track 4GB VRAM budget.

**Timing GPU ops** — always call `torch.cuda.synchronize()` before and after timed sections; `cuda.synchronize()` for Numba kernels.

**4GB VRAM constraints** — recurring techniques: float16 over float32, AMP training, gradient accumulation (simulate large batches), `torch.no_grad()` at inference, OOM-safe recursive batch-size halving.

**Numba kernel conventions** — `@cuda.jit` decorated functions, `cuda.grid(1)` / `cuda.grid(2)` for index computation, always bounds-check with `if i < array.shape[0]`, `cuda.syncthreads()` after shared memory writes.

**LLM quantization** — 7B models via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` with `device_map="auto"`.
