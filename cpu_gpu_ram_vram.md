# CPU vs GPU & RAM vs VRAM

## CPU vs GPU

### Design Philosophy

| | CPU | GPU |
|---|---|---|
| Cores | 8–16 powerful cores | Thousands of small cores |
| Optimized for | Low-latency sequential tasks | High-throughput parallel tasks |
| Clock speed | 3–5 GHz | 1–2 GHz |
| Cache | Large (L1/L2/L3, MBs) | Smaller per core |
| Branch prediction | Sophisticated | Minimal |
| Context switching | Fast | Slow |

### How They Execute Work

**CPU — latency-optimized:**
```
Task: [A] → [B] → [C] → [D]
       ↓     ↓     ↓     ↓
Each step completes before the next begins.
Few cores, each very fast. Great for logic, I/O, branching.
```

**GPU — throughput-optimized:**
```
Task: process 1 million array elements
       ↓
Thread 0: A[0]
Thread 1: A[1]    ← all running simultaneously
Thread 2: A[2]
...
Thread N: A[N]
```

### When to Use Each

| Use CPU | Use GPU |
|---------|---------|
| Complex control flow / branching | Matrix multiplication |
| Low-latency single operations | Neural network training/inference |
| I/O-bound tasks | Image/video processing |
| Small data (< thousands of elements) | Large parallel data transformations |
| OS, database, web server logic | Scientific simulations |

### CUDA Thread Hierarchy

```
Grid  (one kernel launch — the whole job)
 └── Blocks  (groups of threads, share fast shared memory ~48KB)
      └── Threads  (individual workers — each runs the kernel function)
```

- RTX 3050 has **16 SMs** (Streaming Multiprocessors), each running multiple blocks
- Typical block size: **256 threads** (must be a multiple of 32 — the warp size)
- Warp = 32 threads that execute in **lockstep** (SIMT model)

---

## RAM vs VRAM

### Physical Differences

| | RAM (System Memory) | VRAM (GPU Memory) |
|---|---|---|
| Full name | Random Access Memory | Video RAM |
| Location | Motherboard, near CPU | On the GPU die |
| Typical size | 16–64 GB | 4–24 GB |
| Bandwidth | ~50–100 GB/s | ~200–900 GB/s |
| Latency | ~100 ns | ~200–500 ns |
| Controlled by | OS + CPU | GPU driver + CUDA |
| Example (this machine) | System RAM | RTX 3050 — 4 GB GDDR6 |

### Why VRAM Bandwidth Is So Much Higher

RAM is designed for flexible, varied access patterns by a few cores.  
VRAM is designed for one thing: feeding thousands of GPU cores simultaneously with data.

```
RAM:   [CPU 0] [CPU 1] ... [CPU 15]  →  ~8 lanes of traffic
VRAM:  [GPU core 0..2047]            →  hundreds of lanes of traffic
```

GDDR6 (RTX 3050) achieves ~192 GB/s vs DDR4's ~50 GB/s.

### Memory Transfer Bottleneck

CPU and GPU have **separate memory spaces**. Transferring data between them goes over PCIe:

```
[CPU] ←—— PCIe (16 GB/s max) ——→ [GPU]
 RAM                               VRAM
```

PCIe bandwidth (~16 GB/s) is ~12× slower than VRAM bandwidth. This is why minimizing CPU↔GPU transfers is critical.

**Best practices:**
- Create tensors directly on GPU: `torch.randn(N, device='cuda')` not `.to('cuda')` after
- Use **pinned (page-locked) memory** for faster transfers: `tensor.pin_memory()`
- Use `non_blocking=True` to overlap transfer with compute
- Keep data on GPU for the entire pipeline; only pull results back at the end

### Memory Hierarchy on RTX 3050

```
┌──────────────────────────────────────────────┐
│  VRAM (Global Memory) — 4 GB GDDR6           │  ~192 GB/s, high latency
│  ├── L2 Cache — ~1 MB                        │  automatic, ~2× faster
│  └── Per SM (×16 SMs):                       │
│       ├── L1 Cache + Shared Memory — 48 KB   │  ~10× faster, programmer-controlled
│       └── Registers — ~256 KB                │  ultra-fast, per-thread
└──────────────────────────────────────────────┘
```

Access speed (fastest → slowest): **Registers > Shared Memory > L1 > L2 > VRAM > RAM**

### VRAM Sizing Guide for 4 GB

| Data type | 1M parameters | 1B parameters |
|-----------|--------------|---------------|
| float32   | 4 MB         | 4 GB          |
| float16   | 2 MB         | 2 GB          |
| int8      | 1 MB         | 1 GB          |
| int4      | 0.5 MB       | 0.5 GB        |

**Training multiplier:** weights + gradients + optimizer state ≈ **4–6× model size** in VRAM.  
**Inference:** weights only (+ activations per batch).

### Techniques to Fit More in 4 GB VRAM

| Technique | VRAM Reduction | Notes |
|-----------|---------------|-------|
| `float16` over `float32` | 50% | Use Tensor Cores, slight precision loss |
| AMP (`autocast` + `GradScaler`) | ~40–50% | Best of both: fp16 forward, fp32 weights |
| 8-bit quantization (`bitsandbytes`) | 50% | Minimal quality loss for inference |
| 4-bit quantization (NF4) | 75% | Fits 7B LLMs; 10–20% quality trade-off |
| `torch.no_grad()` at inference | Saves activations | Always use during eval |
| Gradient accumulation | Allows smaller batches | Simulates large batch without VRAM cost |
| Gradient checkpointing | ~60–70% activation memory | Recomputes activations on backward pass |

---

## Quick Reference

```python
import torch

# Check what you're working with
torch.cuda.get_device_name(0)               # GPU name
torch.cuda.get_device_properties(0).total_memory / 1e9  # VRAM in GB
torch.cuda.memory_allocated() / 1e6        # currently used VRAM (MB)
torch.cuda.memory_reserved() / 1e6         # held by PyTorch allocator (MB)
torch.cuda.empty_cache()                   # release unused reserved memory

# Move data
tensor.to('cuda')                          # CPU RAM → GPU VRAM
tensor.cpu()                               # GPU VRAM → CPU RAM
tensor.cpu().numpy()                       # GPU → NumPy (must go via CPU)

# Avoid unnecessary transfers
x = torch.randn(N, device='cuda')          # create directly on GPU
with torch.no_grad():                      # skip gradient storage
    out = model(x)
```
