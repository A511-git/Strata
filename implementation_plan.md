# Architecture Analysis & Implementation Plan: 2x T4 Low-RAM Dual-GPU Resident Partitioning

## Goal Description
The objective is to analyze, validate, and execute an optimal multi-GPU memory layout for running **Qwen 3.8 Flash** (~48 GB quantized, e.g. IQ3_S / IQ3_XXS / Q4_K_XL) on a **Kaggle dual NVIDIA T4 system** (2× 15-16 GB VRAM = ~30 GB VRAM, ~30-31 GB host RAM, 2 vCPUs, PCIe Gen3 x16 ~11.5 GB/s).

The core architectural problem: A 48 GB model cannot fit entirely into 30 GB of host RAM if loaded conventionally, nor can it fit into 30 GB of total VRAM alone. The user proposes partitioning the model non-redundantly across VRAM (~22 GB across GPU 0 + GPU 1) and host RAM (~26 GB pinned), creating a closed-loop bidirectional cyclic swap (evicted VRAM experts return to RAM, while needed RAM experts move to VRAM) without requiring a full duplicate copy in RAM.

This document reviews:
1. **Whether this approach has been proposed, developed, or merged in Strata PRs/features.**
2. **Detailed comparison** between the user's proposal, upstream Strata's implementation (`PR #848`, `PR #864`, `PR #876`), and `eddoursul`'s custom branch (`PR #245`).
3. **PCIe and compute bottleneck analysis** on 2× T4 hardware (per-token miss swap vs. batched adaptive swap vs. CPU fallback).
4. **Concrete execution & verification plan** for running this on the Kaggle 2× T4 box.

---

## Part 1: Has Anyone Done This in a PR or Feature Request?

> [!IMPORTANT]
> **Yes! Not only has this been proposed and developed, but core elements of this exact memory partitioning and bidirectional swapping are already implemented in Strata through a series of key PRs.**

Here is the exact lineage of this architectural concept across the codebase and commit history:

### 1. PR #848 (Engine v0.1.40): Split + Resident Complement Mode
* **The Problem It Solved:** Before PR #848, `--resident-experts` (the low-RAM mode) could not be combined with `--layer-split`. When multi-GPU was detected, setup downgraded `--resident-experts` to `--mmap-experts`, which relied on the OS file cache and caused severe disk thrashing when total model size exceeded physical RAM.
* **What PR #848 Implemented (`pin_cache_complement`):**
  - Strata inspects the experts assigned to **all GPUs** (GPU 0 cache and GPU 1 stage cache).
  - Instead of allocating RAM for the entire 48 GB model, `src/core/expert_source.cpp::pin_cache_complement()` calculates the **cache complement** — strictly the subset of experts that *neither GPU holds*.
  - For a 48 GB model with ~22 GB resident across GPU 0 and GPU 1, it allocates and pins **only the remaining ~26 GB in host RAM**.
  - **Zero weight duplication:** Every expert exists in exactly one tier (GPU 0, GPU 1, or host RAM).

### 2. PR #864 & Commit `1e6cec8` / `e57c907`: Resident Expert Exchange Buffer Rotation (`STRATA_EXCHANGE_ROTATE`)
* **The Problem It Solved:** When an expert is evicted from GPU VRAM and replaced by an expert promoted from host RAM, the evicted expert must be written back to host RAM so it isn't lost. Previously, this required a GPU-to-host PCIe copy into a staging buffer followed by a synchronous `std::memcpy` into the host RAM slot vacated by the promoted expert.
* **What PR #864 Implemented (`exchange_storage.hpp` / `docs/EXCHANGE_ROTATION.md`):**
  - Implements **zero-copy host buffer rotation**: The temporary exchange buffer that received the evicted GPU expert becomes the new resident slot, and the old RAM slot becomes the next temporary buffer.
  - Eliminates host memory copy overhead entirely (`no host commit memcpy`).
  - Activated via `STRATA_EXCHANGE_ROTATE=1`.

### 3. PR #876: Asynchronous Adaptive Swapping (`--adapt-async 1`)
* **What It Implemented:** Asynchronously pipelines the bidirectional expert exchange between GPU caches and the host resident complement during generation, preventing the engine from freezing token generation while swapping weights.

### 4. `eddoursul`'s Custom Branch (PR #245) & `docs/SECOND_GPU.md`
* **What Eddoursul Implemented:**
  - Designed the **expert-tier helper** architecture: GPU 0 holds dense weights, KV cache, and MTP; GPU 1 (`--expert-cache-device1 auto` or `--peer-device 1`) acts as a pure expert cache.
  - While this maximizes VRAM on GPU 1 for experts, in early versions it required the entire expert arena in host RAM and lacked the multi-GPU complement partitioning introduced later in PR #848.

---

## Part 2: Deep Architectural Comparison

| Dimension | User's Proposed Cyclic Architecture | Strata Layer-Split Resident (`PR #848` + `#864`) | Eddoursul Helper Tier (`PR #245` / `--peer-device`) | Tensor Parallelism (TP) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Partitioning** | ~11 GB GPU 0, ~11 GB GPU 1, ~26 GB pinned in RAM | Layers 0–$K$ on GPU 0, $K$–47 on GPU 1; complement (~26 GB) in RAM | GPU 0: dense + KV + MTP + experts; GPU 1: pure expert cache | All weights sliced evenly across both GPUs |
| **Weight Duplication** | None (Union of VRAM + RAM = 48 GB) | None (Complement only in RAM; dense weights per-stage or trimmed) | Dense weights on GPU 0; experts unique | Slices on each GPU (no full-weight duplication) |
| **RAM Footprint** | ~26 GB (Fits in 30 GB RAM) | ~26 GB (Fits in 30 GB RAM via `--resident-experts`) | Can exceed 30 GB unless budget-clamped | Sliced weights loaded via mmap or full RAM |
| **Miss Handling** | Swap weights dynamically on every miss (evict + write back) | Decode misses computed by CPU pool or served from pinned RAM; periodic adaptive swap | Misses computed by CPU pool or secondary helper; periodic swap | Whole expert slices must be loaded from RAM if missed |
| **PCIe Traffic** | **Massive per token** (~1.4 GB to 2.8 GB per token roundtrip) | **Minimal** (1 activation transfer per window ~few hundred KB) | Pinned activations/results cross PCIe per layer (~few MB) | **Severe per layer** (all-reduce twice per layer = 96 syncs/token) |
| **Compatibility on 2x T4** | Theoretical | **Production ready** in Strata v0.1.40+ | Experimental (CUDA 12/13 dependencies) | Very slow on PCIe Gen3 without NVLink |

---

## Part 3: The PCIe & Latency Reality Check on 2× T4

> [!WARNING]
> ### Why Per-Token Dynamic Swapping Stalls on PCIe Gen3
> In the user's initial thought, when an expert misses during token generation, the GPU fetches it from RAM and writes an evicted expert back to RAM *on that token*. Let's evaluate the physical throughput:

1. **Model Math (Qwen 3.8 Flash, IQ3_S):**
   - 48 layers, top-8 routing per layer = 384 expert evaluations per token.
   - Each expert is ~12.5 MiB.
   - If 70% of experts are in VRAM, **30% miss in VRAM**.
   - $384 \times 0.30 \approx 115\text{ expert misses per token}$.
2. **PCIe Bandwidth Bottleneck:**
   - 115 experts $\times 12.5\text{ MiB} \approx \mathbf{1.44\text{ GiB}}$ of weights to upload to GPU per token.
   - Bidirectional swap (writing evicted experts back to RAM): another $\mathbf{1.44\text{ GiB}}$ download.
   - Total PCIe traffic: **~2.88 GiB per generated token**.
   - Kaggle T4 PCIe Gen3 x16 measured speed: **~11.5 GiB/s** upload, **~11.5 GiB/s** download.
   - Minimum transfer time: $\frac{2.88\text{ GiB}}{11.5\text{ GiB/s}} \approx \mathbf{250\text{ milliseconds per token}}$.
   - **Maximum theoretical speed: ~4 tokens/second**, completely stalled by PCIe bandwidth!

### How Strata PR #848 & #864 Actually Solve This
Instead of swapping 12.5 MiB weights across PCIe on every individual token miss:
1. **Activations are tiny, weights are heavy:**
   - Instead of moving 12.5 MiB of weights across PCIe to the GPU, Strata routes the **4 KB activation vector** to the CPU pool in host RAM.
   - The CPU pool computes the missing expert rows directly in host RAM from the pinned complement and returns only the resulting row vectors.
   - Moving 4 KB across PCIe takes **< 1 microsecond**, compared to **1.1 milliseconds** for 12.5 MiB!
2. **Periodic Adaptive Swapping (`--adapt-every`, `--adapt-swaps`, `STRATA_EXCHANGE_ROTATE=1`):**
   - Strata monitors which experts are being routed most frequently during generation.
   - Every $N$ tokens (e.g. 32 or 96 tokens), it executes a controlled batch exchange of the hottest missing experts into VRAM and cold experts back to RAM.
   - With `STRATA_EXCHANGE_ROTATE=1`, the host copy is completely zero-copy (pointer rotation).
   - This elevates the effective VRAM hit rate to **85–95%+**, minimizing CPU pool hits without stalling the token pipeline!

---

## User Review Required

> [!IMPORTANT]
> 1. **Do you approve using Strata's native `--layer-split` + `--resident-experts` + `STRATA_EXCHANGE_ROTATE=1`?**
>    This directly accomplishes your architectural design (22 GB split across GPUs, 26 GB non-duplicated pinned RAM complement, bidirectional rotation) without needing custom kernel rewrites.
> 2. **Kaggle T4 CPU Constraint:** Kaggle instances provide only 2 vCPUs. CPU pool fallback will be slower than on high-core desktops. Maximizing VRAM cache via `--stage-trim` and `--vram-reserve-later-mib` is essential to keep VRAM hit rate above 80%.

---

## Proposed Configuration & Execution Plan

### 1. Optimal Configuration for 2× T4 (Kaggle)
To achieve the exact partitioned allocation without OOM:

```json
{
  "exe": "./build/strata",
  "args": [
    "--gpus", "0,1",
    "--layer-split", "auto",
    "--resident-experts",
    "--resident-pin",
    "--adapt-every", "64",
    "--adapt-swaps", "64",
    "--vram-reserve-mib", "400",
    "--vram-reserve-later-mib", "400"
  ],
  "env": {
    "STRATA_EXCHANGE_ROTATE": "1",
    "STRATA_STAGE_TRIM": "1",
    "STRATA_PREFILL_HELP": "1"
  }
}
```

### 2. Breakdown of Engine Flags
- `--gpus 0,1 --layer-split auto`: Automatically places layers across both T4s proportional to free VRAM (~layers 0–23 on GPU 0, 24–47 on GPU 1).
- `STRATA_STAGE_TRIM=1`: Each GPU loads only its own layers' dense weights, freeing ~3.4 GB of VRAM per card for additional expert cache slots.
- `--resident-experts`: Activates `pin_cache_complement`. Allocates and locks *only* the ~26 GB complement in host RAM, keeping system RAM usage well within Kaggle's 31 GiB limit.
- `STRATA_EXCHANGE_ROTATE=1`: Enables PR #864 zero-copy buffer rotation when experts are swapped between VRAM and RAM.
- `--adapt-every 64 --adapt-swaps 64`: Periodically promotes frequently-routed experts into VRAM in batches, maintaining >85% VRAM hit rates.
- `STRATA_PREFILL_HELP=1`: Allows the idle GPU to assist during prompt prefill chunks over mapped host memory.

---

## Verification Plan

### Automated Checks
1. Verify `exchange_storage_test` and `file_expert_source_test`:
   ```bash
   ctest --test-dir build -R '^(exchange_storage_test|file_expert_source_test)$' --output-on-failure
   ```
2. Verify complement sizing and zero-copy rotation activation:
   ```bash
   ./build/strata generate ... | grep -E "FileExpertSource: exchange buffer rotation|resident RAM mode"
   ```

### Manual Verification on Kaggle
- Check `nvidia-smi` to confirm ~11–12 GiB VRAM allocation per T4.
- Check `free -h` to confirm host RAM stays between 24–27 GiB without hitting swap or OOM.
- Observe token generation rate and verify token-identical output.
