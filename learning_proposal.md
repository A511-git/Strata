# Learning Proposal: Local Host & WSL Build Strategies

## Overview
This document captures the exact toolchain paths, configurations, workarounds, and compilation strategies discovered during the development and compilation of both **Windows native MSVC binaries** and **Linux native Ubuntu/WSL CUDA binaries** on this machine.

Preserving this knowledge ensures that future tasks across any repository on this machine can immediately compile native code without rescanning the filesystem, re-troubleshooting compiler errors, or rediscovering architecture constraints.

---

## 1. Windows Native Compilation (`windows-msvc-build`)

### Classification
- **Environment**: Windows 11 Host
- **Compiler**: Visual Studio 2019 BuildTools (MSVC `cl.exe` 14.29.30133)
- **Target**: Windows x64 PE executables / DLLs

### Toolchain Paths
| Tool | Path |
| :--- | :--- |
| **VC Environment Setup** | `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat` |
| **MSVC C/C++ Compiler** | `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe` |
| **NMake** | `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\nmake.exe` |
| **Standalone CMake** | `C:\Program Files\CMake\bin\cmake.exe` |
| **Git** | `C:\Program Files\Git\cmd\git.exe` |

### Key Windows Workarounds
1. **Generator Choice**:
   - Do NOT use `-G "Visual Studio 16 2019"`. BuildTools is not registered in standard IDE discovery paths via `vswhere`.
   - Always invoke via `vcvars64.bat` using `-G "NMake Makefiles"`.
2. **C1902 PDB Mismatch (`mspdbsrv.exe`)**:
   - Avoid `/Zi` debug format. Use `/Z7` (Embedded debug info: `-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded`).
   - Configure in Release mode: `-DCMAKE_BUILD_TYPE=Release`.
3. **Bypass TryCompile Checks**:
   - Supply `-DCMAKE_CXX_COMPILER_WORKS=TRUE -DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_ABI_COMPILED=TRUE -DCMAKE_C_ABI_COMPILED=TRUE`.
4. **16-Core Parallelism**:
   - Set environment variable `CL=/MP16` or add `/MP16` to compiler flags.

### Windows Standard Build Command
```powershell
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cmake -B build -S . -G ""NMake Makefiles"" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_WORKS=TRUE -DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_ABI_COMPILED=TRUE -DCMAKE_C_ABI_COMPILED=TRUE -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded -DCMAKE_CXX_FLAGS=""/DWIN32 /D_WINDOWS /EHsc /Z7 /O2 /Ob2 /DNDEBUG -std:c++20 /MP16"" -DCMAKE_C_FLAGS=""/DWIN32 /D_WINDOWS /Z7 /O2 /Ob2 /DNDEBUG /MP16"""
cmd.exe /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && set CL=/MP16 && cmake --build build --config Release"
```

---

## 2. Linux Ubuntu / WSL CUDA 12.8 Compilation (`ubuntu-wsl-cuda-build`)

### Classification
- **Environment**: WSL2 Linux (`Ubuntu-24.04`) - **No Docker**
- **Compiler**: GCC/G++ 13 (C++20), NVIDIA `nvcc` 12.8.93 (`cuda-12-8`)
- **Target**: Linux ELF 64-bit x86-64 binary targeting **`sm_75` (Turing: Kaggle dual T4, RTX 20 series)**

### Host & WSL Configuration
1. **WSL Resource Allocation (`C:\Users\A-511\.wslconfig`)**:
   ```ini
   [wsl2]
   processors=16
   memory=12GB
   swap=4GB
   ```
2. **Package & Toolchain Setup Inside Ubuntu 24.04**:
   - **System Tools**: `build-essential`, `cmake` (3.28.3), `ninja-build` (1.11.1), `git`
   - **Lightweight CUDA 12.8 Packages** (Installed via NVIDIA official `cuda-keyring` without heavy GUI/drivers):
     - `cuda-compiler-12-8` (`/usr/local/cuda-12.8/bin/nvcc`)
     - `cuda-cudart-dev-12-8`
     - `libcublas-dev-12-8`

### Turing `sm_75` Compatibility Rules (Ampere -> Turing Backports)
CUDA code written for Ampere (`sm_80`+) requires specific guards and fallbacks to compile and assemble (`ptxas`) for Turing (`sm_75`):

1. **Warp Reduction Intrinsics**:
   - `__reduce_max_sync` and `__reduce_min_sync` were introduced in Ampere (`sm_80`). They are undefined on `sm_75`.
   - **Fix**: Use warp XOR shuffle loops (`__shfl_xor_sync`) for architectures `< 800`:
     ```cuda
     __device__ __forceinline__ int warp_reduce_max_i32(int v) {
     #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
         return __reduce_max_sync(0xffffffffu, v);
     #else
     #pragma unroll
         for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
         return v;
     #endif
     }
     ```
2. **Asynchronous Memory Copy (`cp.async`)**:
   - Instructions like `cp.async.ca.shared.global`, `cp.async.commit_group`, and `cp.async.wait_group` do not exist on `sm_75`.
   - **Fix**: Replace copies with synchronous loads/stores (`uint4` copies) and define `cp_async_commit` / `cp_async_wait` as no-ops for `< 800`.
3. **Tensor Core MMA (`.m16n8k32`)**:
   - Ampere-specific MMA shapes fail inside `ptxas` for `compute_75`.
   - **Fix**: Guard assembly kernels with `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800` and emit `__trap()` on `< 800` so cubins assemble cleanly.
4. **CUDA 12.8 Driver API Signature Updates**:
   - In CUDA 12.8, `cudaMemcpyBatchAsync` takes 9 arguments (requires passing an additional `&fail` pointer for the error index output).

### Ubuntu WSL Build Script (`build-linux.sh`)
```bash
#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin"
export CUDACXX="/usr/local/cuda-12.8/bin/nvcc"
export CUDA_HOME="/usr/local/cuda-12.8"

SRC_DIR="/mnt/c/Users/A-511/OneDrive/Desktop/projects/Strata"
BUILD_DIR="${SRC_DIR}/build-linux"

mkdir -p "${BUILD_DIR}"
cd "${SRC_DIR}"

cmake -B "${BUILD_DIR}" -S "${SRC_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="/usr/local/cuda-12.8/bin/nvcc" \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75" \
  -DSTRATA_PORTABLE=ON

ninja -C "${BUILD_DIR}" strata -j 16
```

### Execution & Verification Commands
- **Run Build from Windows Host**:
  ```powershell
  wsl -d Ubuntu-24.04 bash /mnt/c/Users/A-511/OneDrive/Desktop/projects/Strata/build-linux.sh
  ```
- **Verify Architecture Target**:
  ```bash
  /usr/local/cuda-12.8/bin/cuobjdump --list-elf build-linux/strata
  # Output confirms: strata.1.sm_75.cubin ... strata.56.sm_75.cubin (all sm_75)
  ```
- **Verify Executable**:
  ```bash
  file build-linux/strata
  ./build-linux/strata --help
  ```

---

## 3. Recommended Skill Locations

To make these capabilities available for future sessions:
1. **Global Skill**: `C:\Users\A-511\.gemini\config\skills\local-build-environments\SKILL.md` (usable across any project on this PC).
2. **Workspace Skill**: `c:\Users\A-511\OneDrive\Desktop\projects\Strata\.agents\skills\strata-build\SKILL.md` (committed with repository docs).
