# cuVS experimental backend

This directory is the first, independently runnable cuVS integration step for
seekdb. It validates the NVIDIA GPU and the cuVS C++ ABI without changing the
default VSAG path used by seekdb.

## Environment used for validation

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0)
- Windows driver: 592.15
- WSL CUDA toolkit: 13.0 (`nvcc` 13.0.88)
- cuVS: 26.06.00, CUDA 13 package, installed at `/opt/cuvs-env`

## Run the smoke test

```bash
cd /root/oceanbase-competition-2026/seekdb-2026
./tools/cuvs/run_smoke.sh
./tools/cuvs/run_c_api_smoke.sh
```

The test builds a small HNSW index through cuVS's GPU CAGRA/HNSW path, runs a
host-side query, and checks that the nearest row is returned. It should print a
line similar to:

```text
nearest=42 distance=0
```

The script accepts `CUDA_ROOT` and `CUVS_PREFIX` overrides when the toolkit or
conda prefix is installed elsewhere.

`run_c_api_smoke.sh` exercises the same path through cuVS's C API. This is the
ABI intended for the eventual seekdb adaptor because it can be compiled as a
normal C++ translation unit while cuVS owns CUDA/RMM resources.

## Why this is separate from SQL `LIB=VSAG`

cuVS uses RAFT/RMM and GPU-resident index state, while seekdb's current VSAG
adaptor owns CPU pointers, custom allocators, filtering callbacks, and snapshot
serialization. Replacing `libvsag_static.a` with cuVS is therefore unsafe.
The next implementation step is an opt-in `LIB=CUVS` adaptor for dense FP32
HNSW/CAGRA only; VSAG remains the default and sparse/BQ/HGraph variants stay
unchanged until their semantics and persistence are covered by tests.
