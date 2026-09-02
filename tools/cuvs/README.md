# cuVS backend (opt-in)

This directory contains the opt-in cuVS backend for seekdb. The historical
VSAG/OB paths remain unchanged unless a vector index explicitly uses
`LIB=CUVS`.

## Environment used for validation

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0)
- Windows driver: 592.15
- WSL CUDA toolkit: 13.0 (`nvcc` 13.0.88)
- cuVS: 26.06.00, CUDA 13 package, installed at `/opt/cuvs-env`

## Build seekdb with cuVS

```bash
cd /root/oceanbase-competition-2026/seekdb-2026
./build.sh release --init -DOB_ENABLE_CUVS=ON -DCUVS_PREFIX=/opt/cuvs-env
./build.sh release --make -j3
```

The build embeds `/opt/cuvs-env/lib` in the executable RPATH. Set
`CUVS_PREFIX` to the installation prefix when using a different environment.

## Run the standalone smoke tests

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

## Use from SQL

With a cuVS-enabled binary, create a dense FP32 HNSW index as follows:

```sql
CREATE TABLE cuvs_demo (
  id BIGINT PRIMARY KEY,
  embedding VECTOR(8),
  VECTOR INDEX idx_embedding (embedding)
    WITH (DISTANCE=L2, TYPE=HNSW, LIB=CUVS, M=16,
          EF_CONSTRUCTION=100, EF_SEARCH=64)
);
```

The adapter keeps seekdb's logical IDs and extra-info buffers, performs
seekdb-compatible host-side filtering, supports CPU-hierarchy incremental
extension, and stores cuVS's serialized graph together with seekdb metadata.
Current scope is dense `float32` HNSW (including the HGRAPH representation
used internally when extra-info filtering is enabled). Sparse, SQ/BQ, IVF,
and hybrid indexes are rejected explicitly; use VSAG/OB for those forms.

## Why this is separate from the default `LIB=VSAG`

cuVS uses RAFT/RMM and GPU-resident index state, while seekdb's VSAG adaptor
owns CPU pointers, custom allocators, filtering callbacks, and snapshot
serialization. The opt-in adapter bridges those contracts without replacing
`libvsag_static.a`; VSAG remains the default.
