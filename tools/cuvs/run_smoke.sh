#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
CUVS_PREFIX="${CUVS_PREFIX:-/opt/cuvs-env}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/seekdb-cuvs-hnsw-smoke"

"${CUDA_ROOT}/bin/nvcc" \
  -std=c++17 \
  -DCUVS_BUILD_CAGRA_HNSWLIB \
  -DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
  -I"${CUVS_PREFIX}/include/rapids" \
  -I"${CUVS_PREFIX}/include" \
  -I"${CUVS_PREFIX}/targets/x86_64-linux/include" \
  -L"${CUVS_PREFIX}/lib" \
  -Xlinker -rpath -Xlinker "${CUVS_PREFIX}/lib" \
  "${ROOT_DIR}/tools/cuvs/cuvs_smoke.cpp" \
  -lcuvs -lrmm -lrapids_logger \
  -o "${OUT}"

"${OUT}"
