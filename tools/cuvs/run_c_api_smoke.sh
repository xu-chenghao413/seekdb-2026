#!/usr/bin/env bash
set -euo pipefail

CUVS_PREFIX="${CUVS_PREFIX:-/opt/cuvs-env}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/seekdb-cuvs-c-api-smoke"

g++ -std=c++17 \
  -I"${CUVS_PREFIX}/include" \
  -I"${CUVS_PREFIX}/targets/x86_64-linux/include" \
  -L"${CUVS_PREFIX}/lib" \
  -Wl,-rpath,"${CUVS_PREFIX}/lib" \
  "${ROOT_DIR}/tools/cuvs/cuvs_c_api_smoke.cpp" \
  -lcuvs_c -lcuvs -lrmm -lrapids_logger \
  -lcudart -lcublas -lcusolver -lcusparse -lcurand \
  -o "${OUT}"

"${OUT}"
