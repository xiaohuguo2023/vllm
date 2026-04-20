#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Experiment: NUMA bind + AITER MoE routing + persistent Triton cache.
#
# Goal: On top of the NUMA binding that we already know reduces the L1
#       gap_delta by ~70%, also eliminate the legacy triton_kernels
#       SortTokens and hipModuleLoadDataEx stalls identified in
#       docs/prefill_l1_p0_root_cause.md.
#
# Delegates to gpt_oss_mi350_serve_numa.sh — does not modify it.
#
# Usage:
#   VLLM_SERVE_PROFILE=1 bash examples/gpt_oss_mi350_serve_numa_aiter.sh

set -e

# --- AITER master switch + MoE routing path ---
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"

# --- Persistent Triton JIT cache ---
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/work/.triton_cache_gpt_oss}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "[numa+aiter-experiment] VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}"
echo "[numa+aiter-experiment] VLLM_ROCM_USE_AITER_MOE=${VLLM_ROCM_USE_AITER_MOE}"
echo "[numa+aiter-experiment] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

# Delegate to the NUMA serve script. NUMA binding, tp, compile_sizes,
# profiler wiring are all inherited unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/gpt_oss_mi350_serve_numa.sh"
