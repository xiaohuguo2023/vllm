#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Experiment: Baseline + AITER MoE routing + persistent Triton cache.
#
# Goal: Eliminate `SortTokens` (torch.autograd.Function) CPU-op stalls and
#       `hipModuleLoadDataEx` JIT-compile stalls observed at L1.P0 in the
#       baseline trace. See docs/prefill_l1_p0_root_cause.md.
#
# Delegates to gpt_oss_mi350_serve_orig.sh — does not modify the baseline.
#
# Usage:
#   VLLM_SERVE_PROFILE=1 bash examples/gpt_oss_mi350_serve_aiter.sh

set -e

# --- AITER master switch + MoE routing path ---
# These two flip dispatch in gpt_oss_triton_kernels_moe.py:133 from the
# legacy triton_kernels SortTokens path to aiter.ops.triton.moe_routing.routing.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"

# --- Persistent Triton JIT cache ---
# First-hit kernel compiles still happen, but cache is reused on subsequent
# server starts so hipModuleLoadDataEx on the hot path collapses to module
# reload (~50-100us) instead of full compile (~1000us).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/work/.triton_cache_gpt_oss}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "[aiter-experiment] VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}"
echo "[aiter-experiment] VLLM_ROCM_USE_AITER_MOE=${VLLM_ROCM_USE_AITER_MOE}"
echo "[aiter-experiment] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

# Delegate to the original baseline serve script. All other behavior
# (tp, compile_sizes, profiler wiring, quick_reduce, etc.) is inherited.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/gpt_oss_mi350_serve_orig.sh"
