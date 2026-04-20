#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run vLLM benchmark client for all workload sizes (small, medium, large) without profiling.
# Usage: ./examples/gpt_oss_mi350_bench_all.sh
# Ensure the server is ready before running.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for workload in small medium large; do
    echo ""
    echo "=========================================="
    echo "  Running workload: ${workload}"
    echo "=========================================="
    echo ""
    WORKLOAD="${workload}" bash "${SCRIPT_DIR}/gpt_oss_mi350_bench_client_orig.sh"
    echo ""
    echo "  ✓ ${workload} complete"
    echo ""
done

echo "=========================================="
echo "  All workloads complete!"
echo "  Results:"
result_dir="${VLLM_BENCH_RESULT_DIR:-/tmp/vllm_bench_results}"
for workload in small medium large; do
    echo "    ${workload}: $(ls -t "${result_dir}/${workload}/"*.json 2>/dev/null | head -1)"
done
echo "=========================================="
