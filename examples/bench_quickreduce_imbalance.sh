#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# QuickReduce imbalance benchmark wrapper.
#
# Usage:
#   ./examples/bench_quickreduce_imbalance.sh                        # baseline (no jitter)
#   JITTER=50 ./examples/bench_quickreduce_imbalance.sh              # 50us uniform jitter
#   JITTER_SWEEP=1 ./examples/bench_quickreduce_imbalance.sh         # sweep 0/25/50/100/200us
#   JITTER_MODE=straggler JITTER=100 ./examples/bench_quickreduce_imbalance.sh
#   EXP_TAG=exp_A JITTER_SWEEP=1 ./examples/bench_quickreduce_imbalance.sh
#   SWEEP=1 JITTER_SWEEP=1 ./examples/bench_quickreduce_imbalance.sh # all sizes x all jitters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "${SCRIPT_DIR}")"

TP="${TP:-8}"
QUANT="${QUANT:-INT4}"
SIZE="${SIZE:-64}"
DTYPE="${DTYPE:-bf16}"
ITERS="${NUM_ITERS:-100}"
WARMUP="${WARMUP:-10}"
PORT="${PORT:-29500}"
SWEEP="${SWEEP:-0}"
EXP_TAG="${EXP_TAG:-baseline}"

# Jitter settings
JITTER="${JITTER:-0}"
JITTER_MODE="${JITTER_MODE:-uniform}"
JITTER_SWEEP="${JITTER_SWEEP:-0}"
STRAGGLER_RANK="${STRAGGLER_RANK:-3}"

RESULT_DIR="${RESULT_DIR:-/home/work/vllm_traces_tp_varied/reports}"
mkdir -p "${RESULT_DIR}"

timestamp=$(date +%Y%m%d_%H%M%S)
output_file="${RESULT_DIR}/qr_${EXP_TAG}_tp${TP}_${QUANT}_${DTYPE}_${timestamp}.jsonl"

echo "=============================================="
echo "  QuickReduce Imbalance Benchmark"
echo "=============================================="
echo "  Tag:    ${EXP_TAG}"
echo "  TP:     ${TP}"
echo "  Quant:  ${QUANT}"
echo "  Dtype:  ${DTYPE}"
echo "  Iters:  ${ITERS} (warmup: ${WARMUP})"
if [ "${SWEEP}" = "1" ]; then
    echo "  Sizes:  16, 32, 64, 128 MB (sweep)"
else
    echo "  Size:   ${SIZE} MB"
fi
if [ "${JITTER_SWEEP}" = "1" ]; then
    echo "  Jitter: sweep 0/25/50/100/200us (mode: ${JITTER_MODE})"
elif [ "${JITTER}" != "0" ]; then
    echo "  Jitter: ${JITTER}us (mode: ${JITTER_MODE})"
else
    echo "  Jitter: none (synchronized arrival)"
fi
echo "  Output: ${output_file}"
echo "=============================================="
echo ""

sweep_flag=""
if [ "${SWEEP}" = "1" ]; then
    sweep_flag="--sweep"
fi

jitter_sweep_flag=""
if [ "${JITTER_SWEEP}" = "1" ]; then
    jitter_sweep_flag="--jitter-sweep"
fi

python "${SCRIPT_DIR}/bench_quickreduce_imbalance.py" \
    --tp "${TP}" \
    --quant "${QUANT}" \
    --size "${SIZE}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --dtype "${DTYPE}" \
    --port "${PORT}" \
    --output "${output_file}" \
    --jitter "${JITTER}" \
    --jitter-mode "${JITTER_MODE}" \
    --straggler-rank "${STRAGGLER_RANK}" \
    ${sweep_flag} \
    ${jitter_sweep_flag}

echo ""
echo "Results saved to: ${output_file}"
