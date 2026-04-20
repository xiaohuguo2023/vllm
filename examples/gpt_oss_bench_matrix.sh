#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GPT-OSS-120B TP4 vs TP8 benchmark matrix runner.
# Runs the focused 11-test matrix from benchmark_design.md.
# Every test collects profiler traces by default (server must have VLLM_SERVE_PROFILE=1).
#
# Usage:
#   bash gpt_oss_bench_matrix.sh --tp 4 8               # All 11 tests, both TP configs
#   bash gpt_oss_bench_matrix.sh --tp 8 --tests B3 B4 B5  # Specific tests
#   bash gpt_oss_bench_matrix.sh --tp 8 --dry-run        # Show what would run

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST="${VLLM_BENCH_HOST:-localhost}"
PORT="${VLLM_BENCH_PORT:-8000}"
MODEL="${VLLM_BENCH_MODEL:-openai/gpt-oss-120b}"
RESULT_BASE="${VLLM_BENCH_RESULT_DIR:-/tmp/vllm_bench_results}"
WARMUP_PROMPTS=4
DRY_RUN=0
declare -a TP_SIZES=()
declare -a SELECTED_TESTS=()

# ── Test Matrix (11 tests, each with a TP4 vs TP8 hypothesis) ─────────────────
# Format: ID ISL OSL CONCURRENCY NUM_PROMPTS HYPOTHESIS
declare -a TEST_MATRIX=(
    # Decode: CustomAR 1-stage (TP4) vs 2-stage (TP8)
    "A1  128   512   1    8      decode_baseline"
    "A4  128   256   64   256    decode_batched"
    # Prefill: QR threshold crossing
    "B3  1024  16    1    16     prefill_below_qr"
    "B4  2048  16    1    16     prefill_qr_split"
    "B5  4096  16    1    16     prefill_qr_int4_vs_fp"
    # Mixed: realistic serving
    "D2  1024  128   64   256    mixed_serving"
    # Long-context: KV cache sharding
    "F1  4096  128   1    8      long_ctx_decode"
    # Latency-throughput curve (4 points)
    "G1  1024  128   1    16     lat_floor"
    "G4  1024  128   8    64     lat_low_conc"
    "G6  1024  128   32   128    lat_near_knee"
    "G7  1024  128   64   256    tput_ceiling"
)

# ── Parse Arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TP_SIZES+=("$1")
                shift
            done
            ;;
        --tests)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_TESTS+=("$1")
                shift
            done
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --result-dir)
            RESULT_BASE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --tp <sizes...> [--tests <ids...>] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --tp <4|8> ...     TP sizes to benchmark (required)"
            echo "  --tests <ID> ...   Only run these specific test IDs (e.g. B3 B4 B5)"
            echo "  --dry-run          Print test plan without executing"
            echo "  --host <host>      Server host (default: localhost)"
            echo "  --port <port>      Server port (default: 8000)"
            echo "  --result-dir <dir> Base result directory"
            echo ""
            echo "Tests:  A1 A4 B3 B4 B5 D2 F1 G1 G4 G6 G7"
            echo ""
            echo "Server must be started with VLLM_SERVE_PROFILE=1 for trace collection."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ ${#TP_SIZES[@]} -eq 0 ]]; then
    echo "ERROR: --tp is required. Example: --tp 4 8"
    exit 1
fi

# ── Helper Functions ──────────────────────────────────────────────────────────

should_run_test() {
    local test_id="$1"
    if [[ ${#SELECTED_TESTS[@]} -eq 0 ]]; then
        return 0
    fi
    for t in "${SELECTED_TESTS[@]}"; do
        [[ "$t" == "$test_id" ]] && return 0
    done
    return 1
}

wait_for_server() {
    local max_wait=600
    local elapsed=0
    echo -n "  Waiting for server at ${HOST}:${PORT} ..."
    while ! curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            echo " TIMEOUT after ${max_wait}s"
            return 1
        fi
        echo -n "."
    done
    echo " ready (${elapsed}s)"
}

save_env_info() {
    local out_dir="$1"
    {
        echo "=== Environment Info ==="
        echo "Date: $(date -Iseconds)"
        echo "Hostname: $(hostname)"
        echo ""
        echo "=== GPU Info ==="
        rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not available"
        echo ""
        echo "=== GPU Clocks ==="
        rocm-smi --showclkfrq 2>/dev/null || true
        echo ""
        echo "=== GPU Temperature ==="
        rocm-smi --showtemp 2>/dev/null || true
        echo ""
        echo "=== XGMI Topology ==="
        rocm-smi --showtopo 2>/dev/null || true
        echo ""
        echo "=== Software Versions ==="
        python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || true
        python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || true
        python3 -c "import triton; print(f'Triton: {triton.__version__}')" 2>/dev/null || true
        echo ""
        echo "=== NUMA ==="
        numactl --show 2>/dev/null || true
    } > "${out_dir}/env_info.txt"
}

run_warmup() {
    echo "  Running ${WARMUP_PROMPTS} warmup requests ..."
    local warmup_args=(
        --host "${HOST}" --port "${PORT}" --model "${MODEL}"
        --backend vllm --endpoint /v1/completions
        --dataset-name random
        --random-input-len 128 --random-output-len 32
        --max-concurrency 1 --num-prompts "${WARMUP_PROMPTS}"
        --ignore-eos
    )
    vllm bench serve "${warmup_args[@]}" > /dev/null 2>&1 || {
        echo "  WARNING: warmup failed, continuing anyway"
    }
    echo "  Warmup complete."
}

run_single_test() {
    local test_id="$1" isl="$2" osl="$3" conc="$4" nprompts="$5" hypothesis="$6"
    local tp="$7" out_dir="$8"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local result_file="${test_id}_i${isl}_o${osl}_c${conc}_n${nprompts}_${timestamp}.json"

    echo ""
    echo "  ┌─ ${test_id}: ${hypothesis} ────────────────────────────"
    echo "  │ ISL=${isl}  OSL=${osl}  Concurrency=${conc}  Prompts=${nprompts}"
    echo "  │ Result: ${out_dir}/${result_file}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  └─ [DRY RUN] skipped"
        return 0
    fi

    local bench_args=(
        --host "${HOST}" --port "${PORT}" --model "${MODEL}"
        --backend vllm --endpoint /v1/completions
        --dataset-name random
        --random-input-len "${isl}" --random-output-len "${osl}"
        --max-concurrency "${conc}" --num-prompts "${nprompts}"
        --percentile-metrics ttft,tpot,itl,e2el
        --ignore-eos
        --profile
        --save-result --result-dir "${out_dir}" --result-filename "${result_file}"
    )

    local start_ts
    start_ts=$(date +%s)

    if vllm bench serve "${bench_args[@]}" 2>&1 | tee "${out_dir}/${test_id}_${timestamp}.log"; then
        local end_ts
        end_ts=$(date +%s)
        local elapsed=$(( end_ts - start_ts ))
        echo "  └─ PASS (${elapsed}s)"
    else
        local end_ts
        end_ts=$(date +%s)
        local elapsed=$(( end_ts - start_ts ))
        echo "  └─ FAIL (${elapsed}s) — check ${out_dir}/${test_id}_${timestamp}.log"
    fi
}

# ── Build Test Plan ───────────────────────────────────────────────────────────

declare -a PLAN=()
for entry in "${TEST_MATRIX[@]}"; do
    read -r tid isl osl conc nprompts hypothesis <<< "$entry"
    if should_run_test "$tid"; then
        PLAN+=("$entry")
    fi
done

if [[ ${#PLAN[@]} -eq 0 ]]; then
    echo "No tests selected. Available: A1 A4 B3 B4 B5 D2 F1 G1 G4 G6 G7"
    exit 1
fi

# ── Print Plan ────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════════"
echo "  GPT-OSS-120B  TP4 vs TP8 Benchmark"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  TP sizes:    ${TP_SIZES[*]}"
echo "  Tests:       ${#PLAN[@]} per TP config"
echo "  Total runs:  $(( ${#PLAN[@]} * ${#TP_SIZES[@]} ))"
echo "  Traces:      ALL (server must have VLLM_SERVE_PROFILE=1)"
echo "  Results:     ${RESULT_BASE}/"
echo ""
printf "  %-4s  %-5s  %-4s  %-4s  %-3s  %s\n" "ID" "ISL" "OSL" "Conc" "N" "Hypothesis"
echo "  ────  ─────  ────  ────  ───  ────────────────────────────"
for entry in "${PLAN[@]}"; do
    read -r tid isl osl conc nprompts hypothesis <<< "$entry"
    printf "  %-4s  %-5s  %-4s  %-4s  %-3s  %s\n" "$tid" "$isl" "$osl" "$conc" "$nprompts" "$hypothesis"
done
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [DRY RUN] No tests will be executed."
    echo ""
fi

# ── Execute ───────────────────────────────────────────────────────────────────

total_start=$(date +%s)

for tp in "${TP_SIZES[@]}"; do
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  TP=${tp}  —  ${#PLAN[@]} tests (all with profiler traces)         ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"

    out_dir="${RESULT_BASE}/tp${tp}"
    mkdir -p "${out_dir}"

    if [[ $DRY_RUN -eq 0 ]]; then
        save_env_info "${out_dir}"

        echo ""
        echo "  Checking server ..."
        if ! wait_for_server; then
            echo "  ERROR: Server not reachable at ${HOST}:${PORT}"
            echo "  Start the server with profiler:"
            echo "    VLLM_TENSOR_PARALLEL_SIZE=${tp} VLLM_SERVE_PROFILE=1 bash gpt_oss_mi350_serve.sh"
            exit 1
        fi

        run_warmup
    fi

    tp_start=$(date +%s)
    pass_count=0
    fail_count=0

    for entry in "${PLAN[@]}"; do
        read -r tid isl osl conc nprompts hypothesis <<< "$entry"
        if run_single_test "$tid" "$isl" "$osl" "$conc" "$nprompts" "$hypothesis" "$tp" "$out_dir"; then
            pass_count=$((pass_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done

    tp_end=$(date +%s)
    tp_elapsed=$(( tp_end - tp_start ))

    echo ""
    echo "  TP${tp} Summary: ${pass_count} passed, ${fail_count} failed, ${tp_elapsed}s elapsed"
done

total_end=$(date +%s)
total_elapsed=$(( total_end - total_start ))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All done. Total elapsed: ${total_elapsed}s"
echo "  Results in: ${RESULT_BASE}/"
echo ""
echo "  Next steps:"
echo "    1. python bench_compare.py ${RESULT_BASE}/tp4 ${RESULT_BASE}/tp8"
echo "    2. python trace_compare.py <tp4_trace> <tp8_trace> -o <test>_analysis.md"
echo "═══════════════════════════════════════════════════════════════════"
