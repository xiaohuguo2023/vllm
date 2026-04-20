#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch vLLM server for gpt-oss on NVIDIA B200.
# Usage: ./examples/gpt_oss_b200_serve.sh
# Or:    bash examples/gpt_oss_b200_serve.sh
#
# Environment variables:
#   VLLM_TENSOR_PARALLEL_SIZE  TP size (default: 8)
#   VLLM_SERVE_PROFILE         Set to 1 to enable torch profiler (default: off)
#   VLLM_MAX_MODEL_LEN         Max sequence length (default: 10368)
#   VLLM_MAX_NUM_SEQS          Max concurrent sequences (default: 256)

set -e

# --- Config (edit as needed) ---
model="${VLLM_MODEL:-openai/gpt-oss-120b}"

max_model_len="${VLLM_MAX_MODEL_LEN:-10368}"          # 1.125 x (ISL+OSL)
max_num_seqs="${VLLM_MAX_NUM_SEQS:-256}"              # sufficient for benchmarks, saves KV cache
tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
port="${VLLM_PORT:-8000}"
enable_profile="${VLLM_SERVE_PROFILE:-0}"

# Compilation sizes — must be a subset of cudagraph_capture_sizes (padding-aligned)
COMPILE_SIZES='[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512]'

# Build profiler args (off by default to avoid overhead during benchmarking)
profiler_args=()
if [[ "${enable_profile}" == "1" ]]; then
    trace_dir="${VLLM_TRACE_DIR:-/tmp/vllm_traces_tp${tensor_parallel_size}}"
    mkdir -p "${trace_dir}"
    profiler_args=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${trace_dir}\"}")
    echo "Profiler ENABLED → ${trace_dir}"
else
    echo "Profiler disabled (set VLLM_SERVE_PROFILE=1 to enable)"
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GPT-OSS vLLM Server (NVIDIA B200)                            ║"
echo "║  Model:  ${model}"
echo "║  TP:     ${tensor_parallel_size}    Port: ${port}"
echo "║  MaxLen: ${max_model_len}    MaxSeqs: ${max_num_seqs}"
echo "╚══════════════════════════════════════════════════════════════════╝"

vllm serve "${model}" \
  --port "${port}" \
  --swap-space 64 \
  --max-model-len "${max_model_len}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --max-num-seqs "${max_num_seqs}" \
  --gpu-memory-utilization 0.95 \
  --compilation-config "{\"compile_sizes\": ${COMPILE_SIZES}}" \
  --block-size 128 \
  --no-enable-prefix-caching \
  --async-scheduling \
  "${profiler_args[@]}"
