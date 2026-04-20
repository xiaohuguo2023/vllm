#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch vLLM server for gpt-oss on AMD MI350X / MI355X with a *wider*
# CUDA/HIP-graph capture window so that prefill batches up to 2048
# tokens run inside a captured graph instead of falling back to eager.
#
# This is the experiment companion to gpt_oss_mi350_serve_orig.sh:
# every flag/env is identical except for the COMPILE_SIZES list and the
# matching --compilation-config block (compile_sizes, cudagraph_capture_sizes,
# max_cudagraph_capture_size).
#
# Usage: ./examples/gpt_oss_mi350_serve_widegraph.sh
# Or:    bash examples/gpt_oss_mi350_serve_widegraph.sh
#
# Environment variables:
#   VLLM_TENSOR_PARALLEL_SIZE  TP size (default: 8)
#   VLLM_SERVE_PROFILE         Set to 1 to enable torch profiler (default: off)
#   VLLM_MAX_MODEL_LEN         Max sequence length (default: 10368)
#   VLLM_MAX_NUM_SEQS          Max concurrent sequences (default: 256)
#
# Notes:
#   - Capturing graphs up to 2048 tokens roughly doubles startup time
#     and consumes additional GPU memory for the captured graph pool.
#     If you OOM during capture, drop the largest entries from
#     COMPILE_SIZES (e.g. trim 2048, then 1536, then 1280) and lower
#     max_cudagraph_capture_size accordingly.
#   - For graph capture to actually be hit, every Triton JIT shape used
#     by these batch sizes must be pre-warmed. Use the same
#     TRITON_CACHE_DIR you use for aiter_baseline so the JIT modules
#     are already on disk before the first capture pass.

set -e

# --- Config (edit as needed) ---
model="${VLLM_MODEL:-openai/gpt-oss-120b}"

max_model_len="${VLLM_MAX_MODEL_LEN:-10368}"          # 1.125 x (ISL+OSL)
max_num_seqs="${VLLM_MAX_NUM_SEQS:-256}"              # sufficient for benchmarks, saves KV cache
tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
port="${VLLM_PORT:-8000}"
enable_profile="${VLLM_SERVE_PROFILE:-0}"

# AITER/ROCm (override with env if needed).
# These defaults match the ones that produced the `aiter_baseline` traces
# (CK MoeFlatmmKernel, MoE Sorting, ck_tile Rmsnorm2dFwd). Without
# VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 the run silently
# falls back to the v3 newbaseline kernel set (MXFP4 MoE GEMM, TopK +
# pack_bitmatrix, Triton fused RMSNorm), which is ~55% slower on the
# bottleneck rank than aiter_baseline.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-1}"
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
export VLLM_USE_AITER_UNIFIED_ATTENTION="${VLLM_USE_AITER_UNIFIED_ATTENTION:-1}"
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
export VLLM_ROCM_USE_AITER_MHA="${VLLM_ROCM_USE_AITER_MHA:-0}"
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4="${VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4:-1}"
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

# Persistent Triton JIT cache. Wider capture sizes (576..2048) include
# Triton kernels that are not in the original 1..512 cache. Pointing
# TRITON_CACHE_DIR at the same dir used to prime aiter_baseline avoids
# hipModuleLoadDataEx stalls firing inside captured graphs on first hit.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/work/.triton_cache_gpt_oss}"
mkdir -p "${TRITON_CACHE_DIR}"

# Wide capture window: original 1..512 list, extended through prefill-sized
# batches (576..2048). compile_sizes must remain a subset of
# cudagraph_capture_sizes; here we keep them identical so every compiled
# shape is also a captured graph.
COMPILE_SIZES='[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,576,640,768,896,1024,1152,1280,1536,2048]'

# Build profiler args (off by default to avoid overhead during benchmarking)
profiler_args=()
if [[ "${enable_profile}" == "1" ]]; then
    trace_dir="/tmp/vllm_traces_tp${tensor_parallel_size}_widegraph"
    mkdir -p "${trace_dir}"
    profiler_args=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${trace_dir}\"}")
    echo "Profiler ENABLED → ${trace_dir}"
else
    echo "Profiler disabled (set VLLM_SERVE_PROFILE=1 to enable)"
fi

echo "Starting vLLM server (wide HIP-graph capture): model=${model} port=${port} tp=${tensor_parallel_size}"
echo "  compile_sizes / cudagraph_capture_sizes: ${COMPILE_SIZES}"
echo "  max_cudagraph_capture_size: 2048"
echo "  AITER env:"
echo "    VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}"
echo "    VLLM_ROCM_USE_AITER_MOE=${VLLM_ROCM_USE_AITER_MOE}"
echo "    VLLM_ROCM_USE_AITER_RMSNORM=${VLLM_ROCM_USE_AITER_RMSNORM}"
echo "    VLLM_USE_AITER_UNIFIED_ATTENTION=${VLLM_USE_AITER_UNIFIED_ATTENTION}"
echo "    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION}"
echo "    VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=${VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4}"
echo "    VLLM_ROCM_USE_AITER_MHA=${VLLM_ROCM_USE_AITER_MHA}"
echo "    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION}"
echo "    TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

vllm serve "${model}" \
  --port "${port}" \
  --max-model-len "${max_model_len}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --max-num-seqs "${max_num_seqs}" \
  --gpu-memory-utilization 0.95 \
  --compilation-config "{\"compile_sizes\": ${COMPILE_SIZES}, \"cudagraph_capture_sizes\": ${COMPILE_SIZES}, \"max_cudagraph_capture_size\": 2048}" \
  --block-size 64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  "${profiler_args[@]}"
