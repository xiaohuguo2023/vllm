#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run vLLM benchmark client against a running vLLM server on NVIDIA B200
# (e.g. started with gpt_oss_b200_serve.sh).
# Usage: ./examples/gpt_oss_b200_bench_client.sh
# Or:    bash examples/gpt_oss_b200_bench_client.sh
# Ensure the server is ready before running.

set -e

# --- Config (edit as needed) ---
host="${VLLM_BENCH_HOST:-localhost}"
port="${VLLM_BENCH_PORT:-8000}"
model="${VLLM_BENCH_MODEL:-openai/gpt-oss-120b}"

input_tokens="${VLLM_BENCH_INPUT_TOKENS:-1024}"
output_tokens="${VLLM_BENCH_OUTPUT_TOKENS:-128}"
max_concurrency="${VLLM_BENCH_MAX_CONCURRENCY:-64}"
num_prompts="${VLLM_BENCH_NUM_PROMPTS:-256}"

result_dir="${VLLM_BENCH_RESULT_DIR:-/tmp/vllm_bench_results}"
mkdir -p "${result_dir}"

timestamp=$(date +%Y%m%d_%H%M%S)
result_file="bench_b200_i${input_tokens}_o${output_tokens}_c${max_concurrency}_n${num_prompts}_${timestamp}.json"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  GPT-OSS Benchmark Client (NVIDIA B200)                       ║"
echo "║  Server: ${host}:${port}"
echo "║  Model:  ${model}"
echo "║  ISL=${input_tokens}  OSL=${output_tokens}  Concurrency=${max_concurrency}  Prompts=${num_prompts}"
echo "║  Results → ${result_dir}/${result_file}"
echo "╚══════════════════════════════════════════════════════════════════╝"

vllm bench serve \
  --host "${host}" \
  --port "${port}" \
  --model "${model}" \
  --backend vllm \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len "${input_tokens}" \
  --random-output-len "${output_tokens}" \
  --max-concurrency "${max_concurrency}" \
  --num-prompts "${num_prompts}" \
  --percentile-metrics ttft,tpot,itl,e2e1 \
  --ignore-eos \
  --profile \
  --save-result \
  --result-dir "${result_dir}" \
  --result-filename "${result_file}"
