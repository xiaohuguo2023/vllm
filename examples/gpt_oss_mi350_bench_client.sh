#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run vLLM benchmark client against a running vLLM server (e.g. started with gpt_oss_mi350_serve.sh).
# Usage: ./examples/gpt_oss_mi350_bench_client.sh
# Or:    bash examples/gpt_oss_mi350_bench_client.sh
# Ensure the server is ready before running.

set -e

# --- Config (edit as needed) ---
host="${VLLM_BENCH_HOST:-localhost}"
port="${VLLM_BENCH_PORT:-8000}"
model="${VLLM_BENCH_MODEL:-openai/gpt-oss-120b}"
# model=amd/Llama-3.1-4058-Instruct-MXFP4-Preview

input_tokens="${VLLM_BENCH_INPUT_TOKENS:-1024}"
output_tokens="${VLLM_BENCH_OUTPUT_TOKENS:-128}"
max_concurrency="${VLLM_BENCH_MAX_CONCURRENCY:-64}"
num_prompts="${VLLM_BENCH_NUM_PROMPTS:-256}"

result_dir="${VLLM_BENCH_RESULT_DIR:-/tmp/vllm_bench_results}"
mkdir -p "${result_dir}"

timestamp=$(date +%Y%m%d_%H%M%S)
result_file="bench_i${input_tokens}_o${output_tokens}_c${max_concurrency}_n${num_prompts}_${timestamp}.json"

echo "Running vLLM benchmark: host=${host} port=${port} model=${model}"
echo "  input_tokens=${input_tokens} output_tokens=${output_tokens} max_concurrency=${max_concurrency} num_prompts=${num_prompts}"
echo "  results -> ${result_dir}/${result_file}"

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
