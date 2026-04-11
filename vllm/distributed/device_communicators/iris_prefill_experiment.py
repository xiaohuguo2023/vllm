# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Iris CCL communicator for large-message TP8 all-reduce (v4: with barrier).

Uses async_op=False so that a device_barrier synchronizes all ranks after
each all-reduce.  v3 used async_op=True which eliminated the barrier but
caused cumulative rank drift that inflated NCCL latency by 69x.

Gated by VLLM_USE_IRIS_PREFILL_EXPERIMENT=1. ROCm + TP8 only.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger

logger = init_logger(__name__)

_SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16})
_REQUIRED_WORLD_SIZE = 8


class IrisPrefillExperiment:
    _MAX_BUF_CACHE_ENTRIES: int = 32

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
    ) -> None:
        self.disabled = True

        if not envs.VLLM_USE_IRIS_PREFILL_EXPERIMENT:
            logger.info_once(
                "Iris prefill experiment: disabled "
                "(VLLM_USE_IRIS_PREFILL_EXPERIMENT != 1).",
                scope="global",
            )
            return

        import torch.distributed as dist

        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        if self.world_size != _REQUIRED_WORLD_SIZE:
            logger.warning(
                "Iris prefill experiment: disabled, "
                "world_size=%d but only %d is supported.",
                self.world_size,
                _REQUIRED_WORLD_SIZE,
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.variant: str = envs.VLLM_IRIS_PREFILL_VARIANT
        self.min_bytes: int = envs.VLLM_IRIS_PREFILL_MIN_BYTES
        self.max_bytes: int = envs.VLLM_IRIS_PREFILL_MAX_BYTES
        heap_size_bytes: int = envs.VLLM_IRIS_PREFILL_HEAP_SIZE_MB * 1024 * 1024

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and hasattr(vllm_config, "model_config")
            and vllm_config.model_config is not None
        ):
            self.hidden_size: int = vllm_config.model_config.get_hidden_size()
        else:
            logger.warning(
                "Iris prefill experiment: disabled, "
                "cannot determine hidden_size from vllm config."
            )
            return

        if self.hidden_size <= 0:
            logger.warning(
                "Iris prefill experiment: disabled, hidden_size=%d.",
                self.hidden_size,
            )
            return

        try:
            import iris
            from iris.ccl import Config
        except ImportError:
            logger.warning(
                "Iris prefill experiment: disabled, "
                "cannot import iris / iris.ccl."
            )
            return

        try:
            self.shmem = iris.iris(heap_size_bytes)
        except Exception:
            logger.exception(
                "Iris prefill experiment: disabled, iris.iris() init failed."
            )
            return

        self.config = Config(
            all_reduce_variant=self.variant,
            comm_sms=64,
            block_size_m=64,
            block_size_n=64,
            swizzle_size=4,
            all_reduce_distribution=1,
        )

        self._buf_cache: OrderedDict[
            tuple[int, int, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, object],
        ] = OrderedDict()

        self._claimed_calls: int = 0
        self._claimed_bytes: int = 0
        self._fallback_calls: int = 0
        self._log_interval: int = 500
        self._total_calls: int = 0
        self._reject_reasons: dict[str, int] = {}
        self._size_histogram: dict[str, int] = {}
        self._shape_details: dict[str, int] = {}
        self._claimed_shapes: dict[str, int] = {}

        self.disabled = False
        logger.info(
            "Iris prefill experiment v4: ENABLED (sync, with barrier)  "
            "variant=%s  hidden=%d  min_bytes=%d  max_bytes=%d  "
            "heap_mb=%d  rank=%d  world=%d",
            self.variant,
            self.hidden_size,
            self.min_bytes,
            self.max_bytes,
            envs.VLLM_IRIS_PREFILL_HEAP_SIZE_MB,
            self.rank,
            self.world_size,
        )

    # -- gating ---------------------------------------------------------------

    def should_claim(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False

        payload_bytes = inp.numel() * inp.element_size()

        if inp.dtype not in _SUPPORTED_DTYPES:
            self._track_reject("dtype", payload_bytes, inp)
            return False

        if payload_bytes < self.min_bytes:
            self._track_reject("too_small", payload_bytes, inp)
            return False

        if payload_bytes > self.max_bytes:
            self._track_reject("too_large", payload_bytes, inp)
            return False

        if not (inp.is_contiguous() or self._is_weak_contiguous(inp)):
            self._track_reject("not_contiguous", payload_bytes, inp)
            return False

        if inp.dim() < 1:
            self._track_reject("scalar", payload_bytes, inp)
            return False

        N = inp.shape[-1]
        numel = inp.numel()
        if N == 0 or numel % N != 0:
            self._track_reject("shape_N", payload_bytes, inp)
            return False

        M = numel // N

        shape_key = f"{list(inp.shape)}|{inp.dtype}|{payload_bytes}B|M={M}|N={N}"
        self._claimed_shapes[shape_key] = (
            self._claimed_shapes.get(shape_key, 0) + 1
        )
        self._track_reject("CLAIMED", payload_bytes)
        return True

    def _track_reject(self, reason: str, payload_bytes: int,
                       inp: torch.Tensor | None = None) -> None:
        self._reject_reasons[reason] = self._reject_reasons.get(reason, 0) + 1
        mb = payload_bytes / (1024 * 1024)
        bucket = f"{mb:.1f}MB" if mb < 1 else f"{mb:.0f}MB"
        key = f"{reason}:{bucket}"
        self._size_histogram[key] = self._size_histogram.get(key, 0) + 1

        if inp is not None and reason != "CLAIMED" and mb >= 1.0:
            shape_key = f"{reason}|{list(inp.shape)}|{inp.dtype}|{payload_bytes}B"
            self._shape_details[shape_key] = (
                self._shape_details.get(shape_key, 0) + 1
            )

        self._total_calls += 1
        if self.rank == 0 and self._total_calls % self._log_interval == 0:
            self._dump_stats()

    # -- all-reduce -----------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_claim(inp):
            self._fallback_calls += 1
            return None

        N = inp.shape[-1]
        numel = inp.numel()
        M = numel // N
        dtype = inp.dtype

        input_buf, output_buf, workspace = self._get_or_alloc(M, N, dtype)

        input_buf.copy_(inp.view(M, N))

        workspace = self.shmem.ccl.all_reduce_preamble(
            output_buf,
            input_buf,
            config=self.config,
            workspace=workspace,
        )

        self._buf_cache[(M, N, dtype)] = (input_buf, output_buf, workspace)
        self._buf_cache.move_to_end((M, N, dtype))

        self.shmem.ccl.all_reduce(
            output_buf,
            input_buf,
            config=self.config,
            async_op=False,
            workspace=workspace,
        )

        out = torch.empty_like(inp)
        out.view(M, N).copy_(output_buf)

        self._claimed_calls += 1
        self._claimed_bytes += numel * inp.element_size()

        return out

    # -- buffer management ----------------------------------------------------

    def _get_or_alloc(
        self, M: int, N: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        key = (M, N, dtype)
        cached = self._buf_cache.get(key)
        if cached is not None:
            self._buf_cache.move_to_end(key)
            return cached

        input_buf = self.shmem.zeros((M, N), dtype=dtype)
        output_buf = self.shmem.zeros((M, N), dtype=dtype)
        workspace = None
        if len(self._buf_cache) >= self._MAX_BUF_CACHE_ENTRIES:
            self._buf_cache.popitem(last=False)
        self._buf_cache[key] = (input_buf, output_buf, workspace)
        return input_buf, output_buf, workspace

    # -- helpers --------------------------------------------------------------

    def _dump_stats(self) -> None:
        sorted_hist = sorted(
            self._size_histogram.items(),
            key=lambda kv: -kv[1],
        )
        hist_lines = [f"    {k}: {v}" for k, v in sorted_hist[:20]]

        sorted_shapes = sorted(
            self._shape_details.items(),
            key=lambda kv: -kv[1],
        )
        shape_lines = [f"    {k}: {v}" for k, v in sorted_shapes[:15]]

        sorted_claimed = sorted(
            self._claimed_shapes.items(),
            key=lambda kv: -kv[1],
        )
        claimed_lines = [f"    {k}: {v}" for k, v in sorted_claimed[:10]]

        logger.info(
            "Iris dispatch stats (rank 0, after %d calls):\n"
            "  claimed=%d  fallback=%d\n"
            "  reasons: %s\n"
            "  top size buckets:\n%s\n"
            "  rejected shapes (>=1MB):\n%s\n"
            "  claimed shapes:\n%s",
            self._total_calls,
            self._claimed_calls,
            self._fallback_calls,
            dict(self._reject_reasons),
            "\n".join(hist_lines),
            "\n".join(shape_lines) if shape_lines else "    (none)",
            "\n".join(claimed_lines) if claimed_lines else "    (none)",
        )

    @staticmethod
    def _is_weak_contiguous(inp: torch.Tensor) -> bool:
        return (
            inp.untyped_storage().nbytes() - inp.storage_offset() * inp.element_size()
            == inp.numel() * inp.element_size()
        )

    def destroy(self) -> None:
        if hasattr(self, "shmem") and self.shmem is not None:
            try:
                self.shmem.barrier()
            except Exception:
                pass
            self.shmem = None
        if hasattr(self, "_buf_cache"):
            self._buf_cache.clear()
        self.disabled = True

    def __del__(self) -> None:
        self.destroy()
