"""Private test fixture for the retained experimental kernels; not a package API.

The standard EPBuffer never selects this fixture. The Context timeout parameters
below are not deadlines for LL packet/count polling (EXP-TIMEOUT-01 is pending).
Run device tests with finite global JIT and external job timeouts.
"""

import torch
import torch.distributed as dist

from deep_ep import _C, comm
from deep_ep.buffers.base import BufferBase
from deep_ep.buffers.low_latency import LowLatencyState


class _PrivateLowLatencyHandle:
    """Expose experimental test metadata without exposing a native EP handle.

    Deliberately omit native routing fields. Passing this object into standard
    EPBuffer fails in Python before dispatch/combine can call the native runtime.
    """

    _metadata = frozenset(('is_low_latency', 'num_expanded_tokens',
                           'num_recv_tokens_per_expert_list', 'expert_recv_counts',
                           'send_counts', 'relay_read_index'))

    def __init__(self, owner, handle):
        self._owner = owner
        self._handle = handle

    def __getattr__(self, name):
        if name in self._metadata:
            return getattr(self._handle, name)
        raise AttributeError(f'Private LL test handle has no native field {name!r}')


class _PrivateLowLatencyBuffer(BufferBase):
    """Compose existing private runtime/adapter solely for the experimental tests."""

    def __init__(self, group, num_max_tokens_per_rank, *, use_huge1g_page=True,
                 context_cpu_timeout_secs=300, context_gpu_timeout_secs=100):
        if group.size() not in (8, 16, 32, 64, 128, 256):
            raise ValueError('Private LL requires EP8/16/32/64/128/256')
        if not 0 < num_max_tokens_per_rank <= 256:
            raise ValueError('Private LL requires token capacity in [1,256]')
        if context_cpu_timeout_secs <= 0 or context_gpu_timeout_secs <= 0:
            raise ValueError('Context timeout values must be positive')
        super().__init__(explicitly_destroy=True)
        self.group = group
        self.rank_idx, self.num_ranks = group.rank(), group.size()
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self._low_latency = LowLatencyState()
        self.hccl_comm_handle = comm.get_hccl_comm_handle(group)
        self.runtime = _C.LowLatencyBuffer(
            self.rank_idx, self.num_ranks, self.hccl_comm_handle.get(),
            use_huge1g_page, context_cpu_timeout_secs, context_gpu_timeout_secs, True)
        self.context = self.runtime.context
        torch.npu.synchronize()
        dist.barrier(group)
        torch.npu.synchronize()

    def dispatch(self, x, topk_idx=None, topk_weights=None,
                 cumulative_local_expert_recv_stats=None, num_experts=None,
                 num_max_tokens_per_rank=None, expert_alignment=None, num_sms=0,
                 num_qps=0, previous_event=None, async_with_compute_stream=False,
                 allocate_on_comm_stream=False, handle=None, do_handle_copy=False,
                 do_cpu_sync=None, do_expand=False, do_zero_padding=False,
                 use_tma_aligned_col_major_sf=False, defer_epilogue=False):
        result = self._low_latency.dispatch(
            self, x, topk_idx, topk_weights, cumulative_local_expert_recv_stats,
            num_experts, num_max_tokens_per_rank, expert_alignment, num_sms, num_qps,
            previous_event, async_with_compute_stream, allocate_on_comm_stream, handle,
            do_handle_copy, do_cpu_sync, do_expand, do_zero_padding,
            use_tma_aligned_col_major_sf, defer_epilogue)
        recv, indices, weights, private_handle, event = result
        return recv, indices, weights, _PrivateLowLatencyHandle(self, private_handle), event

    def combine(self, x, handle, topk_weights=None, bias=None, num_sms=0, num_qps=0,
                previous_event=None, async_with_compute_stream=False,
                allocate_on_comm_stream=False, defer_epilogue=False):
        if not isinstance(handle, _PrivateLowLatencyHandle) or handle._owner is not self:
            raise ValueError('Private LL combine requires a test handle from this buffer')
        return self._low_latency.combine(
            self, x, handle._handle, topk_weights, bias, num_sms, num_qps,
            previous_event, async_with_compute_stream, allocate_on_comm_stream,
            defer_epilogue)

    def destroy(self):
        super().destroy()
        self.context = None
        self.hccl_comm_handle = None
        self._low_latency.tilings.clear()
        self._low_latency.last_event = None
        self._low_latency.inflight = None
