"""Legacy explicit experimental adapter for the Ascend decode kernels.

The default EPBuffer integrates native transport selection with the standard
metadata and epilogues. This retained private adapter is exercised through
tests/ep/_low_latency_fixture.py; it is not selected by the public EPBuffer API.
Its internal handles must stay inside that fixture, which returns opaque test
handles and rejects native/foreign handles. DecodeBuffer uses native EPBuffer.
Context timeout values do not bound LL packet/count polling (EXP-TIMEOUT-01).

All ranks must use identical B/M/H/K/E and call order. Routing entries must be
valid expert IDs (no masked -1 entries), with no duplicate expert per token.
Dispatch still transports weight metadata. Combine only sums expert outputs;
any routing weights must already have been applied by the expert computation.
"""

import threading

import torch

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from ..utils.event import EventHandle, EventOverlap


class LowLatencyState:
    def __init__(self) -> None:
        self.token = object()
        self.device = torch.device('npu', torch.npu.current_device())
        self.lock = threading.Lock()
        self.last_event = None
        self.failed = False
        self.tilings = {}
        self.inflight = None

    def _check_alive(self, buffer) -> None:
        if buffer.runtime is None:
            raise RuntimeError('EPBuffer has been destroyed')
        if self.failed:
            raise RuntimeError('A low-latency launch failed; restart the distributed job')
        if torch.npu.current_device() != self.device.index:
            raise ValueError('Use the NPU on which this EPBuffer was constructed')

    def _check_tensor(self, value, shape, dtypes, name) -> None:
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype not in dtypes or value.device != self.device or not value.is_contiguous()):
            raise ValueError(f'{name} must be a contiguous {shape} tensor on {self.device}, dtype {dtypes}')

    def _get_tiling(self, operation, shape, dtype, capacity) -> torch.Tensor:
        key = (operation, *shape, dtype, capacity)
        if key not in self.tilings:
            data = getattr(_C, f'low_latency_{operation}_tiling')(*shape, dtype, capacity)
            source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            self.tilings[key] = source.to(device=self.device)
        return self.tilings[key]

    def _begin(self, previous_event) -> None:
        # The device owns persistent per-AIV ping-pong state. Serialize launches
        # even if the application changes its current stream between calls.
        if self.last_event is not None:
            self.last_event.current_stream_wait()
        if previous_event is not None:
            previous_event.current_stream_wait()

    def _retain(self, tensors) -> None:
        retained = tuple(t for t in tensors.values() if isinstance(t, torch.Tensor))
        stream = torch.npu.current_stream()
        for tensor in retained:
            tensor.record_stream(stream)
        self.inflight = retained

    def _finish(self) -> EventOverlap:
        self.last_event = EventHandle()
        event = EventOverlap(self.last_event, self.inflight)
        self.inflight = None
        return event

    @staticmethod
    def _check_schedule(num_sms, num_qps, allocate_on_comm_stream, defer_epilogue) -> None:
        if num_sms not in (0, 32) or num_qps not in (None, 0):
            raise NotImplementedError('Low latency uses exactly 32 AI cores / 64 AIVs and automatic QPs')
        if allocate_on_comm_stream or defer_epilogue:
            raise NotImplementedError('Low latency runs the complete operation on the current stream')

    def dispatch(self, buffer, x, topk_idx, topk_weights, cumulative_local_expert_recv_stats,
                 num_experts, num_max_tokens_per_rank, expert_alignment, num_sms, num_qps,
                 previous_event, async_with_compute_stream, allocate_on_comm_stream, handle,
                 do_handle_copy, do_cpu_sync, do_expand, do_zero_padding,
                 use_tma_aligned_col_major_sf, defer_epilogue):
        from .ep import EPHandle

        self._check_alive(buffer)
        self._check_schedule(num_sms, num_qps, allocate_on_comm_stream, defer_epilogue)
        if (handle is not None or cumulative_local_expert_recv_stats is not None
                or do_handle_copy or use_tma_aligned_col_major_sf or expert_alignment not in (None, 1)):
            raise NotImplementedError('Low latency requires a fresh handle, alignment 1, row-major scales and no stats/copy')
        if not do_expand:
            raise NotImplementedError('Low latency requires do_expand=True')
        values, scales = x if isinstance(x, tuple) and len(x) == 2 else (x, None)
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError('x must be [B,H] or (FP8 [B,H], scales [B,ceil(H/128)])')
        if not isinstance(topk_idx, torch.Tensor) or topk_idx.ndim != 2:
            raise ValueError('topk_idx must be an int64 [B,K] tensor')
        num_tokens, hidden = values.shape
        num_topk = topk_idx.shape[1]
        capacity = buffer.num_max_tokens_per_rank if num_max_tokens_per_rank is None else num_max_tokens_per_rank
        if type(capacity) is not int or not 0 < num_tokens <= capacity <= 256:
            raise ValueError('Low latency requires 0 < B <= num_max_tokens_per_rank <= 256')
        if type(num_experts) is not int or num_experts <= 0:
            raise ValueError('num_experts must be a positive integer')
        self._check_tensor(values, (num_tokens, hidden), (torch.bfloat16, torch.float16, torch.float8_e4m3fn), 'x')
        self._check_tensor(topk_idx, (num_tokens, num_topk), (torch.int64,), 'topk_idx')
        self._check_tensor(topk_weights, (num_tokens, num_topk), (torch.float32,), 'topk_weights')
        if values.dtype == torch.float8_e4m3fn:
            self._check_tensor(scales, (num_tokens, (hidden + 127) // 128), (torch.int32, torch.float32), 'scales')
        elif scales is not None:
            raise ValueError('scales are only accepted with FP8 inputs')
        dtype = str(values.dtype).removeprefix('torch.')
        shape = (buffer.num_ranks, num_tokens, hidden, num_topk, num_experts)
        plan = _C.low_latency_layout(*shape, dtype)
        if plan is None:
            raise ValueError('Low-latency shape exceeds topology, workspace, UB or 10000-row combine capacity')
        # Admit a complete dispatch/combine pair before entering communication.
        _C.low_latency_combine_tiling(*shape, 'float16' if dtype == 'float16' else 'bfloat16', capacity)
        num_capacity_rows = plan['capacity_rows']
        with self.lock:
            self._check_alive(buffer)
            self._begin(previous_event)
            alloc = torch.zeros if do_zero_padding else torch.empty
            tensors = dict(
                input=values, indices=topk_idx.to(torch.int32), weights=topk_weights, scales=scales,
                output=alloc((num_capacity_rows, hidden), dtype=values.dtype, device=self.device),
                output_scales=(alloc((num_capacity_rows, (hidden + 127) // 128), dtype=scales.dtype, device=self.device)
                               if scales is not None else None),
                expert_recv_counts=torch.empty((num_experts // buffer.num_ranks,), dtype=torch.int64, device=self.device),
                send_counts=torch.empty((num_experts,), dtype=torch.int32, device=self.device),
                token_type=torch.empty((num_capacity_rows,), dtype=torch.int32, device=self.device),
                destination_index=torch.empty((num_capacity_rows, 3), dtype=torch.int32, device=self.device),
                relay_read_index=torch.empty((num_capacity_rows, num_topk, 2), dtype=torch.int32, device=self.device),
                source_mask=torch.empty((num_tokens,), dtype=torch.int32, device=self.device),
                source_weights=torch.empty((num_tokens, num_topk), dtype=torch.float32, device=self.device),
                tiling=self._get_tiling('dispatch', shape, dtype, capacity),
            )
            self._retain(tensors)
            try:
                _C.dispatch_low_latency(
                    buffer.runtime, tensors, num_tokens, hidden, num_topk, num_experts, num_capacity_rows, dtype)
                counts = tensors['expert_recv_counts']
                count_list = counts.cpu().tolist() if do_cpu_sync is not False else []
                num_expanded_tokens = sum(count_list) if do_cpu_sync is not False else num_capacity_rows
                out = tensors['output'][:num_expanded_tokens]
                out_scales = tensors['output_scales'][:num_expanded_tokens] if scales is not None else None
                result = EPHandle(
                    do_expand=True, num_experts=num_experts, expert_alignment=1,
                    num_max_tokens_per_rank=capacity, num_sms=32, topk_idx=topk_idx,
                    num_recv_tokens=None, num_expanded_tokens=num_expanded_tokens,
                    num_recv_tokens_per_expert_list=count_list,
                    psum_num_recv_tokens_per_rank=None, psum_num_recv_tokens_per_expert=None,
                    num_unaligned_recv_tokens_per_expert=None, recv_src_metadata=None,
                    dst_buffer_slot_idx=None, dst_gsge_idx=None)
                result.is_low_latency = True
                result._buffer_token = self.token
                result.num_tokens, result.hidden, result.num_topk = num_tokens, hidden, num_topk
                result.expert_recv_counts = counts
                for name in ('send_counts', 'token_type', 'destination_index', 'relay_read_index',
                             'source_mask', 'source_weights'):
                    setattr(result, name, tensors[name])
                event = self._finish()
            except Exception:
                self.failed = True
                # Keep references and existing allocator stream records; do not
                # issue another device operation that could mask the failure.
                raise
        return ((out, out_scales) if scales is not None else out), None, None, result, event

    def combine(self, buffer, x, handle, topk_weights, bias, num_sms, num_qps,
                previous_event, async_with_compute_stream, allocate_on_comm_stream, defer_epilogue):
        self._check_alive(buffer)
        self._check_schedule(num_sms, num_qps, allocate_on_comm_stream, defer_epilogue)
        if not getattr(handle, 'is_low_latency', False) or getattr(handle, '_buffer_token', None) is not self.token:
            raise ValueError('Low-latency combine requires a dispatch handle from this buffer')
        if topk_weights is not None or bias is not None:
            raise NotImplementedError('Legacy low-latency combine has no weight-return side channel or bias')
        self._check_tensor(x, (handle.num_expanded_tokens, handle.hidden), (torch.bfloat16, torch.float16), 'expert output')
        num_tokens, hidden = handle.num_tokens, handle.hidden
        num_topk, num_experts = handle.num_topk, handle.num_experts
        shape = (buffer.num_ranks, num_tokens, hidden, num_topk, num_experts)
        dtype = str(x.dtype).removeprefix('torch.')
        with self.lock:
            self._check_alive(buffer)
            self._begin(previous_event)
            tensors = {name: getattr(handle, name) for name in (
                'send_counts', 'token_type', 'destination_index', 'relay_read_index', 'source_mask')}
            tensors.update(
                input=x, output=torch.empty((num_tokens, hidden), dtype=x.dtype, device=self.device),
                tiling=self._get_tiling('combine', shape, dtype, handle.num_max_tokens_per_rank))
            self._retain(tensors)
            try:
                _C.combine_low_latency(buffer.runtime, tensors, num_tokens, hidden, num_topk, num_experts, dtype)
                event = self._finish()
            except Exception:
                self.failed = True
                raise
        return tensors['output'], None, event
