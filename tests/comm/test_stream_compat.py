"""Additional device tests for stream ordering and EP tensor lifetimes."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch_npu  # Register the NPU backend even when backend autoload is disabled.

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from test_stream import test_bucket, test_ep, test_events

# Both standalone NPU launchers exercise the same candidate/official environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ep'))
from compat_config import configure_backend
from compat_trace import TraceCollector
from ep_reference import agree

__test__ = False  # Standalone multi-rank launcher.


def mte_routes(rank_idx, num_ranks, num_tokens):
    # Advance the local expert when eight slots wrap around a smaller EP group.
    num_local_experts = 128 // num_ranks
    return [[((rank_idx + token + k) % num_ranks) * num_local_experts
             + ((rank_idx + token) // num_ranks + k // num_ranks) % num_local_experts
             for k in range(8)] for token in range(num_tokens)]


def wait_ep_result(result, defer_epilogue, stream, num_outputs):
    if defer_epilogue:
        event = result
        outputs = event.wait()
    else:
        assert isinstance(result, tuple) and len(result) == num_outputs + 1
        *outputs, event = result
        assert event.wait() is None
    assert len(outputs) == num_outputs
    assert event.wait() is None, 'The deferred epilogue must only run once'
    assert torch.npu.current_stream() == stream
    return outputs


def allocation_pressure(shape, dtype):
    # Keep both allocations alive through the wait; use the input's allocation
    # stream so freeing a captured input too early can result in visible reuse.
    return [torch.full(shape, -100 - i, dtype=dtype, device='npu') for i in range(2)]


def check_mte_dispatch(recv_x, recv_idx, recv_weights, handle,
                       rank_idx, num_ranks, num_tokens, capacity, iteration):
    assert recv_idx is None
    assert recv_x.dtype == torch.bfloat16 and recv_x.shape[1] == 7168
    assert recv_weights is not None and recv_weights.dtype == torch.float32
    assert recv_weights.shape == (recv_x.shape[0],)
    assert torch.equal(handle.topk_idx.cpu(),
                       torch.tensor(mte_routes(rank_idx, num_ranks, num_tokens),
                                    dtype=torch.int64, device='cpu'))

    expected_sources = {}
    expected_rank_counts = [0] * num_ranks
    expected_expert_counts = [0] * (128 // num_ranks)
    for source_rank in range(num_ranks):
        for token, experts in enumerate(mte_routes(source_rank, num_ranks, num_tokens)):
            for k, expert in enumerate(experts):
                if expert // (128 // num_ranks) == rank_idx:
                    global_idx = source_rank * capacity + token
                    if global_idx not in expected_sources:
                        expected_sources[global_idx] = {}
                        expected_rank_counts[source_rank] += 1
                    local_expert = expert % (128 // num_ranks)
                    expected_sources[global_idx][k] = local_expert
                    expected_expert_counts[local_expert] += 1

    assert sum(expected_expert_counts) == 8 * num_tokens
    assert torch.equal(handle.psum_num_recv_tokens_per_rank.cpu(),
                       torch.tensor(expected_rank_counts, device='cpu').cumsum(0))
    assert torch.equal(handle.num_unaligned_recv_tokens_per_expert.cpu(),
                       torch.tensor(expected_expert_counts, device='cpu'))
    alignment = handle.expert_alignment
    starts, ends, offset = [], [], 0
    for count in expected_expert_counts:
        starts.append(offset)
        ends.append(offset + count)
        offset += (count + alignment - 1) // alignment * alignment
    assert handle.psum_num_recv_tokens_per_expert.cpu().tolist() == ends

    # do_cpu_sync=False permits padded output/metadata allocations. Only the
    # prefix sums and metadata identify live rows; padding is not test data.
    metadata = handle.recv_src_metadata[:len(expected_sources)].cpu().tolist()
    expected_count = sum(expected_expert_counts)
    expected_x = torch.empty((expected_count, 7168), dtype=torch.bfloat16, device='cpu')
    expected_weights = torch.empty(expected_count, dtype=torch.float32, device='cpu')
    seen_sources, seen_slots = set(), set()
    slots_by_row = []
    for row in metadata:
        slots, global_idx = row[:8], row[-2]
        assert global_idx in expected_sources and global_idx not in seen_sources
        source_rank, token = divmod(global_idx, capacity)
        local_experts = expected_sources[global_idx]
        assert [k for k, slot in enumerate(slots) if slot >= 0] == list(local_experts)
        assert row[-1] == max(local_experts)
        seen_sources.add(global_idx)
        for k, local_expert in local_experts.items():
            slot = slots[k]
            assert starts[local_expert] <= slot < ends[local_expert] and slot not in seen_slots
            index = len(slots_by_row)
            expected_x[index].fill_(source_rank + 10 * iteration)
            expected_x[index, 0] = token
            expected_weights[index] = k + 1
            slots_by_row.append(slot)
            seen_slots.add(slot)
    assert seen_sources == set(expected_sources)
    slots = torch.tensor(slots_by_row, dtype=torch.long, device='cpu')
    assert torch.equal(recv_x[:offset].cpu()[slots], expected_x)
    assert torch.equal(recv_weights[:offset].cpu()[slots], expected_weights)


def make_stream_delay(stream, observer):
    sleep = getattr(torch.npu, '_sleep', None)
    if callable(sleep):
        return lambda: sleep(100_000_000)

    # Some torch_npu builds lack _sleep. Preallocate one scratch matrix per
    # stream and check that the other stream can advance while GEMMs are pending.
    with torch.npu.stream(stream):
        matrix = torch.ones((4096, 4096), dtype=torch.bfloat16, device='npu')
        out = torch.empty_like(matrix)
        torch.mm(matrix, matrix, out=out)
    stream.synchronize()

    def enqueue():
        for _ in range(repeats):
            torch.mm(matrix, matrix, out=out)

    for repeats in (4, 16, 64, 256):
        with torch.npu.stream(stream):
            enqueue()
            tail = torch.npu.Event()
            tail.record()
        with torch.npu.stream(observer):
            probe = torch.npu.Event()
            probe.record()
        probe.synchronize()
        pending = not tail.query()
        stream.synchronize()
        if pending:
            return enqueue
    raise RuntimeError('GEMM delay calibration could not establish an asynchronous device window')


def test_ep_mte(group, producer, consumer, num_tokens, capacity, trace_dir):
    rank_idx, num_ranks = group.rank(), group.size()
    hidden, num_topk, num_experts = 7168, 8, 128
    buffer = deep_ep.EPBuffer(
        group, num_max_tokens_per_rank=capacity, hidden=hidden,
        num_topk=num_topk, explicitly_destroy=True)

    profiler = None
    if trace_dir is not None:
        directory = Path(trace_dir) / f'rank-{rank_idx:05d}'
        agree(lambda: directory.mkdir(parents=True, exist_ok=False))
        profiler = TraceCollector(directory, enabled=True, backend='candidate-auto')

    delays = []
    agree(lambda: delays.extend((make_stream_delay(producer, consumer),
                                 make_stream_delay(consumer, producer))))
    producer_delay, consumer_delay = delays

    for defer_epilogue in (False, True):
        # Keep the delayed unprofiled run even when profiling is requested.
        iterations = 3 if profiler is not None and defer_epilogue else 2
        for iteration in range(iterations):
            def capture(op):
                if iteration == 2:
                    return profiler.capture(op, op, expected_transport='mte')
                return nullcontext()

            with capture('dispatch'):
                with torch.npu.stream(producer):
                    x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='npu')
                    topk_idx = torch.tensor(mte_routes(rank_idx, num_ranks, num_tokens),
                                            dtype=torch.int64, device='npu')
                    topk_weights = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='npu')
                    if iteration:
                        producer_delay()
                    x.fill_(rank_idx + 10 * iteration)
                    x[:, 0] = torch.arange(num_tokens, dtype=torch.float32, device='npu')
                    topk_weights.copy_(torch.arange(1, num_topk + 1, dtype=torch.float32, device='npu'))
                    ready = deep_ep.EventHandle()
                with torch.npu.stream(consumer):
                    ready.current_stream_wait()
                    dispatch_result = buffer.dispatch(
                        x, topk_idx=topk_idx, topk_weights=topk_weights,
                        num_experts=num_experts, expert_alignment=128,
                        do_cpu_sync=False, do_zero_padding=True, do_expand=True,
                        async_with_compute_stream=True, defer_epilogue=defer_epilogue)
                    assert torch.npu.current_stream() == consumer
                with torch.npu.stream(producer):
                    # Try to reuse the inputs on their allocating stream while
                    # the JIT issue kernel can still be reading them on consumer.
                    del x, topk_idx, topk_weights
                    dispatch_pressure = allocation_pressure((num_tokens, hidden), torch.bfloat16)
                    weight_pressure = allocation_pressure((num_tokens, num_topk), torch.float32)
                    recv_x, recv_idx, recv_weights, handle = wait_ep_result(
                        dispatch_result, defer_epilogue, producer, 4)
                    agree(lambda: check_mte_dispatch(recv_x, recv_idx, recv_weights, handle,
                                                     rank_idx, num_ranks, num_tokens, capacity, iteration))
                    dispatched = deep_ep.EventHandle()
                    del dispatch_result

            with capture('combine'):
                with torch.npu.stream(consumer):
                    dispatched.current_stream_wait()
                    if iteration:
                        consumer_delay()
                    # Native combine checks the full expanded allocation, including
                    # the padding between experts. Give it independent inputs.
                    expert_x, expert_weights = recv_x.clone(), recv_weights.clone()
                    expert_shape, weight_shape = expert_x.shape, expert_weights.shape
                    ready = deep_ep.EventHandle()
                with torch.npu.stream(producer):
                    ready.current_stream_wait()
                    combine_result = buffer.combine(
                        expert_x, handle=handle, topk_weights=expert_weights,
                        async_with_compute_stream=True, defer_epilogue=defer_epilogue)
                    assert torch.npu.current_stream() == producer
                with torch.npu.stream(consumer):
                    del expert_x, expert_weights
                    combine_pressure = allocation_pressure(expert_shape, torch.bfloat16)
                    combine_weight_pressure = allocation_pressure(weight_shape, torch.float32)
                    combined_x, combined_weights = wait_ep_result(
                        combine_result, defer_epilogue, consumer, 2)
                    snapshot = combined_x.clone()
                    weight_snapshot = combined_weights.clone()
            producer.synchronize()
            consumer.synchronize()
            expected = torch.full((num_tokens, hidden), (rank_idx + 10 * iteration) * num_topk,
                                  dtype=torch.bfloat16, device='cpu')
            expected[:, 0] = torch.arange(num_tokens, device='cpu') * num_topk
            expected_weights = torch.arange(1, num_topk + 1, dtype=torch.float32, device='cpu')
            assert snapshot.shape == expected.shape and snapshot.dtype == expected.dtype
            assert weight_snapshot.shape == (num_tokens, num_topk)
            assert weight_snapshot.dtype == torch.float32
            assert torch.equal(snapshot.cpu(), expected)
            # Combine transports weights as a side channel; it does not multiply
            # the hidden states by them.
            assert torch.equal(weight_snapshot.cpu(), expected_weights.expand(num_tokens, -1))
            del (dispatch_pressure, weight_pressure, combine_pressure, combine_weight_pressure,
                 recv_x, recv_weights, handle, combine_result, combined_x, combined_weights,
                 snapshot, weight_snapshot)

    if profiler is not None:
        agree(profiler.write_manifest)
    buffer.destroy()


def validate_config(args, num_ranks):
    if args.ep_shape in ('mte', 'both'):
        if num_ranks not in (2, 4, 8, 16, 32, 64, 128):
            raise ValueError('MTE shape requires global EP size 2, 4, 8, 16, 32, 64, or 128')
        if not 1 <= args.ep_bs <= args.ep_capacity <= 256:
            raise ValueError('MTE shape requires 1 <= --ep-bs <= --ep-capacity <= 256')
    if args.verify_mte and (args.ep_shape == 'fallback' or args.backend != 'candidate-auto'):
        raise ValueError('--verify-mte requires --backend candidate-auto and --ep-shape mte or both')


def require_device_delay():
    if not callable(getattr(getattr(torch, 'npu', None), '_sleep', None)):
        raise RuntimeError(
            'This stream harness requires callable torch.npu._sleep for an asynchronous device delay. '
            'The installed torch_npu backend does not provide it; use a compatible backend or validate '
            'a device-delay adapter before running this test. CPU sleep is not a substitute.')


@torch.inference_mode()
def test(local_rank, num_local_ranks, args):
    if args.ep_shape != 'mte':
        require_device_delay()
    configure_backend(args.backend)
    _, _, group = init_dist(local_rank, num_local_ranks)
    validate_config(args, group.size())
    original = torch.npu.current_stream()
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    comm_stream = deep_ep.comm.get_comm_stream(None)
    assert comm_stream not in (original, producer, consumer)
    with torch.npu.stream(producer):
        assert deep_ep.comm.get_comm_stream(None) == comm_stream

    previous_barrier = deep_ep._C.set_barrier_in_prologue(args.barrier_in_prologue == 'true')
    try:
        dist_print(f'Stream test: backend={args.backend}, package={deep_ep.__file__}, '
                   f'EP={group.size()}, shape={args.ep_shape}, B={args.ep_bs}, capacity={args.ep_capacity}, '
                   f'barrier_in_prologue={args.barrier_in_prologue}. '
                   'MTE shape eligibility alone does not prove the selected device transport.',
                   once_in_node=True)
        if args.ep_shape != 'mte' or callable(getattr(torch.npu, '_sleep', None)):
            test_events(producer, consumer, comm_stream)
        if not args.skip_bucket:
            test_bucket(group, producer, consumer)
        if args.ep_shape in ('fallback', 'both'):
            test_ep(group, producer)
        if args.ep_shape in ('mte', 'both'):
            test_ep_mte(group, producer, consumer, args.ep_bs, args.ep_capacity, args.verify_mte)
        assert torch.npu.current_stream() == original
        dist_print('Stream ordering and EP lifetime checks passed', once_in_node=True)
    finally:
        deep_ep._C.set_barrier_in_prologue(previous_barrier)
        deep_ep.destroy_all_managed_hccl_comm()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test NPU stream ordering and tensor lifetime')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--ep-shape', choices=('fallback', 'mte', 'both'), default='fallback',
                        help='Keep the original H1024/K1 case, add H7168/K8/E128, or run both')
    parser.add_argument('--ep-bs', type=int, default=16, help='Local token count for the MTE shape')
    parser.add_argument('--ep-capacity', type=int, default=None,
                        help='MTE shape buffer capacity per rank (default: --ep-bs)')
    parser.add_argument('--backend', choices=('official', 'candidate-disabled', 'candidate-auto'),
                        default='official', help='Label the installed build and configure its native MTE opt-out')
    parser.add_argument('--barrier-in-prologue', choices=('false', 'true'), default='false')
    parser.add_argument('--verify-mte', metavar='TRACE_DIR',
                        help='Profile a separate deferred iteration and require MTE for both kernels on every rank')
    parser.add_argument('--skip-bucket', action='store_true', help='Skip the BucketBuffer public-layer smoke test')
    args = parser.parse_args()
    if args.ep_capacity is None:
        args.ep_capacity = args.ep_bs
    if args.num_processes <= 0:
        parser.error('--num-processes must be positive')
    # init_dist interprets WORLD_SIZE as node count and --num-processes as the
    # number of local ranks. Reject unsupported MTE cases before HCCL setup.
    try:
        validate_config(args, int(os.getenv('WORLD_SIZE', '1')) * args.num_processes)
        if args.ep_shape != 'mte':
            require_device_delay()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    configure_backend(args.backend)
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
