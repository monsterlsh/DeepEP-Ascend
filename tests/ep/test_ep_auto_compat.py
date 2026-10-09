"""Native EPBuffer compatibility across automatic transport decisions.

Run on every node using the node-count WORLD_SIZE / node-index RANK convention.
Do not wrap this mp.spawn launcher in torchrun. To compare the same contracts
with the ordinary transport, set EP_DISABLE_LOW_LATENCY_TRANSPORT=1 on all nodes.
This checks results, not performance or which kernel the device actually ran.
"""

import argparse
from datetime import timedelta
import os

__test__ = False


def launch(buffer, operation, deferred, **kwargs):
    result = getattr(buffer, operation)(defer_epilogue=deferred, **kwargs)
    if deferred:
        event = result
        values = event.wait()
        # A deferred epilogue is consumed exactly once.
        assert event.wait() is None
        return (*values, event)
    result[-1].wait()
    return result


def check_dispatch(recv, weights, handle, routes, rank, world, alignment,
                   deterministic, *, weight_scale=1, payload_scale=1):
    import torch
    from low_latency_reference import expert_entries, payload

    entries = expert_entries(routes, 128, world, rank)
    counts = [len(rows) for rows in entries]
    assert not handle.is_low_latency, 'Automatic transport must retain the native EPHandle contract'
    assert handle.num_unaligned_recv_tokens_per_expert.cpu().tolist() == counts
    assert weights is not None
    valid_aligned = sum((count + alignment - 1) // alignment * alignment for count in counts)
    recv_cpu = recv[:valid_aligned].cpu()
    weights_cpu = weights[:valid_aligned].cpu()
    offset = 0
    expected_ends = []
    for local, rows in enumerate(entries):
        count = len(rows)
        expected_ends.append(offset + count)
        if count:
            block = recv_cpu[offset:offset + count]
            # All row identities are exact integers even after payload scaling.
            ids = (block[:, :4].float() / payload_scale).long()
            sources = ids[:, 0] * 16 + ids[:, 1]
            tokens = ids[:, 2] * 16 + ids[:, 3]
            keys = sources * handle.num_max_tokens_per_rank + tokens
            order = keys.argsort()
            expected_sources, expected_tokens = zip(*rows)
            expected = payload(expected_sources, expected_tokens, 7168, torch.bfloat16) * payload_scale
            torch.testing.assert_close(block[order], expected, atol=0, rtol=0)
            if deterministic:
                assert torch.equal(keys, keys.sort().values)
        next_offset = offset + (count + alignment - 1) // alignment * alignment
        assert torch.count_nonzero(recv_cpu[offset + count:next_offset]).item() == 0
        assert torch.count_nonzero(weights_cpu[offset + count:next_offset]).item() == 0
        offset = next_offset
    assert handle.psum_num_recv_tokens_per_expert.cpu().tolist() == expected_ends
    # Validate reverse-routing slots, deduplicated counts and capacity-strided IDs.
    expected_counts = [sum(any(rank * (128 // world) <= e < (rank + 1) * (128 // world)
                               for e in selected) for selected in source) for source in routes]
    psum, accumulated = [], 0
    for count in expected_counts:
        accumulated += count
        psum.append(accumulated)
    assert handle.psum_num_recv_tokens_per_rank.cpu().tolist() == psum
    metadata = handle.recv_src_metadata[:accumulated].cpu().tolist()
    visited = set()
    seen_tokens = set()
    for row in metadata:
        source, token = divmod(row[-2], handle.num_max_tokens_per_rank)
        assert (source, token) not in seen_tokens
        seen_tokens.add((source, token))
        valid = []
        for topk_slot, slot in enumerate(row[:-2]):
            expert = routes[source][token][topk_slot]
            belongs = rank * (128 // world) <= expert < (rank + 1) * (128 // world)
            assert (slot >= 0) == belongs
            if belongs:
                assert slot not in visited
                visited.add(slot)
                valid.append(topk_slot)
                torch.testing.assert_close(recv_cpu[slot],
                    payload([source], [token], 7168, torch.bfloat16)[0] * payload_scale, atol=0, rtol=0)
                assert weights_cpu[slot].item() == (topk_slot + 1) / 32 * weight_scale
        assert row[-1] == max(valid)
    assert len(visited) == sum(counts)


def run_case(buffer, bs, seed, mode, route, rank, world, retained, *, edge_routes=False):
    import torch
    from low_latency_reference import make_routes, expert_entries, payload, prepare_experts, expected_combine
    from test_low_latency import agree

    alignment, deferred, sync = mode
    routes = make_routes(world, bs, 8, 128, seed)
    if route == 'hotspot':
        routes = [[list(range(8)) for _ in range(bs)] for _ in range(world)]
    if edge_routes:
        # Include empty ranks and both sides of the automatic cutoff in one
        # collective. Keep the normal API's masked and repeated expert slots.
        for source, rows in enumerate(routes):
            sizes = (0, 2, 128, 32)
            routes[source] = rows[:sizes[source % len(sizes)]]
            for token, selected in enumerate(routes[source]):
                selected[-1] = -1
                selected[1] = selected[0]
                if token % 3 == 0:
                    selected[-2] = -1
        bs = len(routes[rank])
    counts = [len(rows) for rows in expert_entries(routes, 128, world, rank)]
    x = payload([rank] * bs, range(bs), 7168, torch.bfloat16).npu()
    indices = torch.tensor(routes[rank], dtype=torch.int64, device='npu').reshape(bs, 8)
    weights = (torch.arange(1, 9, dtype=torch.float32, device='npu') / 32).repeat(bs, 1)
    weights.masked_fill_(indices < 0, 0)
    dispatch_args = dict(x=x, topk_idx=indices, topk_weights=weights, num_experts=128,
                         num_max_tokens_per_rank=buffer.num_max_tokens_per_rank,
                         expert_alignment=alignment, num_sms=0, do_expand=True,
                         do_cpu_sync=sync, do_zero_padding=True, async_with_compute_stream=True)
    recv, recv_idx, recv_weights, handle, _ = launch(buffer, 'dispatch', deferred, **dispatch_args)
    torch.npu.synchronize()
    agree(lambda: check_dispatch(recv, recv_weights, handle, routes, rank, world, alignment,
                                 buffer.deterministic))
    agree(lambda: _check_no_indices(recv_idx))
    if buffer.deterministic:
        again, _, again_weights, _, _ = launch(buffer, 'dispatch', deferred, **dispatch_args)
        torch.npu.synchronize()
        valid_aligned = sum((c + alignment - 1) // alignment * alignment for c in counts)
        agree(lambda: torch.testing.assert_close(again[:valid_aligned], recv[:valid_aligned], atol=0, rtol=0))
        agree(lambda: torch.testing.assert_close(again_weights[:valid_aligned], recv_weights[:valid_aligned],
                                                 atol=0, rtol=0))

    unweighted_expert = prepare_experts(recv, counts, rank, 128, world, torch.bfloat16,
                                       expert_alignment=alignment)
    expected_unweighted = expected_combine(routes, rank, 7168, torch.bfloat16,
                                            reduce_in_local=True, experts=128)
    for weighted in (False, True):
        expert = unweighted_expert.clone()
        if weighted:
            start = 0
            for count in counts:
                expert[start:start + count] = (expert[start:start + count].float()
                    * recv_weights[start:start + count, None]).to(torch.bfloat16)
                start += (count + alignment - 1) // alignment * alignment
        saved = expert.clone()
        for num_bias in (0, 1, 2):
            bias = None if not num_bias else torch.full((bs, 7168), .25, dtype=torch.bfloat16, device='npu')
            if num_bias == 2:
                bias = (bias, torch.full_like(bias, -.125))
            expected_biased = expected_combine(routes, rank, 7168, torch.bfloat16, weighted=weighted,
                reduce_in_local=True, experts=128, bias=0 if not num_bias else .25 if num_bias == 1 else .125)
            for send_weights in (False, True):
                output, output_weights, _ = launch(buffer, 'combine', deferred,
                    x=expert, handle=handle, topk_weights=recv_weights if send_weights else None,
                    bias=bias, num_sms=0, async_with_compute_stream=True)
                torch.npu.synchronize()
                agree(lambda: torch.testing.assert_close(output.float().cpu(), expected_biased.float(),
                                                           atol=.125, rtol=.02))
                agree(lambda: torch.testing.assert_close(expert, saved, atol=0, rtol=0))
                if send_weights:
                    agree(lambda: torch.testing.assert_close(output_weights, weights, atol=0, rtol=0))
                else:
                    agree(lambda: _check_no_indices(output_weights))

    # Change both data and metadata while reusing the original layout. A cached
    # dispatch must not overwrite its stored slots or retain the old weights.
    metadata_fields = ('psum_num_recv_tokens_per_rank', 'psum_num_recv_tokens_per_expert',
                       'num_unaligned_recv_tokens_per_expert', 'recv_src_metadata',
                       'dst_buffer_slot_idx', 'dst_gsge_idx')
    saved_metadata = [getattr(handle, name).clone() for name in metadata_fields]
    cached, cached_idx, cached_weights, cached_handle, _ = launch(buffer, 'dispatch', deferred,
        x=x * 2, handle=handle, topk_weights=weights * 3, do_expand=True, do_cpu_sync=False,
        do_zero_padding=True, async_with_compute_stream=True)
    torch.npu.synchronize()
    agree(lambda: _check_cached_handle(cached_handle, handle, cached_idx))
    agree(lambda: check_dispatch(cached, cached_weights, cached_handle, routes, rank, world, alignment,
                                 buffer.deterministic, weight_scale=3, payload_scale=2))
    for name, saved in zip(metadata_fields, saved_metadata):
        agree(lambda: torch.testing.assert_close(getattr(handle, name), saved, atol=0, rtol=0))
    # Reuse the same unweighted expert data: changing route weight metadata must
    # not change combined x, and returned weights must reflect the cached call.
    output, output_weights, _ = launch(buffer, 'combine', deferred,
        x=unweighted_expert, handle=cached_handle, topk_weights=cached_weights,
        num_sms=0, async_with_compute_stream=True)
    torch.npu.synchronize()
    agree(lambda: torch.testing.assert_close(output.float().cpu(), expected_unweighted.float(), atol=.125, rtol=.02))
    agree(lambda: torch.testing.assert_close(output_weights, weights * 3, atol=0, rtol=0))

    if retained is not None:
        old_expert, old_handle, old_expected = retained
        output, output_weights, _ = launch(buffer, 'combine', deferred,
            x=old_expert, handle=old_handle, num_sms=0, async_with_compute_stream=True)
        torch.npu.synchronize()
        agree(lambda: torch.testing.assert_close(output.float().cpu(), old_expected.float(), atol=.125, rtol=.02))
        agree(lambda: _check_no_indices(output_weights))
    else:
        retained = unweighted_expert, handle, expected_unweighted
    if rank == 0:
        print(f'PASS EP{world} BS={bs} {route=} {alignment=} {deferred=} {sync=} '
              f'deterministic={buffer.deterministic} {edge_routes=}', flush=True)
    return retained


def _check_no_indices(value):
    assert value is None


def _check_cached_handle(actual, expected, indices):
    assert actual is expected and indices is None


def worker(local_rank, args):
    import torch
    import torch_npu
    import torch.distributed as dist
    import deep_ep

    torch.npu.set_device(local_rank)
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    rank = int(os.getenv('RANK', '0')) * args.num_processes + local_rank
    dist.init_process_group('hccl', rank=rank, world_size=world,
        device_id=torch.device(f'npu:{local_rank}'),
        init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8361')}",
        timeout=timedelta(seconds=600))
    with torch.inference_mode():
        for deterministic in (False, True):
            buffer = deep_ep.EPBuffer(dist.group.WORLD, num_max_tokens_per_rank=max(args.bs),
                hidden=7168, num_topk=8, deterministic=deterministic,
                explicitly_destroy=True, use_huge1g_page=not args.no_huge1g_page)
            # Ordinary EPBuffer owns one native arena and native handles even
            # when its dispatch/combine kernels choose the direct peer path.
            assert not buffer.low_latency_mode
            retained = None
            for mode in ((1, False, False), (16, False, True), (1, True, True), (16, True, False)):
                for route in args.route:
                    for step, bs in enumerate(args.bs):
                        retained = run_case(buffer, bs, step, mode, route, rank, world, retained)
            if max(args.bs) >= 128:
                retained = run_case(buffer, 128, 23, (16, True, False), 'random', rank, world,
                                    retained, edge_routes=True)
            buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--bs', nargs='+', type=int, default=[2, 32, 64, 128, 256, 2])
    parser.add_argument('--route', nargs='+', choices=['random', 'hotspot'], default=['random', 'hotspot'])
    parser.add_argument('--no-huge1g-page', action='store_true')
    args = parser.parse_args()
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    if world not in (8, 16, 32, 64, 128) or min(args.bs) <= 0 or args.num_processes <= 0:
        parser.error('Require EP8/16/32/64/128, positive BS and positive local process count')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.num_processes, join=True)
