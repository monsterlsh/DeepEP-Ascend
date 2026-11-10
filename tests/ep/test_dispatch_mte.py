"""Supplemental dispatch boundary/reuse correctness; use test_ep.py for performance."""
import argparse

from test_ep import deep_ep, torch, dist, init_dist, dist_print, launch, ref_dispatch, align
from deep_ep.utils.gate import get_unbalanced_scores


def check_dispatch(result, reference, capacity, exact_counts, send_weights=True):
    recv_x, recv_idx, recv_weights, handle, _ = result
    torch.npu.synchronize()
    ref_x, ref_idx, ref_weights, ref_sources, ref_counts = reference
    num_ranks = dist.get_world_size()
    assert ref_counts.numel() == num_ranks
    count = int(handle.psum_num_recv_tokens_per_rank[-1].item())
    assert count == ref_sources.numel() and recv_idx is None
    assert handle.num_recv_tokens == (count if exact_counts else num_ranks * capacity)
    assert torch.equal(handle.psum_num_recv_tokens_per_rank.cpu(), ref_counts.cpu().cumsum(0))
    metadata = handle.recv_src_metadata[:count]
    order = metadata[:, -2].argsort()
    ref_order = ref_sources.argsort()
    assert torch.equal(metadata[order, -2], ref_sources[ref_order])
    slots = metadata[order, :-2]
    ref_idx = ref_idx[ref_order]
    valid = slots >= 0
    assert torch.equal(valid, ref_idx >= 0)
    master = torch.where(valid, torch.arange(8, device=metadata.device), -1).amax(dim=1)
    assert torch.equal(metadata[order, -1], master)
    slot_ids = slots[valid].long()
    expanded = align(int(handle.psum_num_recv_tokens_per_expert[-1].item()), 128)
    if slot_ids.numel():
        assert slot_ids.min().item() >= 0 and slot_ids.max().item() < expanded
        assert slot_ids.unique().numel() == slot_ids.numel()
    assert torch.equal(recv_x[slot_ids], ref_x[ref_order].unsqueeze(1).expand(-1, 8, -1)[valid])
    if send_weights:
        assert recv_weights is not None
        assert torch.equal(recv_weights[slot_ids], ref_weights[ref_order][valid])
    else:
        assert recv_weights is None
    padding = torch.ones(expanded, dtype=torch.bool, device=recv_x.device)
    padding[slot_ids] = False
    assert torch.count_nonzero(recv_x[:expanded][padding]).item() == 0
    if send_weights:
        assert torch.count_nonzero(recv_weights[:expanded][padding]).item() == 0
    psum = [0] + handle.psum_num_recv_tokens_per_expert.tolist()
    for expert in range(128 // num_ranks):
        expected = (ref_idx == expert).sum().item()
        assert handle.num_unaligned_recv_tokens_per_expert[expert].item() == expected
        assert psum[expert + 1] - align(psum[expert], 128) == expected
        assert torch.equal(slots[ref_idx == expert].sort().values,
                           torch.arange(align(psum[expert], 128), psum[expert + 1], device=slots.device))
        if exact_counts:
            assert handle.num_recv_tokens_per_expert_list[expert] == align(expected, 128)
    return handle


@torch.inference_mode()
def test_loop(local_rank, num_local_ranks, args):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    capacity_limits = {8: (64, 64), 16: (64, 64), 32: (48, 32), 64: (32, 16)}
    assert world_size in capacity_limits, 'Run this regression with EP8, EP16, EP32 or EP64'
    fresh_limit, cached_limit = capacity_limits[world_size]
    buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=257, hidden=7168, num_topk=8,
                             allow_hybrid_mode=False, explicitly_destroy=True)
    # B is a rank-varying input bound; M is the independent per-call capacity.
    # The shared capacity selects transport even when local token counts differ.
    cases = [(fresh_limit, fresh_limit, False), (fresh_limit + 1, fresh_limit + 1, False),
             (1, fresh_limit, False), (1, fresh_limit + 1, False),
             (cached_limit, cached_limit, False), (cached_limit + 1, cached_limit + 1, False),
             (1, cached_limit, True)]
    for batch, capacity, cpu_sync in dict.fromkeys(cases):
        tokens = max(1, batch - rank)
        x = torch.randn((tokens, 7168), dtype=torch.bfloat16, device='npu')
        scores = get_unbalanced_scores(tokens, 128, world_size, 8, 1.0, False)
        weights, indices = torch.topk(scores, 8, dim=-1, sorted=False)
        indices = indices.to(torch.int64)
        reference = ref_dispatch(x, indices, weights, capacity, 128)
        params = dict(num_experts=128, num_max_tokens_per_rank=capacity, expert_alignment=128,
                      num_sms=32, async_with_compute_stream=True, do_expand=True, do_zero_padding=True)
        result = launch(buffer, 'dispatch', args.defer_epilogue, x=x, topk_idx=indices,
                        topk_weights=weights, do_cpu_sync=cpu_sync, **params)
        handle = check_dispatch(result, reference, capacity, cpu_sync)
        assert handle.num_sms == 32 and handle.num_max_tokens_per_rank == capacity
        saved = [tensor.clone() for tensor in (
            handle.psum_num_recv_tokens_per_rank, handle.psum_num_recv_tokens_per_expert,
            handle.num_unaligned_recv_tokens_per_expert, handle.recv_src_metadata,
            handle.dst_buffer_slot_idx, handle.dst_gsge_idx)]

        # Cached calls retain the initial allocation/count contract, including CPU-sync handles.
        for send_weights in (True, False):
            updated_x = torch.randn_like(x) * (8 if send_weights else 0.125)
            updated_weights = torch.randn_like(weights)
            reference = ref_dispatch(updated_x, indices, updated_weights, capacity, 128)
            result = launch(buffer, 'dispatch', args.defer_epilogue, x=updated_x, handle=handle,
                            topk_weights=updated_weights if send_weights else None,
                            do_cpu_sync=False, **params)
            assert check_dispatch(result, reference, capacity, cpu_sync, send_weights) is handle
            for tensor, snapshot in zip((
                    handle.psum_num_recv_tokens_per_rank, handle.psum_num_recv_tokens_per_expert,
                    handle.num_unaligned_recv_tokens_per_expert, handle.recv_src_metadata,
                    handle.dst_buffer_slot_idx, handle.dst_gsge_idx), saved):
                assert torch.equal(tensor, snapshot)
        dist_print(f'Passed dispatch/reuse correctness: EP={world_size}, B={batch}, M={capacity}, '
                   f'cpu_sync={cpu_sync}, deferred={args.defer_epilogue}', once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--defer-epilogue', type=int, choices=(0, 1), default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
