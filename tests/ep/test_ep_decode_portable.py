"""EP8 BF16 decode smoke test for the NVIDIA and Ascend DeepEP V2.5 packages.

Run against each installed package with the same command:
    python tests/ep/test_ep_decode_portable.py --num-processes 8
WORLD_SIZE and RANK denote the node count and node rank, as in test_ep.py.
"""

import argparse
import os


def worker(local_rank, num_processes):
    import torch
    import torch.distributed as dist
    from deep_ep import EPBuffer, topk_idx_t
    from deep_ep.utils.envs import init_dist

    # Each package's test helper selects the backend and sets the default device.
    rank, world, group = init_dist(local_rank, num_processes)
    capacity, hidden, num_topk, num_experts, alignment = 2, 7168, 8, 128, 128
    batch = max(1, capacity - rank)
    buffer = EPBuffer(group, num_max_tokens_per_rank=capacity, hidden=hidden, num_topk=num_topk,
                      allow_hybrid_mode=False, explicitly_destroy=True)

    tokens = torch.arange(batch)[:, None]
    slots = torch.arange(num_topk)[None, :]
    x = (1 + rank * capacity + tokens + torch.arange(hidden) % 5).to(torch.bfloat16)
    # Each source uses its own local expert on each of the eight destinations.
    routes = (slots * (num_experts // world) + rank).to(topk_idx_t).repeat(batch, 1)
    weights = ((rank * capacity + tokens) * num_topk + slots + 1).float() / 128

    ready = buffer.dispatch(x, topk_idx=routes, topk_weights=weights, num_experts=num_experts,
                            expert_alignment=alignment, num_sms=32, do_expand=True, do_zero_padding=True,
                            do_cpu_sync=False, async_with_compute_stream=True, defer_epilogue=True)
    recv, recv_indices, recv_weights, handle = ready.current_stream_wait()
    # Rank-specific expert outputs make the sum depend on the destinations visited.
    done = buffer.combine(recv + rank, handle, topk_weights=recv_weights,
                          async_with_compute_stream=True, defer_epilogue=True)
    combined, returned_weights = done.current_stream_wait()

    # Finish all collective calls before reporting rank-local comparison failures.
    counts = handle.num_unaligned_recv_tokens_per_expert.cpu().tolist()
    result, weight_result = combined.cpu(), returned_weights.cpu()
    buffer.destroy()
    dist.destroy_process_group()

    assert recv_indices is None
    expected_counts = [max(1, capacity - source) for source in range(world)]
    assert counts == expected_counts + [0] * (num_experts // world - world), counts
    torch.testing.assert_close(result, x.cpu() * num_topk + sum(range(world)), atol=0, rtol=0)
    torch.testing.assert_close(weight_result, weights.cpu(), atol=0, rtol=0)
    print(f'PASS EP8 decode rank {rank} on {x.device}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8, help='devices per node')
    args = parser.parse_args()
    if args.num_processes <= 0 or int(os.getenv('WORLD_SIZE', '1')) * args.num_processes != 8:
        parser.error('This case requires eight ranks across all nodes.')
    from torch.multiprocessing import spawn

    spawn(worker, args=(args.num_processes,), nprocs=args.num_processes)
