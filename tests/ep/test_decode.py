"""NPU regressions for empty ranks, deferred combine and decode payload formats."""

import argparse

import torch


def tokens(rank, count):
    # Include a low FP16 mantissa bit that BF16 would discard.
    rows = torch.arange(count, dtype=torch.float32, device='cpu')[:, None]
    columns = torch.arange(256, dtype=torch.float32, device='cpu')[None, :]
    return (1 + (rank % 8) / 16 + rows / 4 + (columns % 4) / 8 + 1 / 1024).half()


def scales(rank, dtype):
    offsets = torch.arange(4, dtype=torch.int32, device='cpu').reshape(2, 2)
    if dtype == torch.float32:
        return (0x3F000001 + rank * 65536 + offsets).view(dtype)
    # Four finite E8M0 exponent bytes per word.
    return 0x74757677 + ((rank + offsets) % 8) * 0x01010101


@torch.inference_mode()
def worker(local_rank, num_processes):
    import torch.distributed as dist
    import torch_npu  # Register the NPU backend when autoload is disabled.
    from deep_ep import EPBuffer
    from deep_ep.utils.envs import init_dist
    from test_ep import launch

    rank, world, group = init_dist(local_rank, num_processes)
    buffer = EPBuffer(group, num_max_tokens_per_rank=17, hidden=256, num_topk=3, explicitly_destroy=True)
    peer = (rank + 1) % world
    source = (rank - 1) % world
    # Both rows go to the next rank, one to each local expert, in expert order.
    routes = 2 * peer + torch.arange(2, device='npu', dtype=torch.int64)[:, None]
    options = dict(num_experts=2 * world, num_max_tokens_per_rank=16, do_expand=True,
                   do_cpu_sync=True, async_with_compute_stream=True)

    # All ranks reject these shapes before submission and then reuse this buffer.
    bad_padding = torch.empty((2, 192), dtype=torch.bfloat16, device='npu')
    bad_fp8 = (torch.empty((2, 264), dtype=torch.float8_e4m3fn, device='npu'),
               torch.empty((2, 3), dtype=torch.float32, device='npu'))
    for payload, padding, message in (
            (bad_padding, True, 'Zero padding requires a vector-aligned hidden row'),
            (bad_fp8, False, '32-bit FP8 scales require a 32-byte aligned hidden row')):
        try:
            buffer.dispatch(payload, topk_idx=routes, do_zero_padding=padding, defer_epilogue=True, **options)
        except RuntimeError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError('Invalid dispatch geometry reached submission')

    batch = 2 * (rank % 2)
    original = tokens(rank, batch)
    pending = buffer.dispatch(original.npu(), topk_idx=routes[:batch], defer_epilogue=True, **options)
    consumer = torch.npu.Stream()
    with torch.npu.stream(consumer):
        recv, _, _, handle = pending.wait()
        combined = buffer.combine(recv, handle, async_with_compute_stream=True, defer_epilogue=True)
    output, _ = combined.wait()
    checks = [(recv.cpu(), tokens(source, 2 * (source % 2))), (output.cpu(), original)]

    # Rank 0 reduces experts 0 and 1 to FP16. The source then adds its bias and rank 1's expert 2.
    first = (1 + torch.arange(256, dtype=torch.float32, device='cpu') / 1024).half().reshape(1, 256)
    half_ulp = 1 / 2048
    routing = torch.tensor([[0, 1, 2]], dtype=torch.int64, device='npu')
    received, _, _, handle, _ = launch(buffer, 'dispatch', 1, x=first.npu(), topk_idx=routing, **options)
    if rank == 0:
        received[world:].fill_(half_ulp)  # Expert-major layout, one row per source per expert.
    elif rank == 1:
        received.fill_(-1)
    returned, _, _ = launch(buffer, 'combine', 1, x=received, handle=handle,
                            bias=torch.full_like(first, half_ulp).npu(), async_with_compute_stream=True)
    local_partial = (first.float() + half_ulp).half()
    checks.append((returned.cpu(), (half_ulp + local_partial.float() - 1).half()))

    # An empty sender can supply CPU weights while its peers send weights to its NPU.
    cpu_weights = torch.tensor([[0.25], [0.75]], dtype=torch.float32, device='cpu')
    local_weights = cpu_weights.npu() if batch else cpu_weights[:0]
    batch_x = original.bfloat16().npu()
    for capacity in (16, 17):
        step = dict(options, num_max_tokens_per_rank=capacity, do_cpu_sync=False)
        _, _, received_weights, _, _ = launch(
            buffer, 'dispatch', 1, x=batch_x, topk_idx=routes[:batch], topk_weights=local_weights, **step)
        count = 2 * (source % 2)
        checks.append((received_weights[:count], cpu_weights[:count, 0].to(batch_x.device)))

    # A synchronized, all-empty dispatch retains the empty weight tensor's device.
    _, _, empty_weights, _, _ = launch(
        buffer, 'dispatch', 1, x=batch_x[:0], topk_idx=routes[:0], topk_weights=cpu_weights[:0], **options)
    checks.append((empty_weights, cpu_weights[:0, 0]))

    for dtype in (torch.float32, torch.int32):
        x = tokens(rank, 2).to(torch.float8_e4m3fn)
        (values, received_scales), _, _, _, _ = launch(
            buffer, 'dispatch', 1, x=(x.npu(), scales(rank, dtype).npu()), topk_idx=routes,
            use_tma_aligned_col_major_sf=True, **options)
        # The next dispatch materializes the column-major scales on its own stream.
        with torch.npu.stream(consumer):
            (_, forwarded_scales), _, _, _, _ = launch(
                buffer, 'dispatch', 1, x=(values, received_scales), topk_idx=routes, **options)
            checks.append((forwarded_scales.cpu(), scales((source - 1) % world, dtype)))
        checks.append((values.cpu(), tokens(source, 2).to(torch.float8_e4m3fn)))
        checks.append((received_scales.cpu(), scales(source, dtype)))

    # Alternate implementations on the same buffer. A cached call inherits its
    # originating handle's capacity, including when the constructor default differs.
    x = tokens(rank, 2).bfloat16().npu()
    weights = torch.tensor([[0.25], [0.75]], dtype=torch.float32, device='npu')
    for capacity in (16, 17, 16):
        step = dict(options, num_max_tokens_per_rank=capacity)
        # Native dispatch accepts contiguous [B, K, 1] indices. Combine uses the
        # 2-D indices held by the separate handle below, as required by main.
        routed, _, routed_weights, _, _ = launch(
            buffer, 'dispatch', 1, x=x, topk_idx=routes.unsqueeze(-1),
            topk_weights=weights.unsqueeze(-1), **step)
        checks.extend(((routed.cpu(), tokens(source, 2).bfloat16()), (routed_weights.cpu(), weights.cpu().flatten())))
        _, _, _, handle, _ = launch(buffer, 'dispatch', 1, x=x, topk_idx=routes,
                                    topk_weights=weights.unsqueeze(-1), **step)
        received, _, received_weights, _, _ = launch(
            buffer, 'dispatch', 1, x=x, handle=handle, topk_weights=weights.unsqueeze(-1),
            do_expand=True, async_with_compute_stream=True)
        returned, returned_weights, _ = launch(
            buffer, 'combine', 1, x=received, handle=handle,
            topk_weights=received_weights, async_with_compute_stream=True)
        checks.extend(((returned.cpu(), x.cpu()), (returned_weights.cpu(), weights.cpu())))

    if world >= 4:
        # For a [0,0,1,1] server map the old relay assigned the second token
        # to rank 3, which has no expert rows and intentionally omits weights.
        routing = torch.tensor([[0, 2, 4], [0, 2, 4]], dtype=torch.int64, device='npu')
        weights = torch.tensor([[0.125, 0.5, 0.375], [0.5, 0.125, 0.375]], dtype=torch.float32, device='cpu')
        x = torch.zeros((2, 256), dtype=torch.bfloat16, device='npu')
        received, _, received_weights, handle, _ = launch(
            buffer, 'dispatch', 1, x=x, topk_idx=routing, topk_weights=weights.npu(), num_sms=32, **options)
        partials = [0.73828125, 0.375, 0.59765625]
        received.fill_(partials[rank] if rank < 3 else 0)
        returned, returned_weights, _ = launch(
            buffer, 'combine', 1, x=received, handle=handle,
            topk_weights=None if rank == 3 else received_weights, async_with_compute_stream=True)
        # main's flat sum rounds to 1.7109375. Rounding ranks 0+1 first gives 1.703125.
        checks.append((returned.cpu(), torch.full((2, 256), 1.7109375, dtype=torch.bfloat16, device='cpu')))
        checks.append((returned_weights.cpu() if returned_weights is not None else None,
                       None if rank == 3 else weights))

    buffer.destroy()
    dist.destroy_process_group()
    # Compare after the last collective so a rank-local mismatch cannot strand peers.
    for actual, expected in checks:
        if expected is None:
            assert actual is None
            continue
        assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
        torch.testing.assert_close(actual.contiguous().view(torch.uint8),
                                   expected.contiguous().view(torch.uint8), rtol=0, atol=0)
    if rank == 0:
        print(f'PASS EP{world} decode payloads and cross-stream epilogues')


if __name__ == '__main__':
    import os
    from torch.multiprocessing import spawn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-processes', type=int, default=8, help='devices per node, as in test_ep.py')
    args = parser.parse_args()
    world = int(os.getenv('WORLD_SIZE', '1')) * args.num_processes
    if args.num_processes < 1 or world < 2:
        parser.error('This test needs at least one device per node and two ranks in total')
    spawn(worker, args=(args.num_processes,), nprocs=args.num_processes)
