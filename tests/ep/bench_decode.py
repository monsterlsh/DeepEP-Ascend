"""Unprofiled public EPBuffer latency. Use the same harness for main and candidate."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
from statistics import median
import sys
from time import perf_counter_ns

from compat_config import configure_backend, configure_server_ids, validate_server_ids
from ep_reference import agree, make_decode_routes
from test_ep_auto_compat import build_identity, file_hash, make_input


def measure(operation, warmup, samples, verify):
    import torch
    import torch.distributed as dist

    elapsed = []
    for step in range(warmup + samples):
        dist.barrier()
        torch.npu.synchronize()
        start = perf_counter_ns()
        result = operation()
        torch.npu.synchronize()
        duration = (perf_counter_ns() - start) / 1000
        if step == warmup + samples - 1:
            verify(result)
        del result
        if step >= warmup:
            elapsed.append(duration)
    # Align by step before taking the median: a different rank may be slow each time.
    longest = torch.tensor(elapsed, device='npu', dtype=torch.float32)
    dist.all_reduce(longest, op=dist.ReduceOp.MAX)
    return longest.cpu().tolist()


def run_capacity(buffer, args, m, rank, world):
    import torch

    routes = make_decode_routes(world, m, args.num_topk, args.num_experts, args.rank_load)[rank]
    batch = len(routes)
    x = make_input(rank, batch, 'bf16', 'row', hidden=args.hidden).clamp(-4, 4)
    indices = torch.tensor(routes, dtype=torch.int64, device='npu')
    weights = (torch.arange(1, args.num_topk + 1, dtype=torch.float32, device='npu') / 32).repeat(batch, 1)
    common = dict(x=x, topk_weights=weights, do_cpu_sync=False, do_expand=True,
                  do_zero_padding=True, async_with_compute_stream=True, defer_epilogue=True)

    def dispatch(route):
        return buffer.dispatch(**common, **route).current_stream_wait(release_handle=True)

    def combine(received):
        recv, _, side, handle = received
        return buffer.combine(recv, handle, topk_weights=side, async_with_compute_stream=True,
                              defer_epilogue=True).current_stream_wait(release_handle=True)

    fresh = dict(topk_idx=indices, num_experts=args.num_experts, expert_alignment=128,
                 num_max_tokens_per_rank=m, num_sms=32)
    handle = dispatch(fresh)[-1]
    rows = []

    def verify_pair(result):
        combined, restored = result

        def check():
            torch.testing.assert_close(combined, x * args.num_topk, atol=0, rtol=0)
            torch.testing.assert_close(restored, weights, atol=0, rtol=0)

        agree(check)

    for mode, route in [('fresh', fresh), ('cached', dict(handle=handle))]:
        def record(name, run, verify):
            samples = measure(run, args.warmup, args.samples, verify)
            rows.append(dict(M=m, mode=mode, operation=name, max_rank_us=samples))

        record('dispatch', lambda: dispatch(route), lambda result: verify_pair(combine(result)))
        received = dispatch(route)
        record('combine', lambda: combine(received), verify_pair)
        del received
        # No synchronization between dispatch and combine inside the round trip.
        record('roundtrip', lambda: combine(dispatch(route)), verify_pair)
    return rows


def worker(local_rank, args):
    import torch
    import torch.distributed as dist
    import torch_npu

    configure_backend(args.backend)
    torch.npu.set_device(local_rank)
    import deep_ep
    rank = int(os.getenv('RANK', 0)) * args.num_processes + local_rank
    world = int(os.getenv('WORLD_SIZE', 1)) * args.num_processes
    configure_server_ids(args.server_ids, rank)
    dist.init_process_group('hccl', rank=rank, world_size=world, device_id=torch.device(f'npu:{local_rank}'),
                            init_method=f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '8361')}")
    group = dist.group.WORLD
    identity = build_identity(deep_ep, torch, torch_npu, args)
    metadata = dict(host=socket.gethostname(), local_rank=local_rank,
                    device=torch.npu.get_device_name(local_rank), torch=identity['torch'],
                    torch_npu=identity['torch_npu'], python=identity['python'],
                    cann=getattr(torch.version, 'cann', None), ascend_home=identity['ascend_home_path'],
                    visible_devices=os.getenv('ASCEND_RT_VISIBLE_DEVICES'))
    config = dict(world=world, hidden=args.hidden, num_topk=args.num_topk, num_experts=args.num_experts,
                  server_ids=args.server_ids, bs=args.bs, pool_capacity=args.capacity, rank_load=args.rank_load,
                  barrier_in_prologue=args.barrier_in_prologue, warmup=args.warmup, samples=args.samples,
                  driver=file_hash(__file__), harness=identity['harness_files_sha256'])
    peers = [None] * world
    dist.all_gather_object(peers, (metadata, identity, config, args.backend), group)
    if any((plan, backend, build['extension_sha256']) != (config, args.backend, identity['extension_sha256'])
           for _, build, plan, backend in peers):
        raise ValueError('Ranks have different benchmark configurations or builds')
    report = dict(backend=args.backend, config=config, ranks=[peer[0] for peer in peers],
                  builds=[peer[1] for peer in peers], rows=[])
    deep_ep._C.set_barrier_in_prologue(args.barrier_in_prologue)
    with torch.inference_mode():
        buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=args.capacity, hidden=args.hidden, num_topk=args.num_topk,
                                  deterministic=False, explicitly_destroy=True)
        for m in args.bs:
            rows = run_capacity(buffer, args, m, rank, world)
            report['rows'].extend(rows)
            if rank == 0:
                print(f'EP{world} M={m} measured', flush=True)
        buffer.destroy()
    dist.destroy_process_group()
    if rank == 0:
        with args.output.open('x', encoding='utf-8') as out:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write('\n')


def compare(main, candidate, stream):
    if (main['backend'], candidate['backend']) != ('official', 'candidate-auto'):
        raise ValueError('Expected an official main run followed by a candidate-auto run')
    if main['config'] != candidate['config'] or main['ranks'] != candidate['ranks']:
        raise ValueError('Different inputs, harness, timing options, hosts, or runtimes')
    if main['builds'][0]['extension_sha256'] == candidate['builds'][0]['extension_sha256']:
        raise ValueError('The same extension binary was used in both runs')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(('M', 'mode', 'operation', 'main_p50_us', 'candidate_p50_us', 'change_percent',
                     'main_p99_us', 'candidate_p99_us', 'p99_change_percent'))
    expected = ((m, mode, operation) for m in main['config']['bs'] for mode in ('fresh', 'cached')
                for operation in ('dispatch', 'combine', 'roundtrip'))
    for key, left, right in zip(expected, main['rows'], candidate['rows'], strict=True):
        for row in (left, right):
            if tuple(row[field] for field in ('M', 'mode', 'operation')) != key:
                raise ValueError('Measurement rows do not match the benchmark plan')
            samples = row['max_rank_us']
            if len(samples) != main['config']['samples'] or any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError('Missing or invalid latency samples')
        baseline, current = median(left['max_rank_us']), median(right['max_rank_us'])
        index = math.ceil(0.99 * main['config']['samples']) - 1
        tail = [sorted(row['max_rank_us'])[index] for row in (left, right)]
        writer.writerow((*key, f'{baseline:.2f}', f'{current:.2f}', f'{100*(current/baseline-1):+.2f}',
                         f'{tail[0]:.2f}', f'{tail[1]:.2f}', f'{100*(tail[1]/tail[0]-1):+.2f}'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--backend', required=True, choices=('official', 'candidate-auto'))
    run.set_defaults(build_label=None)
    run.add_argument('--expected-source-commit')
    run.add_argument('--hidden', type=int, default=7168)
    run.add_argument('--num-topk', type=int, default=8)
    run.add_argument('--num-experts', type=int, default=128)
    run.add_argument('--server-ids', type=int, nargs='+', help='Physical server IDs in EP rank order')
    run.add_argument('--bs', nargs='+', type=int, default=[1, 2, 8, 16, 32, 64, 128, 256])
    run.add_argument('--rank-load', choices=('decode', 'uniform'), default='decode')
    run.add_argument('--barrier-in-prologue', action='store_true')
    run.add_argument('--num-processes', type=int, default=8, help='NPUs per node')
    run.add_argument('--capacity', type=int, default=256, help='allocation capacity shared across cases')
    run.add_argument('--warmup', type=int, default=15)
    run.add_argument('--samples', type=int, default=60)
    run.add_argument('--output', type=Path, required=True, help='rank 0 output JSON')
    diff = sub.add_parser('compare')
    diff.add_argument('official', type=Path)
    diff.add_argument('candidate', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'compare':
        compare(json.loads(args.official.read_text(encoding='utf-8')),
                json.loads(args.candidate.read_text(encoding='utf-8')), sys.stdout)
        return
    if min(args.bs) < 1 or max(args.bs) > min(args.capacity, 256):
        parser.error('each M must be in [1, 256] and fit the allocated pool capacity')
    if args.warmup < 1 or args.samples < 1 or args.num_processes < 1:
        parser.error('warmup, samples and num-processes must be positive')
    world = int(os.getenv('WORLD_SIZE', 1)) * args.num_processes
    if world not in (2, 4, 8, 16, 32, 64, 128):
        parser.error('expected a power-of-two EP group size from 2 through 128')
    try:
        validate_server_ids(args.server_ids, world)
    except ValueError as error:
        parser.error(str(error))
    if args.num_experts <= 0 or args.num_experts % world or not 1 <= args.num_topk <= min(32, args.num_experts):
        parser.error('experts must divide evenly across ranks and 1 <= K <= min(32, E)')
    if args.hidden <= 0 or args.hidden % 128:
        parser.error('BF16 combine requires H divisible by 128')
    if args.output.exists():
        parser.error('select a new output JSON path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from torch.multiprocessing import spawn
    spawn(worker, args=(args,), nprocs=args.num_processes)


if __name__ == '__main__':
    main()
