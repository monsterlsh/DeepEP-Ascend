"""Replay bench_decode inputs and count hidden payloads per dispatch or combine.

Use the same harness revision and Python version as the benchmark. The report's
server IDs are used by default. Override them to evaluate another grouping in
process-group rank order. Counts exclude metadata,
scales, barriers, protocol overhead and physical network effects.
"""

import argparse
import json
from pathlib import Path

from ep_reference import make_decode_routes


def count_rows(routes, rank_domains, num_experts, capacity):
    world = len(routes)
    if not world or len(rank_domains) != world:
        raise ValueError('provide one server label for each EP rank')
    if num_experts <= 0 or num_experts % world:
        raise ValueError('experts must divide evenly across EP ranks')
    experts_per_rank = num_experts // world
    members = {}
    for rank, domain in enumerate(rank_domains):
        members.setdefault(domain, []).append(rank)
    tokens = cross_rank = cross_server = within_server = combine_within = 0
    for source, rows in enumerate(routes):
        if len(rows) > capacity:
            raise ValueError('source batch exceeds per-rank capacity')
        for token, row in enumerate(rows):
            tokens += 1
            if any(expert < -1 or expert >= num_experts for expert in row):
                raise ValueError('invalid expert index')
            destinations = {expert // experts_per_rank for expert in row if expert >= 0}
            remote = {rank for rank in destinations if rank_domains[rank] != rank_domains[source]}
            cross_rank += len(remote)
            cross_server += len({rank_domains[rank] for rank in remote})
            within_server += len(destinations - remote - {source})
            # Combine balances relay ownership over all ranks on each server.
            # Even a rank with no expert for this token can be elected.
            for domain in {rank_domains[rank] for rank in destinations}:
                peers = members[domain]
                relay = peers[(source * capacity + token) % len(peers)]
                combine_within += sum(rank != relay and rank_domains[rank] == domain for rank in destinations)
                combine_within += int(domain == rank_domains[source] and relay != source)
    return dict(tokens=tokens, direct_cross_server_rows=cross_rank, relay_cross_server_rows=cross_server,
                direct_within_server_rows=within_server,
                dispatch_relay_within_server_rows=within_server + cross_rank - cross_server,
                combine_relay_within_server_rows=combine_within)


def analyze(config, rank_domains=None):
    world, hidden = config['world'], config['hidden']
    mapping_source = 'counterfactual_override' if rank_domains is not None else 'benchmark_configuration'
    if rank_domains is None:
        rank_domains = config.get('server_ids')
        if rank_domains is None:
            raise ValueError('benchmark has no server IDs; supply --rank-domains to model a grouping')
    if world != len(rank_domains):
        raise ValueError('provide one server label for each EP rank')
    rows = []
    for m in config['bs']:
        routes = make_decode_routes(world, m, config['num_topk'], config['num_experts'], config['rank_load'])
        count = count_rows(routes, rank_domains, config['num_experts'], m)
        # bench_decode uses BF16 for both dispatch and combine. Bytes here are
        # for one invocation in either direction, not for the entire round trip.
        rows.append(dict(M=m, **count,
                         direct_cross_server_hidden_bytes=count['direct_cross_server_rows'] * hidden * 2,
                         relay_cross_server_hidden_bytes=count['relay_cross_server_rows'] * hidden * 2))
    return dict(rank_domains=rank_domains, mapping_source=mapping_source, rows=rows,
                units='Logical BF16 hidden payload per operation; excludes metadata, scales and protocol traffic.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='bench_decode run JSON')
    parser.add_argument('--rank-domains', nargs='+', help='override the benchmark grouping with one server label per EP rank')
    args = parser.parse_args()
    config = json.loads(args.report.read_text(encoding='utf-8'))['config']
    try:
        result = analyze(config, args.rank_domains)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
