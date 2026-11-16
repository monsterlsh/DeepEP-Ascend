import copy
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bench_decode import compare, measure


class BenchmarkComparisonTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(backend='official', ranks=['host 0, device 0'],
                         builds=[dict(extension_sha256='main')], config=dict(bs=[256], samples=3),
                         rows=[dict(M=256, mode=mode, operation=op, max_rank_us=[100, 500, 200])
                               for mode in ('fresh', 'cached') for op in ('dispatch', 'combine', 'roundtrip')])
        self.candidate = copy.deepcopy(self.base)
        self.candidate.update(backend='candidate-auto', builds=[dict(extension_sha256='candidate')])

    def test_p50_and_signed_latency_change(self):
        self.candidate['rows'][0]['max_rank_us'] = [120, 160, 400]
        output = io.StringIO()
        compare(self.base, self.candidate, output)
        self.assertEqual(output.getvalue().splitlines()[1], '256,fresh,dispatch,200.00,160.00,-20.00,500.00,400.00,-20.00')

    def test_rejects_incomparable_or_incomplete_runs(self):
        for changes in (dict(builds=self.base['builds']), dict(ranks=['another host']),
                        dict(rows=self.candidate['rows'][:-1]), dict(config=dict(bs=[1], samples=3))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                compare(self.base, dict(self.candidate, **changes), io.StringIO())
        self.candidate['rows'][0]['max_rank_us'].pop()
        with self.assertRaisesRegex(ValueError, 'latency samples'):
            compare(self.base, self.candidate, io.StringIO())

    def test_warmup_is_excluded_then_all_reduce_matches_steps(self):
        values = []
        tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=values.copy))

        def to_tensor(elapsed, **_):
            values.extend(elapsed)
            return tensor

        def reduce_longest(local, op):
            self.assertIs(local, tensor)
            self.assertEqual(op, 'max')
            values[:] = [max(a, b) for a, b in zip(values, [3, 11])]

        dist = SimpleNamespace(barrier=Mock(), all_reduce=reduce_longest, ReduceOp=SimpleNamespace(MAX='max'))
        torch = SimpleNamespace(distributed=dist, npu=SimpleNamespace(synchronize=Mock()), tensor=to_tensor, float32=0)
        op, verify = Mock(side_effect=['warmup', 'a', 'b']), Mock()
        with patch.dict(sys.modules, torch=torch, **{'torch.distributed': dist}), \
                patch('bench_decode.perf_counter_ns', side_effect=[0, 99000, 0, 7000, 0, 1000]):
            self.assertEqual(measure(op, warmup=1, samples=2, verify=verify), [7, 11])
        self.assertEqual(dist.barrier.call_count, 3)
        self.assertEqual(torch.npu.synchronize.call_count, 6)
        verify.assert_called_once_with('b')
