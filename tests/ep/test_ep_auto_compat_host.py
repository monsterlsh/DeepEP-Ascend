"""CPU tests of the compatibility harness, not validation of NPU kernels."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_ep_auto_compat as suite
from compat_config import parse_args, plan_cases


class ContractTests(unittest.TestCase):
    def setUp(self):
        # The C++ policy grid exercises the production implementation on CPU.
        # Here verify that the profiler passes the case geometry to that policy.
        self.shape_supported = Mock(return_value=True)
        native = SimpleNamespace(_C=SimpleNamespace(_decode_shape_supported=self.shape_supported))
        modules = patch.dict(sys.modules, deep_ep=native)
        modules.start()
        self.addCleanup(modules.stop)

    def test_positive_batches_at_all_supported_world_sizes(self):
        runtime = dict(arch=3510, num_vec_cores=64, num_ai_cores=32)
        for world in (2, 4, 8, 16, 32, 64, 128):
            for m in (1, 3, 16, 32, 64, 128, 255, 256):
                for loads in ([m] * world, [max(1, m - rank) for rank in range(world)], [1] * world):
                    case = dict(rank_bs=loads, M=m, alignment=128, dtype='bf16', cpu_sync=False,
                                deterministic=False, deferred=True, hidden=7168, num_topk=8, num_experts=128)
                    with self.subTest(world=world, m=m, loads=loads):
                        self.assertEqual(suite.expected_transport(case, 'candidate-auto', runtime), ['urma', 'mte'])
                        self.shape_supported.assert_called_with(world, m, 7168, 8, 128, 0)

    def test_unsupported_shapes_options_and_backends_fall_back(self):
        runtime = dict(arch=3510, num_vec_cores=64, num_ai_cores=32)
        case = dict(rank_bs=[2] + [1] * 7, M=2, alignment=128, dtype='bf16', cpu_sync=False,
                    deterministic=False, deferred=True, hidden=7168, num_topk=6, num_experts=384)
        for changes in (dict(rank_bs=[0] + [1] * 7), dict(rank_bs=[3] + [1] * 7),
                        dict(alignment=1), dict(cpu_sync=True), dict(deferred=False),
                        dict(deterministic=True), dict(dtype='int8')):
            with self.subTest(changes=changes):
                self.assertEqual(suite.expected_transport(dict(case, **changes), 'candidate-auto', runtime), 'urma')
        for options in (dict(has_weights=False), dict(has_stats=True), dict(has_bias=True)):
            self.assertEqual(suite.expected_transport(case, 'candidate-auto', runtime, **options), 'urma')
        for backend in ('candidate-disabled', 'official'):
            self.assertEqual(suite.expected_transport(case, backend, runtime), 'urma')
        self.shape_supported.assert_not_called()

    def test_model_geometry_scale_width_and_native_resource_result(self):
        runtime = dict(arch=3510, num_vec_cores=64, num_ai_cores=32)
        case = dict(rank_bs=[2] * 64, M=256, alignment=128, cpu_sync=False, deterministic=False,
                    deferred=True, hidden=7168, num_topk=6, num_experts=384)
        for dtype, words in [('bf16', 0), ('fp16', 0), ('fp8', 112)]:
            for fits in [True, False]:
                self.shape_supported.return_value = fits
                for layout in ['row', 'col']:
                    result = suite.expected_transport(dict(case, dtype=dtype, scale_layout=layout),
                                                      'candidate-auto', runtime)
                    self.assertEqual(result, ['urma', 'mte'] if fits else 'urma')
                    self.shape_supported.assert_called_with(64, 256, 7168, 6, 384, words)

    def test_old_handle_trace_uses_its_original_capacity_and_layout(self):
        current = plan_cases(parse_args(['--bs', '32', '--capacity-mode', 'batch', '--alignment', '1',
                                         '--deferred', 'true', '--deterministic', 'false',
                                         '--edge-routes', 'off']))[0]
        old = dict(current, alignment=128, M=3, rank_bs=[2] * 8)
        handle = SimpleNamespace(expert_alignment=128, num_max_tokens_per_rank=3,
                                 topk_idx=SimpleNamespace(size=lambda dim: 2))
        evidence = {}
        def capture(label, operation, **kwargs):
            evidence.update(kwargs)
            return nullcontext()
        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        runtime = dict(arch=3510, num_vec_cores=64, num_ai_cores=32)
        with patch.dict(sys.modules, torch=fake_torch), patch.object(suite, 'agree', lambda check: check()), \
                patch.object(suite, 'launch'):
            suite.invoke(None, 'combine', 'old_handle', current, 0, 'candidate-auto', runtime,
                         SimpleNamespace(capture=capture), handle_case=old, handle=handle, topk_weights=object())
        self.assertEqual(evidence['expected_transport'], ['urma', 'mte'])
        self.assertEqual(evidence['metadata']['M'], 3)
        self.assertEqual(evidence['metadata']['B'], 2)
        self.assertEqual(evidence['metadata']['case_id'], current['case_id'])

    def test_both_versions_have_no_public_experiment_markers(self):
        for backend in ('official', 'candidate-auto', 'candidate-disabled'):
            for attribute in ('low_latency_mode', 'is_low_latency'):
                suite.check_native_contract(SimpleNamespace(), attribute, backend)
                for value in (False, True):
                    with self.subTest(backend=backend, attribute=attribute, value=value), self.assertRaises(AssertionError):
                        suite.check_native_contract(SimpleNamespace(**{attribute: value}), attribute, backend)

    def test_deferred_return_and_hook_once(self):
        class Event:
            calls = 0
            def wait(self):
                self.calls += 1
                return ('x', None, 'weights', 'handle') if self.calls == 1 else None
        event = Event()
        buffer = SimpleNamespace(dispatch=lambda **_: event)
        self.assertEqual(suite.launch(buffer, 'dispatch', True), ('x', None, 'weights', 'handle', event))
        self.assertEqual(event.calls, 2)

    def test_entrypoint_dry_run_does_not_import_torch(self):
        root = Path(__file__).resolve().parent
        code = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'deep_ep'):
            raise AssertionError('Unexpected device dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['test_ep_auto_compat.py', '--dry-run', '--backend', 'official', '--bs', '2',
            '--alignment', '1', '--deferred', 'false', '--deterministic', 'false',
            '--route', 'random', '--edge-routes', 'off']
runpy.run_path('test_ep_auto_compat.py', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['num_cases'], 1)
        self.assertEqual(plan['backend'], 'official')
        self.assertEqual(plan['path_status'], 'unverified')

    def test_capture_setup_failure_never_submits_communication(self):
        class Capture:
            def __enter__(self):
                raise OSError('trace directory unavailable')
        collector = SimpleNamespace(capture=lambda *args, **kwargs: Capture())
        calls = []
        buffer = SimpleNamespace(dispatch=lambda **kwargs: calls.append(kwargs))
        case = plan_cases(parse_args(['--bs', '2', '--edge-routes', 'off']))[0]
        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        with patch.dict(sys.modules, torch=fake_torch), patch.object(suite, 'agree', lambda check: check()):
            with self.assertRaisesRegex(OSError, 'trace directory'):
                suite.invoke(buffer, 'dispatch', 'fresh', case, 0, 'candidate-auto',
                             dict(arch=3510, num_vec_cores=64, num_ai_cores=32), collector, x=None)
        self.assertEqual(calls, [])


class DispatchOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('Torch CPU is needed only for synthetic tensor oracle tests')
        cls.torch = torch

    def fixture(self, dtype='bf16', layout='row', backend='official', scale_format='int16'):
        torch = self.torch
        from ep_reference import payload
        world, rank, alignment, capacity = 8, 0, 16, 256
        # One source sends three slots to rank 0, two to the same expert.
        routes = [[[0, 0, 1, -1, 16, 32, 48, 64]]] + [
            [[16, 17, 18, 19, 20, 21, 22, 23]] for _ in range(7)]
        values = torch.zeros((32, 7168), dtype=torch.bfloat16)
        row = payload([0], [0], 7168, torch.bfloat16)[0]
        for slot in (0, 1, 16):
            values[slot] = row
        weights = torch.zeros(32)
        weights[0], weights[1], weights[16] = 1 / 32, 2 / 32, 3 / 32
        recv = values.half() if dtype == 'fp16' else values
        if dtype == 'fp8':
            scale = suite.scale_packs([0], [0], scale_format=scale_format)[0]
            sf = torch.zeros((32, scale.numel()), dtype=scale.dtype)
            for slot in (0, 1, 16):
                sf[slot] = scale
            if layout == 'col':
                sf = sf.t().contiguous().t()
            recv = (values.to(torch.float8_e4m3fn), sf)
        handle = SimpleNamespace(
            num_max_tokens_per_rank=capacity, expert_alignment=alignment,
            num_unaligned_recv_tokens_per_expert=torch.tensor([2, 1] + [0] * 14),
            num_expanded_tokens=32, num_recv_tokens=1,
            num_recv_tokens_per_expert_list=[16, 16] + [0] * 14,
            psum_num_recv_tokens_per_expert=torch.tensor([2, 17] + [32] * 14),
            psum_num_recv_tokens_per_rank=torch.ones(8, dtype=torch.int32),
            recv_src_metadata=torch.tensor([[0, 1, 16, -1, -1, -1, -1, -1, 0, 2]]))
        options = dict(backend=backend, dtype=dtype, scale_layout=layout, cpu_sync=True, scale_format=scale_format)
        return recv, weights, handle, routes, rank, world, alignment, False, options

    def check_fixture(self, fixture):
        *args, options = fixture
        suite.check_dispatch(*args, **options)

    def test_valid_native_metadata_with_masked_duplicate_and_padding(self):
        for backend in ('official', 'candidate-auto'):
            self.check_fixture(self.fixture(backend=backend))

    def test_fp8_row_and_column_scales_are_checked_as_native_int16_packs(self):
        for layout in ('row', 'col'):
            fixture = self.fixture(dtype='fp8', layout=layout)
            self.check_fixture(fixture)
            fixture[0][1][16, 1] += 1
            with self.assertRaises(AssertionError):
                self.check_fixture(fixture)

    def test_fp16_rows_and_fp8_typed_scales_match_the_dispatch_reference(self):
        self.check_fixture(self.fixture(dtype='fp16', backend='candidate-auto'))
        for fmt in ('fp32', 'e8m0'):
            for layout in ('row', 'col'):
                with self.subTest(fmt=fmt, layout=layout):
                    fixture = self.fixture(dtype='fp8', layout=layout, backend='candidate-auto', scale_format=fmt)
                    self.check_fixture(fixture)
                    fixture[0][1][16, 1] += 1
                    with self.assertRaises(AssertionError):
                        self.check_fixture(fixture)

    def test_padding_route_weight_and_count_corruption_are_rejected(self):
        for field in ('padding', 'weight', 'route', 'counts'):
            fixture = self.fixture()
            if field == 'padding':
                fixture[0][2, 0] = 1
            elif field == 'weight':
                fixture[1][1] = 1
            elif field == 'route':
                fixture[2].recv_src_metadata[0, 1] = 0
            else:
                fixture[2].num_recv_tokens_per_expert_list[0] = 2
            with self.assertRaises(AssertionError):
                self.check_fixture(fixture)

    def test_fp8_input_is_exact_and_cached_scale_bits_change(self):
        torch = self.torch
        original = suite.make_input(127, 256, 'fp8', 'row', device='cpu')
        updated = suite.make_input(127, 256, 'fp8', 'col', payload_scale=2, scale_version=1, device='cpu')
        torch.testing.assert_close(updated[0].float(), original[0].float() * 2, atol=0, rtol=0)
        self.assertEqual(original[1].dtype, torch.int16)
        self.assertEqual(updated[1].stride(), (1, 256))
        self.assertFalse(torch.equal(updated[1], original[1]))
        self.assertTrue(torch.isfinite(original[1].contiguous().view(torch.float32)).all().item())
        empty = suite.make_input(0, 0, 'fp8', 'col', device='cpu')
        self.assertEqual(empty[0].shape, (0, 7168))
        self.assertEqual(empty[1].shape, (0, 112))

    def test_model_scales_preserve_typed_words_across_cached_updates(self):
        torch = self.torch
        for fmt in ('fp32', 'e8m0'):
            initial = suite.scale_packs([0, 4, 2], [1, 0, 2], 4096, 0, fmt)
            cached = suite.scale_packs([0, 4, 2], [1, 0, 2], 4096, 1, fmt)
            self.assertEqual(initial.dtype, torch.float32 if fmt == 'fp32' else torch.int32)
            self.assertEqual(initial.shape, (3, 32))
            self.assertTrue((initial != cached).all())
            self.assertFalse(torch.equal(initial[0], initial[1]))
            _, column = suite.make_input(2, 3, 'fp8', 'col', device='cpu', hidden=4096, scale_format=fmt)
            self.assertEqual(column.stride(), (1, 3))
        native = suite.scale_packs([1], [3], 4096)
        fp32 = suite.scale_packs([1], [3], 4096, scale_format='fp32')
        self.assertTrue(torch.equal(native, fp32.view(torch.int16)))

    def test_nonunit_weights_apply_per_expert_slot_and_server_partials_round(self):
        from ep_reference import expected_combine, make_routes
        torch = self.torch
        routes = [[[0, 0, -1]], []]
        flat = expected_combine(routes, 0, 128, torch.float16, experts=4)
        weighted = expected_combine(routes, 0, 128, torch.float16, experts=4, weighted=True)
        # At column 4 each of the two expert outputs is exactly -1.125.
        self.assertEqual(flat[0, 4].item(), -2.25)
        self.assertEqual(weighted[0, 4].item(), -1.125 * (1 + 2) / 32)
        routes = make_routes(8, 4, 6, 384, 33)
        flat = expected_combine(routes, 0, 128, torch.bfloat16, experts=384, weighted=True)
        relay = expected_combine(routes, 0, 128, torch.bfloat16, experts=384, weighted=True,
                                 server_ids=[7, 9] * 4)
        self.assertTrue((flat != relay).any())


if __name__ == '__main__':
    unittest.main()
