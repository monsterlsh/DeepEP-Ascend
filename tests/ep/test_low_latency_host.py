"""CPU contracts for the private experimental LL test fixture.

Mocks check allocation, argument forwarding and Python lifecycle only. Device
correctness and EXP-TIMEOUT-01 remain separate NPU acceptance requirements.
"""

import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from _host_test_helpers import Device, Event, Tensor, fake_modules, load

ROOT = Path(__file__).resolve().parents[2]


class HostContracts(unittest.TestCase):
    def setUp(self):
        modules, self.torch, self.native = fake_modules()
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        load('deep_ep.buffers.base', 'deep_ep/buffers/base.py')
        load('deep_ep.utils.event', 'deep_ep/utils/event.py')
        self.ep = load('deep_ep.buffers.ep', 'deep_ep/buffers/ep.py')
        load('deep_ep.buffers.low_latency', 'deep_ep/buffers/low_latency.py')
        self.fixture = load('_low_latency_fixture', 'tests/ep/_low_latency_fixture.py')
        self.group = SimpleNamespace(rank=lambda: 0, size=lambda: 8)
        self.runtime = SimpleNamespace(context=object(), destroy=Mock())
        self.native.LowLatencyBuffer = Mock(return_value=self.runtime)
        self.native.EPBuffer = Mock(side_effect=AssertionError('Unexpected native allocation'))
        modules['deep_ep.comm'].get_hccl_comm_handle = lambda group: SimpleNamespace(get=lambda: 'hccl')
        self.torch.npu.synchronize = Mock()
        self.torch.distributed.barrier = Mock()
        self.buffer = self.fixture._PrivateLowLatencyBuffer(self.group, 16)

    def dispatch(self, **kwargs):
        params = dict(topk_idx=Tensor((2, 2), self.torch.int64),
                      topk_weights=Tensor((2, 2), self.torch.float32), num_experts=16, do_expand=True)
        params.update(kwargs)
        return self.buffer.dispatch(Tensor((2, 128)), **params)

    def test_private_fixture_allocates_only_the_existing_private_runtime(self):
        self.native.LowLatencyBuffer.assert_called_once_with(0, 8, 'hccl', True, 300, 100, True)
        self.native.EPBuffer.assert_not_called()
        self.assertFalse(hasattr(self.buffer, 'low_latency_mode'))
        self.assertIs(self.buffer.context, self.runtime.context)
        self.torch.distributed.barrier.assert_called_once_with(self.group)
        self.assertEqual(self.torch.npu.synchronize.call_count, 2)

    def test_private_fixture_rejects_invalid_construction_before_allocation(self):
        self.native.LowLatencyBuffer.reset_mock()
        for size, capacity, options in ((4, 16, {}), (8, 0, {}), (8, 257, {}),
                                         (8, 16, {'context_gpu_timeout_secs': 0})):
            with self.subTest(size=size, capacity=capacity, options=options), self.assertRaises(ValueError):
                self.fixture._PrivateLowLatencyBuffer(
                    SimpleNamespace(size=lambda: size, rank=lambda: 0), capacity, **options)
        self.native.LowLatencyBuffer.assert_not_called()

    def test_private_handle_cannot_enter_native_or_another_private_buffer(self):
        recv, _, _, handle, _ = self.dispatch()
        self.assertNotIsInstance(handle, self.ep.EPHandle)
        native_buffer = self.ep.EPBuffer.__new__(self.ep.EPBuffer)
        native_buffer.runtime = SimpleNamespace(dispatch=Mock(), combine=Mock())
        with self.assertRaises(AttributeError):
            native_buffer.combine(recv, handle)
        with self.assertRaises(AttributeError):
            native_buffer.dispatch(Tensor((2, 128)), handle=handle, do_expand=True)
        native_buffer.runtime.dispatch.assert_not_called()
        native_buffer.runtime.combine.assert_not_called()
        other = self.fixture._PrivateLowLatencyBuffer(self.group, 16)
        with self.assertRaisesRegex(ValueError, 'from this buffer'):
            other.combine(recv, handle)
        with self.assertRaisesRegex(ValueError, 'from this buffer'):
            self.buffer.combine(recv, SimpleNamespace())
        self.assertEqual([op for op, _ in self.native.calls], ['dispatch'])

    def test_private_fixture_destroy_clears_state_and_is_idempotent(self):
        recv, _, _, handle, _ = self.dispatch()
        self.buffer.destroy()
        self.buffer.destroy()
        self.runtime.destroy.assert_called_once()
        self.assertIsNone(self.buffer.runtime)
        self.assertIsNone(self.buffer.context)
        self.assertIsNone(self.buffer.hccl_comm_handle)
        self.assertEqual(self.buffer._low_latency.tilings, {})
        self.assertIsNone(self.buffer._low_latency.last_event)
        self.assertIsNone(self.buffer._low_latency.inflight)
        with self.assertRaisesRegex(RuntimeError, 'destroyed'):
            self.buffer.combine(recv, handle)

    def test_private_ep8_executes_low_latency_and_returns_exact_rows(self):
        recv, indices, weights, handle, event = self.dispatch()
        self.assertTrue(handle.is_low_latency)
        self.assertEqual(recv.shape, (3, 128))
        self.assertEqual(handle.num_recv_tokens_per_expert_list, [2, 1])
        self.assertIsNone(indices)
        self.assertIsNone(weights)
        self.assertEqual(self.native.calls[0][0], 'dispatch')
        self.assertTrue(all(t.recorded for t in event.extra_tensors))

    def test_throughput_buffer_keeps_its_native_dispatch(self):
        native_buffer = self.ep.EPBuffer.__new__(self.ep.EPBuffer)
        native_buffer.runtime = SimpleNamespace(dispatch=lambda *args: (_ for _ in ()).throw(RuntimeError('HT selected')))
        native_buffer.num_max_tokens_per_rank = 16
        with self.assertRaisesRegex(RuntimeError, 'HT selected'):
            native_buffer.dispatch(Tensor((2, 128)), topk_idx=Tensor((2, 2), self.torch.int64),
                                   topk_weights=Tensor((2, 2), self.torch.float32),
                                   num_experts=16, do_expand=True, num_sms=32)
        self.assertEqual(self.native.calls, [])

    def test_no_host_sync_uses_actual_batch_capacity(self):
        with patch.object(Tensor, 'cpu', side_effect=AssertionError('host sync')):
            recv, _, _, handle, _ = self.dispatch(do_cpu_sync=False)
        self.assertEqual(recv.shape, (32, 128))
        self.assertEqual(handle.num_expanded_tokens, 32)

    def test_repeat_combine_serializes_without_passing_dispatch_weights(self):
        recv, _, _, handle, dispatch_event = self.dispatch()
        self.buffer.combine(recv, handle)
        self.buffer.combine(recv, handle)
        self.assertEqual(dispatch_event.event.waits, 1)
        self.assertEqual([op for op, _ in self.native.calls], ['dispatch', 'combine', 'combine'])
        for op, args in self.native.calls:
            if op == 'combine':
                self.assertNotIn('source_weights', args[1])
                self.assertIs(args[1]['input'], recv)

    def test_rejects_incompatible_options_before_launch(self):
        for options in (dict(do_expand=False), dict(expert_alignment=128), dict(do_handle_copy=True),
                        dict(defer_epilogue=True), dict(num_sms=16), dict(num_qps=1)):
            with self.subTest(options=options), self.assertRaises(NotImplementedError):
                self.dispatch(**options)
        self.assertEqual(self.native.calls, [])

    def test_rejects_wrong_device_dtype_and_capacity(self):
        for options in (dict(topk_idx=Tensor((2, 2), self.torch.int32)),
                        dict(topk_weights=Tensor((2, 2), self.torch.float32, ('cpu', None))),
                        dict(num_max_tokens_per_rank=1), dict(num_max_tokens_per_rank=257)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.dispatch(**options)
        self.assertEqual(self.native.calls, [])

    def test_rejects_foreign_handle_and_unsupported_weight_metadata(self):
        recv, _, _, handle, _ = self.dispatch()
        with self.assertRaises(NotImplementedError):
            self.buffer.combine(recv, handle, topk_weights=Tensor((3,), self.torch.float32))
        handle._handle._buffer_token = object()
        with self.assertRaises(ValueError):
            self.buffer.combine(recv, handle)
        self.assertEqual(len(self.native.calls), 1)

    def test_failed_launch_is_not_retried(self):
        with patch.object(self.native, 'dispatch_low_latency', side_effect=RuntimeError('launch failed')):
            with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                self.dispatch()
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            self.dispatch()

    def test_failed_combine_keeps_inputs_alive_and_blocks_further_launches(self):
        recv, _, _, handle, _ = self.dispatch()
        with patch.object(self.native, 'combine_low_latency', side_effect=RuntimeError('combine failed')) as launch:
            with self.assertRaisesRegex(RuntimeError, 'combine failed'):
                self.buffer.combine(recv, handle)
            self.assertIn(recv, self.buffer._low_latency.inflight)
            self.assertIn(handle.relay_read_index, self.buffer._low_latency.inflight)
            with self.assertRaisesRegex(RuntimeError, 'restart'):
                self.buffer.combine(recv, handle)
            with self.assertRaisesRegex(RuntimeError, 'restart'):
                self.dispatch()
        launch.assert_called_once()

    def test_combine_capacity_is_checked_before_dispatch_launch(self):
        with patch.object(self.native, 'low_latency_combine_tiling', side_effect=ValueError('combine capacity')):
            with self.assertRaisesRegex(ValueError, 'combine capacity'):
                self.dispatch()
        self.assertEqual(self.native.calls, [])
        # Rejected input has not entered communication and does not poison the buffer.
        self.dispatch()
        self.assertEqual([operation for operation, _ in self.native.calls], ['dispatch'])

    def test_destroyed_buffer_is_rejected(self):
        self.buffer.runtime = None
        with self.assertRaisesRegex(RuntimeError, 'destroyed'):
            self.dispatch()

    def test_fp8_keeps_opaque_scale_dtype(self):
        for scale_dtype in (self.torch.int32, self.torch.float32):
            result = self.buffer.dispatch(
                (Tensor((2, 128), self.torch.float8_e4m3fn), Tensor((2, 1), scale_dtype)),
                topk_idx=Tensor((2, 2), self.torch.int64), topk_weights=Tensor((2, 2), self.torch.float32),
                num_experts=16, do_expand=True)
            self.assertEqual(result[0][1].dtype, scale_dtype)

    def test_capacity_selects_tiling_and_survives_handle_reuse(self):
        # Equal B with different M must not share tiling across the direct/dedup
        # threshold. Combine must use the originating handle after another dispatch.
        with patch.object(self.native, 'low_latency_dispatch_tiling', return_value=b'\0' * 32) as dispatch_tiling, \
             patch.object(self.native, 'low_latency_combine_tiling', return_value=b'\0' * 20) as combine_tiling:
            small, _, _, direct, _ = self.dispatch(num_max_tokens_per_rank=32)
            large, _, _, dedup, _ = self.dispatch(num_max_tokens_per_rank=33)
            self.buffer.combine(small, direct)
            self.buffer.combine(large, dedup)
            self.buffer.combine(small, direct)
            self.dispatch(num_max_tokens_per_rank=256)
        self.assertEqual([call.args[-1] for call in dispatch_tiling.call_args_list], [32, 33, 256])
        self.assertEqual([call.args[-1] for call in combine_tiling.call_args_list], [32, 33, 32, 33, 256])


class SourceContracts(unittest.TestCase):
    def test_jit_dependency_closure_is_packaged_and_acyclic(self):
        include = ROOT / 'deep_ep/include'
        seen, active = set(), set()

        def visit(name):
            self.assertNotIn(name, active, f'Circular DeepJIT include: {name}')
            if name in seen:
                return
            path = include / name
            self.assertTrue(path.is_file(), f'Missing JIT header: {name}')
            self.assertIn(path.suffix, ('.h', '.hpp'))
            active.add(name)
            code = path.read_text(encoding='utf-8')
            self.assertNotRegex(code, r'#\s*include\s*"')
            for dep in re.findall(r'#\s*include\s*<([^>]+)>', code):
                if dep.startswith('deep_ep/'):
                    visit(dep)
            active.remove(name)
            seen.add(name)

        visit('deep_ep/impls/low_latency/dispatch.hpp')
        visit('deep_ep/impls/low_latency/combine.hpp')
        self.assertGreater(len(seen), 20)

    def test_ep8_test_matrix_is_available_without_torch(self):
        spec = importlib.util.spec_from_file_location('ll_st', ROOT / 'tests/ep/test_low_latency.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parser().parse_args(['--dtype', 'bf16', 'fp16', 'fp8', '--fp8-scale-dtype', 'int32', 'float32'])
        module.validate(args, 8)
        self.assertEqual(len(list(module.cases(args))), 16)
        with self.assertRaises(ValueError):
            module.validate(args, 4)

    def test_capacity_matrix_covers_both_paths_and_rejects_invalid_capacity(self):
        module = load('ll_capacity_st', 'tests/ep/test_low_latency.py')
        args = module.parser().parse_args(['--num-tokens', '16', '--num-max-tokens-per-rank', '32', '33', '256'])
        module.validate(args, 8)
        self.assertEqual({case['m'] for case in module.cases(args)}, {32, 33, 256})
        self.assertEqual({case['route'] for case in module.cases(args)}, {'random', 'hotspot'})
        for capacity in (0, 15, 257):
            args.num_max_tokens_per_rank = [capacity]
            with self.assertRaises(ValueError):
                module.validate(args, 8)


if __name__ == '__main__':
    unittest.main()
