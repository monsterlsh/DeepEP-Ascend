import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def build_helpers():
    tree = ast.parse((ROOT / 'setup.py').read_text(encoding='utf-8'))
    names = {'shmem_revision', 'shmem_header_hashes', 'get_shmem_build_config'}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names
                                                 for target in node.targets)]
    namespace = {'Path': Path, 'os': os, 'hashlib': hashlib, 'current_dir': str(ROOT)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'setup.py'), 'exec'), namespace)
    return namespace


class ShmemBuildTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helpers = build_helpers()
        self.helpers['current_dir'] = str(self.root)

    def make_sdk(self):
        expected = {}
        for name in self.helpers['shmem_header_hashes']:
            path = self.root / 'include' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'mock header\r\n')
            expected[name] = hashlib.sha256(b'mock header\n').hexdigest()
        path = self.root / 'include/host/data_plane/shmem_host_rma.h'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'mock rma\n')
        library = self.root / 'lib/libshmem.so'
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b'mock shared library')
        self.helpers['shmem_header_hashes'] = expected

    def config(self):
        return self.helpers['get_shmem_build_config']()

    def test_optional_build(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.config(), {'enabled': False})

    def test_requested_sdk_missing(self):
        with patch.dict(os.environ, {'SHMEM_ROOT': str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, 'header missing'):
                self.config()

    def test_manifest_records_headers_and_library(self):
        self.make_sdk()
        with patch.dict(os.environ, {'SHMEM_ROOT': str(self.root)}):
            config = self.config()
            self.assertTrue(config['enabled'])
            self.assertFalse(config['binary_source_revision_verified'])
            self.assertEqual(config['expected_source_revision'], '4dd80e2129da42bd773982c2bb3758af1ed0bba8')
            self.assertEqual(len(config['header_sha256_lf']), 7)
            self.assertEqual(config['library_sha256'], hashlib.sha256(b'mock shared library').hexdigest())

    def test_wrong_headers_rejected(self):
        self.make_sdk()
        (self.root / 'include/host/init/shmem_host_init.h').write_bytes(b'wrong SDK')
        with patch.dict(os.environ, {'SHMEM_ROOT': str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                self.config()

    def test_library_required(self):
        self.make_sdk()
        (self.root / 'lib/libshmem.so').unlink()
        with patch.dict(os.environ, {'SHMEM_ROOT': str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, 'library missing'):
                self.config()

    def test_library_digest_changes(self):
        self.make_sdk()
        with patch.dict(os.environ, {'SHMEM_ROOT': str(self.root)}):
            original = self.config()['library_sha256']
            (self.root / 'lib/libshmem.so').write_bytes(b'changed library')
            self.assertNotEqual(original, self.config()['library_sha256'])

    def test_host_session(self):
        compiler = os.environ.get('CXX') or shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            self.skipTest('host C++ compiler unavailable')
        headers = ('acl/acl.h', 'hccl/hccl.h', 'hccl/hccl_comm.h',
                   'torch_npu/csrc/core/npu/NPUStream.h', 'host/init/shmem_host_init.h',
                   'host/mem/shmem_host_heap.h', 'host/data_plane/shmem_host_cc.h',
                   'host/data_plane/shmem_host_rma.h')
        for name in headers:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('#pragma once\n', encoding='utf-8')
        environment = os.environ.copy()
        environment['PATH'] = str(Path(compiler).parent) + os.pathsep + environment.get('PATH', '')
        for enabled in (False, True):
            executable = self.root / ('host_enabled.exe' if enabled else 'host_disabled.exe')
            command = [compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                       '-I', str(self.root), str(ROOT / 'tests/ep/shmem_host_test.cpp'), '-o', str(executable)]
            if enabled:
                command.append('-DDEEPEP_WITH_SHMEM=1')
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cases = ('lifecycle', 'shared_failure', 'foreign', 'group', 'memory', 'capacity', 'peer_reject', 'size_mismatch',
                     'uid', 'attributes', 'init_failure', 'mapping_failure', 'remote_init_failure',
                     'abandon', 'collective', 'wait',
                     'close_order', 'finalize') if enabled else ('disabled',)
            cases += ('capability', 'mixed_capability', 'capability_error')
            for case in cases:
                with self.subTest(enabled=enabled, case=case):
                    result = subprocess.run([str(executable), case], capture_output=True, text=True,
                                            env=environment, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
