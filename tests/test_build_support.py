"""CPU-only checks for SDK paths, build entries, and persistent configuration.

Run with `python -B -m unittest discover -s tests -p test_build_support.py`.
No compiler, Torch, CANN, network, or NPU is used. Build entries run only in
isolated temporary fixtures; the checkout's build artifacts are never removed.
"""

from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import build_support as support


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class BuildSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def toolkit(root):
        for relative in ('include', 'asc/include', 'lib64'):
            (root / relative).mkdir(parents=True, exist_ok=True)

    def test_native_architecture_and_aliases(self):
        self.toolkit(self.root)
        for host, sdk in (('x86_64', 'x86_64'), ('AMD64', 'x86_64'),
                          ('aarch64', 'aarch64'), ('arm64', 'aarch64')):
            native = self.root / f'{sdk}-linux'
            self.toolkit(native)
            with self.subTest(host=host), patch.object(support.platform, 'machine', return_value=host):
                includes, library = support.get_ascend_build_paths(self.root)
                self.assertEqual(includes, [str(native / name) for name in ('include', 'asc', 'asc/include')])
                self.assertEqual(library, str(native / 'lib64'))

    def test_canonical_root_fallback(self):
        self.toolkit(self.root)
        with patch.object(support.platform, 'machine', return_value='x86_64'):
            includes, library = support.get_ascend_build_paths(self.root)
        self.assertEqual(includes[0], str(self.root / 'include'))
        self.assertEqual(library, str(self.root / 'lib64'))

    def test_wrong_cpu_tree_is_rejected(self):
        self.toolkit(self.root / 'aarch64-linux')
        with patch.object(support.platform, 'machine', return_value='x86_64'):
            with self.assertRaisesRegex(RuntimeError, 'host CPU x86_64'):
                support.get_ascend_build_paths(self.root)

    def fixture(self, text=None):
        text = text if text is not None else '#pragma once\n' + support.DEEP_JIT_TOOLKIT_ANCHOR + '\n'
        source = self.root / 'source/include'
        backend = source / support.DEEP_JIT_BACKEND_PATH
        backend.parent.mkdir(parents=True)
        backend.write_text(text, encoding='utf-8')
        (backend.parent / 'device.hpp').write_text('// sibling include\n', encoding='utf-8')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return source, self.root / 'build/include', text, digest

    def test_overlay_is_isolated_and_repeatable(self):
        source, output, original, digest = self.fixture()
        with patch.object(support, 'DEEP_JIT_BACKEND_SHA256', digest):
            support.prepare_deep_jit_headers(source, output)
            first = (output / support.DEEP_JIT_BACKEND_PATH).read_bytes()
            support.prepare_deep_jit_headers(source, output)
            self.assertEqual((output / support.DEEP_JIT_BACKEND_PATH).read_bytes(), first)
        self.assertEqual((source / support.DEEP_JIT_BACKEND_PATH).read_text(encoding='utf-8'), original)
        self.assertTrue((output / support.DEEP_JIT_BACKEND_PATH.parent / 'device.hpp').is_file())
        self.assertIn(b'home_path / "asc/include/adv_api"', first)
        self.assertIn(b'#if defined(__x86_64__)', first)
        self.assertIn(b'#elif defined(__aarch64__)', first)

    def test_unknown_header_version_is_rejected(self):
        source, output, original, digest = self.fixture()
        (source / support.DEEP_JIT_BACKEND_PATH).write_text(original + '// changed\n', encoding='utf-8')
        with patch.object(support, 'DEEP_JIT_BACKEND_SHA256', digest):
            with self.assertRaisesRegex(RuntimeError, 'differs from pinned'):
                support.prepare_deep_jit_headers(source, output)
        self.assertFalse(output.exists())

    def test_exact_anchor_is_required(self):
        source, output, _, digest = self.fixture('// different toolkit locator\n')
        with patch.object(support, 'DEEP_JIT_BACKEND_SHA256', digest):
            with self.assertRaisesRegex(RuntimeError, 'anchor did not match exactly once'):
                support.prepare_deep_jit_headers(source, output)

    def test_missing_submodule_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, 'initialize third-party/deep_jit'):
            support.prepare_deep_jit_headers(self.root / 'missing', self.root / 'build')


def find_bash():
    found = shutil.which('bash')
    if found:
        return found
    candidate = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe'
    return str(candidate) if candidate.is_file() else None


@unittest.skipUnless(find_bash(), 'Bash is needed for isolated shell-entry tests')
class BuildEntryTests(unittest.TestCase):
    def run_entry(self, name, python_body):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
            script = root / name
            script.write_text((ROOT / name).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
            (root / 'deep_ep').mkdir()
            shell = 'export PATH="/usr/bin:/bin:$PATH"\npython() {\n' + python_body + '\n}\nsource "$1"'
            result = subprocess.run([find_bash(), '--noprofile', '--norc', '-c', shell,
                                     'build-entry-test', script.as_posix()],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            linked = list((root / 'deep_ep').iterdir())
            return result, [path.name for path in linked]

    def test_wheel_build_failure_is_not_masked_by_final_cd(self):
        result, _ = self.run_entry('build.sh', 'return 23')
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)

    def test_development_failure_does_not_link_partial_extension(self):
        result, linked = self.run_entry('develop.sh',
            'mkdir -p build/lib\nprintf partial > build/lib/_C.partial.so\nreturn 23')
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertEqual(linked, [])

    def test_successful_wheel_entry_returns_success(self):
        result, _ = self.run_entry('build.sh', 'test "$1" = setup.py && test "$2" = bdist_wheel')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_successful_development_entry_links_extension(self):
        result, linked = self.run_entry('develop.sh',
            'test "$1" = setup.py && test "$2" = build || return 29\n'
            'mkdir -p build/lib\nprintf fixture > build/lib/_C.fixture.so')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(linked, ['_C.fixture.so'])


class PersistentEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import setuptools
            from setuptools.command.build_ext import build_ext
        except ImportError as error:
            raise unittest.SkipTest('setuptools is required for the build configuration check') from error

        # Capture the real build_py class without compiling or packaging anything.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            toolkit = root / 'toolkit'
            for relative in ('include', 'asc/include', 'lib64'):
                (toolkit / relative).mkdir(parents=True, exist_ok=True)

            modules = {name: ModuleType(name) for name in
                       ('torch_npu', 'torch', 'torch.utils', 'torch.utils.cpp_extension')}
            modules['torch_npu'].__file__ = str(root / 'torch_npu/__init__.py')
            extension = modules['torch.utils.cpp_extension']
            extension.BuildExtension = build_ext
            extension.include_paths = lambda: []
            extension.library_paths = lambda: []
            captured = {}

            def capture_setup(**kwargs):
                captured.update(kwargs)

            original_check_output = subprocess.check_output

            def git_output(args, **kwargs):
                if args[0] != 'git':
                    return original_check_output(args, **kwargs)
                return b'' if args[1] == 'status' else b'fixture\n'

            with working_directory(ROOT), patch.dict(sys.modules, modules), \
                    patch.dict(os.environ, {'ASCEND_HOME_PATH': str(toolkit)}), \
                    patch.object(setuptools, 'setup', side_effect=capture_setup), \
                    patch.object(subprocess, 'check_output', side_effect=git_output), \
                    redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / 'setup.py'), run_name='__main__')
            cls.build_py_class = captured['cmdclass']['build_py']

    def test_persistent_environment_values_round_trip_as_data(self):
        from setuptools.dist import Distribution
        value = "/tmp/user's cache\\nested\nsecond-line"
        with tempfile.TemporaryDirectory() as temporary:
            command = self.build_py_class(Distribution())
            command.build_lib = temporary
            with patch.dict(os.environ, {'EP_JIT_CACHE_DIR': value}):
                command.generate_default_envs()
            generated = (Path(temporary) / 'deep_ep/envs.py').read_text(encoding='utf-8')
            namespace = {}
            exec(compile(generated, 'generated-envs.py', 'exec'), namespace)
            self.assertEqual(namespace['persistent_envs']['EP_JIT_CACHE_DIR'], value)


if __name__ == '__main__':
    unittest.main()
