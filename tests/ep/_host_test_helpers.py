"""Helpers for running buffer and kernel tests on the host."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def run_host_cpp(source, *, include_dirs=(), headers=None):
    """Build and run a C++ test in an isolated directory and return its stdout."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for name, content in (headers or {}).items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        cpp, exe = temporary / 'test.cpp', temporary / 'test.exe'
        cpp.write_text(source, encoding='utf-8')
        command = [os.environ.get('CXX', 'g++'), '-std=c++20', '-O2', '-Wall', '-Wextra']
        for path in (temporary, *include_dirs):
            command.extend(['-I', str(path)])
        subprocess.run([*command, str(cpp), '-o', str(exe)], check=True)
        return subprocess.run([str(exe)], check=True, stdout=subprocess.PIPE, text=True).stdout


class Tensor:
    def __init__(self, shape, dtype='torch.bfloat16'):
        self.shape, self.dtype = tuple(shape), dtype

    def size(self, index):
        return self.shape[index]

    def is_inference(self):
        return True


class Event:
    def __init__(self):
        self.waits = 0

    def current_stream_wait(self):
        self.waits += 1


def fake_modules():
    torch = ModuleType('torch')
    torch.Tensor = Tensor
    for dtype in ('bfloat16', 'float16', 'float8_e4m3fn', 'int64', 'int32', 'float32', 'uint8'):
        setattr(torch, dtype, 'torch.' + dtype)
    torch.npu = SimpleNamespace()
    torch.distributed = ModuleType('torch.distributed')
    torch.distributed.ProcessGroup = object
    native = ModuleType('deep_ep._C')
    native.EventHandle = Event
    modules = {name: ModuleType(name) for name in (
        'deep_ep', 'deep_ep.buffers', 'deep_ep.comm', 'deep_ep.utils',
        'deep_ep.buffers.allocator', 'deep_ep.utils.math', 'deep_ep.utils.semantic')}
    modules.update({'torch': torch, 'torch.distributed': torch.distributed, 'deep_ep._C': native})
    modules['deep_ep']._C = native
    modules['deep_ep'].comm = modules['deep_ep.comm']
    for name in ('barrier', 'get_comm_stream', 'get_physical_domain_size', 'get_logical_domain_size'):
        setattr(modules['deep_ep.comm'], name, lambda *args: None)
    modules['deep_ep.buffers.allocator'].BufferAllocator = type('BufferAllocator', (), {})
    modules['deep_ep.utils.math'].align = lambda a, b: (a + b - 1) // b * b
    modules['deep_ep.utils.math'].ceil_div = lambda a, b: (a + b - 1) // b
    modules['deep_ep.utils.semantic'].value_or = lambda x, default: default if x is None else x
    return modules, torch, native


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
