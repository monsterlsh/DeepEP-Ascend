import ast
import hashlib
import json
import re
import os
import shutil
import subprocess
import setuptools
import torch_npu

from pathlib import Path
from setuptools import Extension
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import include_paths as torch_include_paths
from torch.utils.cpp_extension import library_paths as torch_library_paths

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_env_names = (
    'EP_JIT_CACHE_DIR',
    'EP_JIT_DEBUG',
    'EP_JIT_DUMP_ASM',
    'EP_JIT_KERNEL_DEBUG_INFO',
    'EP_JIT_LAUNCH_TIMEOUT',
    'EP_JIT_PRINT_COMPILER_COMMAND',
    'EP_JIT_PRINT_LOAD_TIME',
)

shmem_revision = '4dd80e2129da42bd773982c2bb3758af1ed0bba8'
shmem_header_hashes = {
    'host/shmem_host_def.h': '659aaa719af9c38bafef72921a73cca325c3c30f0f43be30b6ba9f0535de6cf6',
    'host/init/shmem_host_init.h': '9f17760542e9f3b6dfa8bd6aad724250f686233398b5b2043c7d28f66d6f5fbb',
    'host/mem/shmem_host_heap.h': '5ef864fd9c6c812169de6e1dfbbb6b55481cd9cb67e543a681c3e19d5ce627b4',
    'host/data_plane/shmem_host_cc.h': 'dc8704f0d3fe00abe99ad5c9fa0bb74a79f7cda8c2af9242dc717cdd9f45ceb6',
    'host_device/shmem_common_types.h': '528cd9698815312358253c40b74f4f0668b86af637843f9f767f6d625c273a34',
    'host_device/shmem_common_macros.h': 'd1540f47e041ddc544a975ddf9fda2e9801c5243d5730a5966598229adb7af60',
}


def get_shmem_build_config():
    explicit_root = os.environ.get('SHMEM_ROOT')
    root = Path(explicit_root or Path(current_dir) / 'third-party' / 'shmem' / 'install' / 'shmem')
    if not explicit_root and not root.exists():
        return {'enabled': False}
    root = root.resolve()
    hashes = {}
    # The RMA header is required and recorded, but has no pinned digest.
    for name in (*shmem_header_hashes, 'host/data_plane/shmem_host_rma.h'):
        path = root / 'include' / name
        if not path.is_file():
            raise RuntimeError(f'SHMEM SDK header missing: {path}')
        hashes[name] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if name in shmem_header_hashes and hashes[name] != shmem_header_hashes[name]:
            raise RuntimeError(f'SHMEM SDK header does not match {shmem_revision}: {path}')
    for lib_dir in ('lib', 'lib64'):
        library = root / lib_dir / 'libshmem.so'
        if library.is_file():
            break
    else:
        raise RuntimeError(f'SHMEM SDK library missing under {root}: lib/libshmem.so or lib64/libshmem.so')
    return {
        'enabled': True,
        'root': str(root),
        'expected_source_revision': shmem_revision,
        'header_sha256_lf': hashes,
        'library': str(library),
        'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
        'binary_source_revision_verified': False,
    }


def get_package_version():
    version_file = Path(current_dir) / 'deep_ep' / '__init__.py'
    with open(version_file, 'r') as f:
        content = f.read()
    version_match = re.search(r'^__version__\s*=\s*(.*)$', content, re.MULTILINE)
    if version_match:
        public_version = ast.literal_eval(version_match.group(1))
    else:
        public_version = '0.1.0'

    # noinspection PyBroadException
    try:
        status_output = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        if status_output:
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
            assert False, 'Git working directory is not clean'

        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception:
        revision = '+local'
    return f'{public_version}{revision}'


def get_ascend_home():
    ascend_home = os.environ.get('ASCEND_HOME_PATH') or os.environ.get(
        'ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest')
    if not os.path.isdir(ascend_home):
        ascend_home = '/usr/local/Ascend/cann'
    if not os.path.isdir(ascend_home):
        raise RuntimeError(
            'Cannot find ASCEND_HOME_PATH. Please set the ASCEND_HOME_PATH environment variable '
            'to your CANN installation directory.'
        )
    return ascend_home


class CustomBuildPy(build_py):
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        self.generate_default_envs()
        self.generate_shmem_manifest()

        # Then, copy csrc into the wheel for agent-side error lookup
        self.prepare_agent_files()

        # Finally, run the regular build
        build_py.run(self)

    def generate_shmem_manifest(self):
        path = Path(self.build_lib) / 'deep_ep' / 'shmem-sdk.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(get_shmem_build_config(), indent=2) + '\n', encoding='utf-8')

    def prepare_agent_files(self):
        # Copy csrc into the wheel for agent-side error lookup
        package_dir = os.path.join(self.build_lib, 'deep_ep')
        for name in ('csrc', ):
            dst = os.path.join(package_dir, name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(current_dir, name), dst)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in persistent_env_names:
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        # Create temporary build directory
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')
        os.makedirs(build_include_dir, exist_ok=True)
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:
            f.write(code)


if __name__ == '__main__':
    ascend_home = get_ascend_home()
    torch_npu_dir = os.path.dirname(torch_npu.__file__)
    package_dir = Path(current_dir) / 'deep_ep'

    sources = ['csrc/python_api.cpp']

    cxx_flags = [
        '-O3', '-g1', '-fPIC', '-std=c++20',
        '-Wno-deprecated-declarations',
        '-Wno-unused-variable',
        '-Wno-sign-compare',
        '-Wno-reorder',
        '-Wno-attributes',
    ]

    include_dirs = [
        f'{current_dir}/csrc',
        f'{current_dir}/deep_ep/include',
        f'{current_dir}/third-party/deep_jit/include',
        f'{ascend_home}/aarch64-linux/include',
        # TODO(HUAWEI): cleanup include paths
        f'{ascend_home}/aarch64-linux/asc',
        f'{ascend_home}/aarch64-linux/asc/include',
        f'{torch_npu_dir}/include',
        f'{torch_npu_dir}/include/third_party/acl/inc',
    ] + torch_include_paths()

    library_dirs = [
        f'{ascend_home}/lib64',
        f'{torch_npu_dir}/lib',
    ] + torch_library_paths()

    libraries = ['ascendcl', 'hccl', 'hcomm', 'torch_npu']
    extra_link_args = [
        f'-Wl,-rpath,{ascend_home}/lib64',
        '-Wl,-Bsymbolic-functions',
        '-Wl,--enable-new-dtags',
    ]
    shmem_config = get_shmem_build_config()
    define_macros = []
    if shmem_config['enabled']:
        include_dirs.append(str(Path(shmem_config['root']) / 'include'))
        shmem_libdir = str(Path(shmem_config['library']).parent)
        library_dirs.append(shmem_libdir)
        libraries.append('shmem')
        extra_link_args.append(f'-Wl,-rpath,{shmem_libdir}')
        define_macros.append(('DEEPEP_WITH_SHMEM', '1'))

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Library dirs: {library_dirs}')
    print(f' > Libraries: {libraries}')
    print(f' > CXX flags: {cxx_flags}')
    print(f' > ASCEND_HOME_PATH: {ascend_home}')
    print(f' > TORCH_NPU_DIR: {torch_npu_dir}')
    print(f' > SHMEM: {json.dumps(shmem_config, sort_keys=True)}')
    # Print persistent env variables
    persistent_envs = []
    for name in persistent_env_names:
        if name in os.environ:
            persistent_envs.append((name, os.environ[name]))
    if len(persistent_envs) > 0:
        print(f' > Persistent envs:')
        for k, v in persistent_envs:
            print(f'   > {k}: {v}')
    print()

    ext_module = Extension(
        name='deep_ep._C',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=cxx_flags,
        extra_link_args=extra_link_args,
        language='c++',
    )

    # noinspection bad-argument-type
    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        # Older setuptools versions do not expand recursive package-data globs.
        package_data={'deep_ep': [str(path.relative_to(package_dir)) for path in (package_dir / 'include').rglob('*.hpp')]},
        ext_modules=[ext_module],
        cmdclass={'build_ext': BuildExtension, 'build_py': CustomBuildPy},
    )
