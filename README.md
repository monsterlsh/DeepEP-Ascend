# DeepEP-Ascend

**（中文介绍）**[DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) 是面向华为昇腾 NPU 的高性能机器学习训练与推理通信库，提供 MoE dispatch/combine 的专家并行（EP）all-to-all 操作，支持 FP8 dispatch 和延迟 epilogue。此外还提供流水线并行（PP）、面向上下文并行和数据并行（CP/DP）的 Bucket 集合通信，以及 Engram 远端内存访问等通信原语（开发中）。其公开 buffer API 与 NVIDIA 版 [DeepEP](https://github.com/deepseek-ai/DeepEP) 对齐。Ascend C 内核使用 HCCL/HCOMM、UBMEM 和 URMA 完成通信，并通过 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 在运行时编译。

**(English introduction)** [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) is a high-performance communication library for machine learning training and inference on Huawei Ascend NPUs. It provides expert-parallel (EP) all-to-all operations for MoE dispatch and combine, including FP8 dispatch and deferred epilogues. It also offers communication primitives (work in progress) for pipeline parallelism (PP), context and data parallelism through Bucket collectives (CP/DP), and remote memory access through Engram. Its public buffer APIs are aligned with the NVIDIA version of [DeepEP](https://github.com/deepseek-ai/DeepEP). Ascend C kernels use HCCL/HCOMM, UBMEM and URMA, and are compiled at runtime through [DeepJIT](https://github.com/deepseek-ai/DeepJIT).

## Performance

The following prefill measurements predate the [capacity-based decode implementation](#capacity-based-ep-dispatch-and-combine).

Measured on Ascend 950DT NPUs with CANN 9.2.0 and the manually configured PoC HDK described under [Recommended HDK and firmware](#recommended-hdk-and-firmware). All measurements use netlayer 1, the supernode's external Clos network.

The following runs use 16,384 tokens/rank capacity, hidden size 7168, top-6 routing over 256 experts, 32 AI cores / 64 AIVs, expert alignment 128 and no bias. Dispatch uses FP8 with row-major scales; combine uses BF16. Each rank uses 10 warmups and 50 samples with cache flushing.

| EP size | Dispatch (GB/s) | Combine (GB/s) |
| --- | ---: | ---: |
| EP8 | 373–375 | 345–347 |
| EP16 | 348–352 | 338–341 |
| EP32 | 335–340 | 320–324 |
| EP64 | 323–327 | 294–298 |
| EP128 | 313–320 | 272–278 |

Bandwidth ranges cover all ranks, using timings that include issue and drain but exclude final epilogues. Sustained dispatch reaches roughly 90–95% of the physical payload bandwidth limit for EP sizes up to 32. Larger EP sizes and combine remain under optimization; combine has additional local reduction overhead and HBM contention with URMA. Huawei firmware improvements are also planned to reduce this contention.

## Quick start

### Requirements

- Linux on an Ascend host with supported NPU drivers and firmware.
- Ascend 950 (A5) with UBMEM connectivity and UBC_CTP/URMA channels between the participating ranks.
- An even number of ranks for multi-rank communication.
- CANN with Ascend C, the Bisheng compiler, HCCL and HCOMM headers/libraries.
- Python 3.10 or newer, matching CPU-build PyTorch/`torch_npu` packages, and NumPy for the tests.
- A C++20 compiler and standard library supporting `std::format`. For GCC/libstdc++, use GCC 13 or newer with matching development headers and runtime libraries ([GCC 13 release notes](https://gcc.gnu.org/gcc-13/changes.html)).
- elfutils development headers, including `elfutils/libdwfl.h`, for DeepJIT's host-side exception handling.

The validated stack is Ascend 950DT, CANN 9.2.0, Python 3.12, PyTorch 2.13.0+cpu and torch_npu 2.13.0rc1, with the PoC HDK configuration described below. Kernel support on other Ascend generations or CANN versions has not been established by these measurements.

Installation builds the host extension; device kernels are compiled on first use. Keep the CANN toolkit and Bisheng available at runtime. Set `ASCEND_HOME_PATH` to the active CANN installation, or use `ASCEND_TOOLKIT_HOME`.

Use the CPU build of PyTorch with `torch_npu`. In the Python 3.11 build check,
PyTorch `2.12.0+cu130` and `torch_npu 2.12.0.post1` failed during import because
both registered the same allocator configuration keys. Using `2.12.0+cpu`
resolved that failure; disabling backend autoload alone did not.

### Recommended HDK and firmware

Use Huawei's Q3 commercial HDK release for Atlas 850E when it becomes publicly available. Huawei has advised that this release includes the configuration required for full-bandwidth operation. Public availability is currently planned for mid-October 2026, around October 15, through the [Huawei Atlas 850E software download page](https://support.huawei.com/enterprise/zh/atlas-computing/atlas-850e-pid-265102853/software), where users can apply for access when the release becomes available. The date is a vendor release plan; availability is subject to Huawei's publication schedule.

The performance results in this README were collected using a PoC HDK supplied to DeepSeek, with additional manual configuration. That configuration is not a publicly distributed release. Users on earlier PoC versions may observe lower bandwidth and should first check their HDK version and configuration. Use the Q3 commercial release, once available, as the recommended public deployment baseline; these measurements are not results from that unreleased commercial version.

### Installation

Install the elfutils development packages in the host or container where you build the extension. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libdw-dev libelf-dev
```

On openEuler/Fedora/RHEL, use the corresponding packages:

```bash
sudo dnf install -y elfutils-devel elfutils-libelf-devel
```

On openEuler 24.03 LTS SP4, enable [GCC Toolset 14](https://docs.openeuler.org/zh/docs/25.03/server/development/gcc/gcc-14-secondary-version-compilation-toolchain-user-guide.html) for C++20 `<format>` support. The C++ package in the [SP4 aarch64 repository](https://repo.openeuler.org/openEuler-24.03-LTS-SP4/everything/aarch64/Packages/) is named `gcc-toolset-14-gcc-c++`:

```bash
sudo dnf install -y gcc-toolset-14-gcc gcc-toolset-14-gcc-c++ \
    gcc-toolset-14-binutils gcc-toolset-14-libstdc++-devel
source /opt/openEuler/gcc-toolset-14/enable
export CC=/opt/openEuler/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/openEuler/gcc-toolset-14/root/usr/bin/g++
```

Activate the CANN environment and install the matching PyTorch/`torch_npu` packages first, then:

```bash
git clone https://github.com/deepseek-ai/DeepEP-Ascend.git
cd DeepEP-Ascend

# Fetch the required DeepJIT submodule over HTTPS.
git submodule update --init --recursive third-party/deep_jit

python -m pip install --no-build-isolation .
```

For development, use `bash develop.sh` to build and link the extension into the source tree. The `clangd-ascend` submodule and CMake configuration are for IDE/debugging support.

## Features and API compatibility

Training, inference prefill and decoding share the same `EPBuffer` interface and the `deep_ep` Python package name. The implementation follows DeepEP's EPBuffer-based V2.5 APIs; supported modes and stream behavior are specific to Ascend.

| Interface | Ascend support |
| --- | --- |
| `EPBuffer` | Expanded BF16/FP8 dispatch and BF16 combine, FP16 dispatch/combine at capacities up to 16, cached handles, deterministic layouts, expert padding, deferred epilogues and native MTE transport with optional cross-server deduplication |
| `PPBuffer` | Send/receive between adjacent pipeline ranks |
| `EngramBuffer` | NPU-backed BF16/FP8 tables and per-layer fetch hooks |
| `BucketBuffer` | Batched all-gather, registered storage, multiple groups and sessions for ordinary tensors |
| `BufferAllocator`, `BufferBase`, `deep_ep.comm` | Allocation planning, buffer lifecycle, communicator queries, barriers and stream access |

PP, Engram and Bucket are experimental; remaining work is listed under [Ongoing](#ongoing). EP requires `do_expand=True` and `allow_multiple_reduction=True`; `do_handle_copy` must remain `False`.

Routing indices use `deep_ep.topk_idx_t`, currently fixed to `torch.int64`. Explicit RDMA service levels and QP counts are unsupported; leave `sl_idx=None` and QP arguments at their defaults. The NV-compatible `num_sms` argument selects AI cores on Ascend EP, with `0` selecting the default; cached dispatch and combine reuse the count saved in the handle.

## Ongoing

The Ascend backend is under active development:

- Bucket collectives: batched all-gather is available; Ascend reduce-scatter and all-reduce kernels are still being built.
- Expert load balancing: `EPBuffer.lb_prefetch_weights` and `EPBuffer.lb_reduce_grads` expose the aligned APIs, but their Ascend communication kernels are not implemented yet.
- PP and Engram: these interfaces remain experimental, with further correctness and performance validation ongoing.

Hybrid communication, CPU-backed Engram storage and graph capture remain unsupported.

## EP usage in training and inference

### Buffer initialization

Create one long-lived buffer per EP group and reuse it across layers or microbatches. All ranks must agree on the token capacity. Complete outstanding operations before resizing, reusing or destroying a buffer.

```python
import torch
import torch.distributed as dist
import torch_npu

from deep_ep import EPBuffer

# First select the local NPU and initialize torch.distributed with backend='hccl'.
ep_group = dist.group.WORLD
max_tokens_per_rank = 4096
hidden = 7168
num_topk = 6
num_experts = 256

buffer = EPBuffer(
    ep_group,
    num_max_tokens_per_rank=max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=False,  # Reserve enough storage for BF16 and FP8 dispatch.
    explicitly_destroy=True,
)
```

Use `EPBuffer.get_buffer_size_hint()` to check capacity when managing a buffer pool. After all uses and their waits have finished, call `buffer.destroy()` before tearing down the process group.

### Dispatch and combine

Training, prefill and decoding use the same `EPBuffer` dispatch and combine APIs. The following helpers use the expanded expert layout and defer epilogues with `defer_epilogue=True` and `async_with_compute_stream=True`.

```python
def dispatch_forward(buffer, x, topk_idx, topk_weights, num_experts,
                     expert_alignment=1, do_cpu_sync=True):
    """Route BF16, FP16 or FP8 inputs to their selected experts."""
    return buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        expert_alignment=expert_alignment,  # Match the grouped GEMM's token alignment.
        do_cpu_sync=do_cpu_sync,  # False when expert kernels use device-side counts.
        do_expand=True,
        do_zero_padding=True,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )


def dispatch_backward(buffer, grad_recv_x, grad_recv_weights, handle, bias=None):
    """The backward of dispatch is a combine."""
    return buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_weights,
        bias=bias,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )


def combine_forward(buffer, expert_output, handle, bias=None):
    """Reduce BF16 or FP16 expert outputs back to their original ranks."""
    return buffer.combine(
        expert_output,
        handle=handle,
        bias=bias,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )


def combine_backward(buffer, grad_output, handle):
    """The backward of combine reuses the forward handle for dispatch."""
    return buffer.dispatch(
        grad_output,
        handle=handle,
        do_expand=True,
        do_zero_padding=True,
        async_with_compute_stream=True,
        defer_epilogue=True,
    )
```

Launch communication first, run independent computation, then call `.wait()` when its result is needed. EP kernels and their epilogues run on the current NPU stream:

```python
pending = dispatch_forward(
    buffer, x, topk_idx, topk_weights, num_experts,
    expert_alignment=expert_alignment,
    do_cpu_sync=need_cpu_expert_counts,
)

# Overlap application computation with communication.
shared_output = shared_experts(x)
recv_x, _, recv_weights, handle = pending.wait()

# Expert GEMMs apply recv_weights once and produce BF16 outputs.
expert_output, expert_ctx = local_experts_forward(recv_x, recv_weights, handle)

pending = combine_forward(buffer, expert_output, handle, bias=shared_output)
# ... other independent computation ...
output, _ = pending.wait()
```

Training saves `handle` for `combine_backward` and `dispatch_backward`; wait on their returned events in the same way. See [EP tests](tests/ep/test_ep.py) for complete examples.

## Capacity-based EP dispatch and combine

The effective `M = num_max_tokens_per_rank` selects the complete dispatch and combine implementation.
Fresh dispatch uses its explicit capacity or the buffer default. Cached dispatch and combine use the
capacity saved in the originating handle. All ranks use the same `M`, and their local token counts may
range from zero to `M`.

| Capacity per rank | Fresh dispatch, cached dispatch and combine |
|---|---|
| `1 ≤ M ≤ 16` | Decode implementation, including FP16 and extended FP8 scale formats. |
| `M > 16` | Original implementation from main, including BF16 and native FP8 dispatch and BF16 combine. |

For BF16 H7168/K6/E384 and alignment 128, the EP64 combine measurement at `M=16` was
55.479 µs on `7004d16` and 53.774 µs on `main@4dfccac` (+3.17%).

Within the decode implementation, MTE transport is available for the following configurations:

- Ascend950DT_9582 on A5, 32 AI cores (64 vector cores), 216 KiB of UB per vector core
  and the default 100-second device timeout.
- EP2, EP4, EP8, EP16, EP32, EP64 or EP128.
- BF16/FP16 activations with `H` divisible by 128, or FP8 E4M3 with `H` divisible by 256.
  FP8 dispatch carries native 16-bit scale packs, FP32 scales per 128 channels, or INT32 words
  containing four E8M0 scales per 128 channels. The returned scales preserve their dtype and bits.
- `1 ≤ K ≤ min(E, 32)`, `E ≤ 512`, and `E` evenly divisible by the EP group size. The
  dispatch stages and the combine reduction ring must fit UB. This covers H7168/K6/E384,
  as well as H7168/K8/E128. Alignment, host count sync, padding and epilogue scheduling
  keep their usual expanded dispatch semantics. Deterministic buffers use URMA.
- The expert computation applies the routing weight once to each selected expert output.
  Combine sums BF16 or FP16 expert rows. Its optional `topk_weights` argument restores the
  weights in their original routing slots. Bias is applied by the reduce epilogue.

Dispatch writes each token into the destination rank's registered EP communication buffer.
The native copy epilogue and cached handle use the same token layout as URMA.
All ranks use the same H/K/E, capacity, payload dtype and scale format. Use distinct expert
IDs within each token's top-k selection, as returned by `torch.topk`. Masked slots may
contain `-1`. With `L = E / world` local experts, a token contributes at most one row to
any expert and at most `min(K, L)` rows to a rank. A fresh dispatch without host count
synchronization reserves `world × M × min(K, L)` rows plus alignment space. Synchronized
dispatch uses the actual aligned expert counts, and cached dispatch reuses its handle's size.

For cross-server deduplication, set `EP_DECODE_SERVER_ID` on **each rank** before constructing
the `EPBuffer`. Use the same nonnegative integer for ranks on one physical server and different
integers for different servers. EP ranks may be interleaved. An incomplete or invalid map disables
server relaying for the group. With a multi-server group containing a shared server,
dispatch sends the full packet once per remote server and its other recipients fetch it within that server.
Combine always sends each expert rank's partial directly to the token owner when using MTE.
The owner applies biases and reduces rank partials in the same routing-slot order as URMA.
Each contributing rank returns its own optional weights directly to the corresponding slots.

Dispatch relay descriptors occupy reserved space in the existing registered EP workspace.
During buffer construction, ranks exchange server IDs, device capabilities and transport settings
alongside their communication ports.
Dispatch and combine select their transport from this group configuration and the shared launch
geometry. Use the same vector core count, fresh/cached mode and barrier placement on all ranks.
Empty ranks participate in the collective. Local token count, statistics, bias and deferred
execution do not enter transport selection. Graph capture remains unsupported. Parameter validation
errors detected on all ranks before submission can be corrected and retried on the same buffer.
If a rank has already entered the collective, a lone rank cannot retry it. When a submission fails
and work may still be outstanding, restart the distributed job. Releasing the failed buffer retains
its registered memory and outstanding URMA sources until process exit. Explicit destruction reports the failure to the caller.

Dispatch and combine both use the communication buffer allocated by `EPBuffer` and registered
with HCCL. Complete a deferred epilogue with `event.current_stream_wait()` before reusing
or destroying the buffer. If a call with `M > 16` does not request an event or defer its epilogue,
establish a stream dependency yourself before reusing that buffer on another stream. For example,
record an event on the producer stream immediately after the call and wait on it in the consumer stream.

Set `EP_DISABLE_LOW_LATENCY_TRANSPORT=1` before constructing the `EPBuffer` to use URMA in the
decode implementation. If any rank disables MTE, the group uses URMA.

For decode calls that return routing weights, pass the `recv_weights` from
`EPBuffer.dispatch` to `EPBuffer.combine` as `topk_weights`. Combine returns
these weights in the original token order. The expert computation applies the
weights to its outputs once, as in [Dispatch and combine](#dispatch-and-combine).
To compare decode correctness and latency with main, run the same EP test on each build.
Set `--num-tokens` to the per-rank capacity you want to measure.
The stock benchmark labels the communication-kernel time `urma:` for both transports.
Use the JIT specialization names (`dispatch_mte`, `dispatch_mte_relay`, `combine_mte`)
to confirm which MTE kernels were selected.

```bash
python tests/ep/test_ep.py --num-processes 8 --num-ai-cores 32 \
    --num-tokens 16 --hidden 7168 --num-topk 6 --num-experts 384 --dispatch-dtype bf16
```

Run [the decode regression](tests/ep/test_decode.py) for empty ranks, deferred combine
across streams, capacity overrides, FP16 activations and 32-bit FP8 scales. With at
least four ranks it also checks BF16 rounding and optional routing weights byte for byte.

## Other buffer interfaces

### Bucket all-gather

Bucket communication runs on the comm stream; its epilogue runs on the caller's current stream. Set `EP_AVOID_RECORD_STREAM=1` before use. A session permits ordinary contiguous NPU inputs:

```python
from deep_ep import BucketBuffer, get_num_allocation_alignment

shard = torch.full((1024,), ep_group.rank(), dtype=torch.float32, device='npu')
alignment = get_num_allocation_alignment()
gathered_bytes = ep_group.size() * shard.nbytes
num_bytes = (gathered_bytes + alignment - 1) // alignment * alignment
bucket = BucketBuffer(
    ep_group, num_bytes, explicitly_destroy=True)
with bucket.session():
    pending = bucket.all_gather(shard)
    gathered = pending.wait()  # Flat, rank-ordered view of session storage.
    # Consume or copy gathered before the session storage is reused.
bucket.destroy()
```

All ranks must supply equally sized shards, and the bucket must have enough space for the gathered output; sessions do not grow its storage. Outside a session, output storage must belong to the bucket; without explicit `dsts`, each input must be the local rank's shard of a registered gathered tensor. Use `BufferAllocator` to plan persistent registered storage. Batches support up to 64 tensors, and multi-group buffers select a group with `group=`. Session exit adds no barrier. See [all-gather tests](tests/bucket/test_all_gather.py) and [session tests](tests/bucket/test_session.py).

### Pipeline parallelism and Engram

`PPBuffer` reserves send/receive storage with `num_max_tensor_bytes` and `num_max_inflight_tensors`, then provides `send(x, dst_rank_idx)` and `recv(x, src_rank_idx)` for adjacent pipeline ranks. See [PP tests](tests/pp/test_pp.py).

`EngramBuffer` uses `get_storage_size_hint()`, `set_config()` and `write()` to prepare NPU-backed tables. `fetch(indices)` returns per-layer completion hooks; invoke each hook before consuming its result. See [Engram tests](tests/engram/test_engram.py), including FP8 mode. PP and Engram run entirely on the current stream.

## Environment and tests

Set environment variables consistently on every rank, before importing the package or constructing buffers.

| Variable | Use |
| --- | --- |
| `EP_AVOID_RECORD_STREAM` | Required for Bucket all-gather; EP retains URMA sources through epilogue captures independently of this flag |
| `EP_BUFFER_DEBUG` | Print EP buffer initialization diagnostics; unset to disable |
| `EP_DISABLE_BARRIER_PROFILING` | Disable the extra per-iteration synchronization used by benchmark helpers |

DeepJIT settings use `EP_JIT_*` for DeepEP, falling back to the corresponding `DJ_JIT_*` process-wide setting and then the built-in default. For example, `EP_JIT_CACHE_DIR` takes precedence over `DJ_JIT_CACHE_DIR`. The Ascend backend supports:

| Variable | Default | Use |
| --- | --- | --- |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | Cache root or colon-separated roots; search all roots in order and compile misses into the first |
| `EP_JIT_DEBUG` | `0` | Print compiler commands, binary paths and load timings |
| `EP_JIT_DUMP_ASM` | `0` | Save Ascend assembly under the cache entry's `asm/` directory on a cache miss |
| `EP_JIT_KERNEL_DEBUG_INFO` | `0` | Include kernel source and line information for debugging |
| `EP_JIT_LAUNCH_TIMEOUT` | `300` | Kernel launch timeout in seconds; `0` disables it |
| `EP_JIT_PRINT_COMPILER_COMMAND` | `0` | Print Bisheng and linker commands |
| `EP_JIT_PRINT_LOAD_TIME` | `0` | Print kernel binary load timings |

See [DeepJIT's environment reference](https://github.com/deepseek-ai/DeepJIT#environment-variables) for details.

To reproduce the default EP performance cases after activating the CANN environment:

```bash
export EP_AVOID_RECORD_STREAM=1
export TASK_QUEUE_ENABLE=0
export HCCL_IF_BASE_PORT=48361
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH=/tmp/deepep-ascend-logs
mkdir -p "$ASCEND_PROCESS_LOG_PATH"

python tests/ep/test_ep.py --num-processes 8 --num-ai-cores 32 \
    --num-tokens 16384 --hidden 7168 --num-topk 6 --num-experts 256 \
    --dispatch-dtype bf16 --test-first-only
python tests/ep/test_ep.py --num-processes 8 --num-ai-cores 32 \
    --num-tokens 16384 --hidden 7168 --num-topk 6 --num-experts 256 \
    --dispatch-dtype fp8 --test-first-only
```

For multiple nodes, launch the same command on each node with a shared `MASTER_ADDR` and `MASTER_PORT`, setting `WORLD_SIZE` to the number of nodes and `RANK` to the node index. Omit `--test-first-only` and the dtype filter for the full mode matrix. Additional checks include:

```bash
python tests/buffer/test_allocation.py
python tests/comm/test_stream.py
python tests/comm/test_barrier.py
python tests/bucket/test_all_gather.py
python tests/bucket/test_session.py
python tests/pp/test_pp.py
python tests/engram/test_engram.py
python tests/engram/test_engram.py --use-fp8
python tests/utils/test_gate.py
```

The test helpers in [envs.py](deep_ep/utils/envs.py) interpret `WORLD_SIZE` and `RANK` as node count and node index, and spawn `--num-processes` local workers. Multi-node invocations also require a shared `MASTER_ADDR` and `MASTER_PORT`; these launch conventions do not imply that every topology is supported by the current backend.

## Acknowledgement

DeepEP-Ascend is built on top of Huawei's HCCL/HCOMM, UBMEM and URMA communication stack. Thanks to the asc-comm team for their support!

## Citation

```bibtex
@misc{deepepascend2026,
    title = {{DeepEP-Ascend}: an efficient expert-parallel communication library for {Ascend} {NPUs}},
    author = {Chenggang Zhao and Shangyan Zhou and Kexing Zhou and Rui Tian and Chenqi Zhao and Chenhao Xu and Yizhi Wang and Kuai Yu},
    year = {2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/deepseek-ai/DeepEP-Ascend}},
}
```
