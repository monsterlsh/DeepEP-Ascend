# T08：dispatch 融合 + 服务器级去重接入默认接口（实现方案）

> 状态：实现方案（2026-10-22），待实施
> 前置：T07 设计（`docs/design/ll-default-integration.md`）

## 1. 调用链与接口映射

现状调用链（显式 LL）：
```
low_latency.py :: LowLatencyState.dispatch
  → _C.dispatch_low_latency(buffer.runtime, tensors, b, h, k, e, c, dtype)
    → csrc/buffers/low_latency_api.hpp  lambda
      → csrc/kernels/ep/low_latency.hpp :: launch_dispatch_low_latency(EpServerDedupKernelArgs)
        → DeepJIT compile/launch dispatch_server_dedup_kernel (num_blocks=64)
```

默认调用链（待接入目标）：
```
ep.py :: EPBuffer.dispatch
  → runtime.dispatch(...) → csrc/buffers/ep.hpp :: dispatch(...)
    → launch_dispatch(...) → dispatch_impl (native MTE/URMA)
    → epilogue: launch_dispatch_copy_epilogue → 标准 recv_x / recv_src_metadata / handle
```

## 2. 关键差异（决定桥接形态）

| 输出 | 默认 dispatch | LL dispatch |
| --- | --- | --- |
| 数据 | `recv_x` `[num_expanded_tokens, hidden]`（官方展开布局） | `expandXOut`（服务器去重紧凑布局，`capacity_rows` 行） |
| 路由元数据 | `recv_src_metadata` `[num_recv_tokens, topk+2]` | `destinationIndexOut` `[rows,3]`、`relayReadIndexOut` `[rows,k,2]`、`tokenTypeOut` `[rows]`、`sourceMaskOut` `[B]` |
| 计数 | `psum_num_recv_tokens_per_expert`、`num_unaligned_recv_tokens_per_expert` | `expertTokenNumsOut`、`sendCountsOut` |

结论：LL 输出的物理布局与官方不同。**不能**在 LL kernel 之后直接复用 `dispatch_copy_epilogue` 消费，需要在 host epilogue 侧按官方布局**重建** `recv_x` / `recv_src_metadata`，或新增一个「LL→官方布局」的展开内核。

## 3. 实施步骤

### 步骤 1：准入判定（`csrc/buffers/ep.hpp`，`dispatch()` 内）
新增 LL 去重准入分支，条件（全 rank 确定性）：
- `EP_ENABLE_LL_SERVER_DEDUP == "1"`（显式开关，默认关）
- 非 cached、非 FP8 非预量化、`expert_alignment == 1`、无 `cumulative_local_expert_recv_stats`
- 形状满足 LL tiling 约束（`0 < B ≤ M ≤ 256`、H/K/E 匹配、32 AI / 64 AIV）
- `defer_epilogue == False`、`barrier_in_prologue` 兼容
- 路由合法性（步骤 2）通过

### 步骤 2：路由合法性扫描（D06）
新增轻量设备内核 `dispatch_route_check`（或复用现有 host 端 Python 检查，但需覆盖 masked/-1 与 token 内重复专家）。产出 `route_ok` 标量；`route_ok=false` 回退 URMA。回退在通信开始前决定（D08）。

### 步骤 3：构造 `EpServerDedupKernelArgs` 并 launch
复用 `launch_dispatch_low_latency`，但输入取自默认 `dispatch()` 的 `x`/`topk_idx`/`topk_weights`。注意：LL 要求 `topk_idx` 转 int32（默认是 int64）。

### 步骤 4：epilogue 重建官方布局
LL kernel 后，用新增「LL→官方」展开逻辑（或 host 端 + 小内核）：
- 依据 `destinationIndex`/`relayReadIndex`/`tokenType` 把 `expandXOut` 还原成官方 `recv_x` 展开顺序；
- 依据 `sourceMask` 还原 `recv_src_metadata` 的 `[src_rank, src_idx]` 与 `topk` 来源列；
- 用 `expertTokenNumsOut`/`sendCountsOut` 填充标准 handle 的 `psum_num_recv_tokens_per_expert`、`num_unaligned_recv_tokens_per_expert`、`num_recv_tokens_per_expert_list`（可复用 D2H 回读，见 CPU-sync 修复）。

### 步骤 5：兼容回退
不满足准入的调用，全部走现有 native MTE/URMA 路径；`EP_ENABLE_LL_SERVER_DEDUP` 只开启 LL 去重，不与 `EP_DISABLE_LOW_LATENCY_TRANSPORT` 冲突。

## 4. 风险与待确认
- **布局重建的正确性**是最大风险：官方展开布局的 token 顺序必须与 `recv_src_metadata` 严格一致，否则 combine 无法正确消费。建议先写一个 Python 参考实现（对照 `low_latency_reference.py` 扩展）验证映射关系。
- LL 的 `capacity_rows` 与官方 `num_expanded_tokens` 的对应关系需精确核对（重复专家/去重后的行数差异）。
- 服务器级去重只在 EP≥32 多机才有收益（EP8 单机无 FIRST_HOP relay），性能验证需多机环境。
