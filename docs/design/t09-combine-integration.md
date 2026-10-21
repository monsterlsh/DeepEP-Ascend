# T09：combine 聚合 + 去重 + 融合接入默认接口（实现方案）

> 状态：实现方案（2026-10-22），待实施
> 前置：T07 设计、T08 dispatch 接入（LL 路由表已随 handle 持有）

## 1. 调用链与接口映射

现状调用链（显式 LL）：
```
low_latency.py :: LowLatencyState.combine
  → _C.combine_low_latency(buffer.runtime, tensors, b, h, k, e, dtype)
    → csrc/buffers/low_latency_api.hpp  lambda
      → csrc/kernels/ep/low_latency.hpp :: launch_combine_low_latency(EpServerDedupCombineKernelArgs)
        → DeepJIT compile/launch combine_server_dedup_kernel (num_blocks=64)
```

LL combine 输入复用 dispatch 的 LL 路由表：`destinationIndex`、`sendCounts`、`tokenType`、`relayReadIndex`、`sourceMask`；输出 `yOut`（直接是最终合并结果）。

默认 combine 目标：
```
ep.py :: EPBuffer.combine → runtime.combine → csrc/buffers/ep.hpp :: combine(...)
  → launch_combine → combine_impl（RingReducer）+ combine_reduce_epilogue
  → 输出 combined_x / combined_topk_weights(可选) / event
```

## 2. 关键差异

| 输出 | 默认 combine | LL combine |
| --- | --- | --- |
| 数据 | `combined_x` `[B,H]` | `yOut` `[B,H]`（等价） |
| 权重侧通道 | 支持 `combined_topk_weights` 回传 | 无 |
| bias | 支持 0/1/2 个 bias | 无 |
| 输入 | 标准 handle（psum/dst_buffer_slot_idx） | LL 路由表（destinationIndex/sendCounts/...） |
| 归约 | 本地 RingReducer 单级 | 服务器级 partial + 源端聚合 |

结论：LL combine 的 `yOut` 与默认 `combined_x` 语义等价（都是对专家输出按路由求和），可直接作为 `combined_x` 返回。差异集中在「能力子集」与「输入路由表形态」。

## 3. 实施步骤

### 步骤 1：handle 持有 LL 路由表（T07 D-A 落地）
T08 的 dispatch 走 LL 去重后，在标准 `EPHandle` 内部持有 `send_counts`/`token_type`/`destination_index`/`relay_read_index`/`source_mask`（可选字段，仅 LL 路径填充），并打 `_buffer_token` 标记。combine 据此识别「LL 布局 handle」。

### 步骤 2：combine 准入（`csrc/buffers/ep.hpp`，`combine()` 内）
走 LL combine 的条件（全 rank 确定性）：
- `EP_ENABLE_LL_SERVER_DEDUP == "1"` 且 handle 为 LL 布局（步骤 1 标记）
- `bias is None`、`topk_weights is None`（无权重侧通道）
- `defer_epilogue == False`（LL 无 deferred）
- 形状/核数满足 LL tiling 约束

不满足任一条 → 回退官方 combine（RingReducer + epilogue），保持 bias/权重/deferred 兼容。

### 步骤 3：复用 `launch_combine_low_latency`
输入取自标准 handle 的 LL 路由表字段 + 专家输出 `x`；`yOut` 直接作为 `combined_x` 返回，`combined_topk_weights = None`。

### 步骤 4：事件语义
`async_with_compute_stream=True` 时返回有效事件；`previous_event`/`allocate_on_comm_stream` 按默认语义处理（LL 在当前 stream 完整执行）。

## 4. 风险与待确认
- **handle 生命周期**：LL 路由表随 handle 持有，需保证旧 handle 复用、batch 切换后仍有效（D04）；`_buffer_token` 校验防止混用不同布局 handle。
- **数值语义**：LL 服务器级 partial 聚合是 FP32 累加 → BF16 partial，与默认 RingReducer 的归约顺序不同，可能引入 ≤1ulp 差异；需用独立求和参考做容差验收（D03/C4 数值风险点）。
- **性能**：combine 的服务器级聚合只在 EP≥32 多机有收益；EP8 单机无 FIRST_HOP relay，收益有限甚至回退，默认必须关（D09）。
