# 完整 LL 接入默认接口设计（T07）

> 状态：设计评审稿（2026-10-22）
> 前置：T01 兼容矩阵（F01–F25 / O01–O10 / D01–D09，见 `DeepEP-gitcode/docs/handoff/deepep-ascend-t01-compatibility.md`）
> 目标提交：`lowLatencyDC` `8d41d2f` → `670ee41`（MTE policy 收缩为仅 EP128 且 B≤32）→ 本分支
> 目标：把完整 LL 的 dispatch 融合、服务器级去重、combine 聚合/去重/融合接入默认 `EPBuffer`，同时保持官方接口、返回值、数值语义与执行行为兼容。

## 1. 目标与非目标

**目标**
- 复用已移植的 LL 内核（`deep_ep/include/deep_ep/impls/low_latency/` 下的 `dispatch_server_dedup_kernel.h`、`combine_server_dedup_kernel.h`、`partial`、`source_aggregate` 等），在默认 `EPBuffer.dispatch/combine` 内，对满足准入的调用启用服务器级去重 + 单 kernel 融合。
- 不满足准入的调用保留官方路径（native URMA / 现有窄门 MTE）。

**非目标（本轮不做）**
- 不改变 `low_latency_mode=True` 的显式 LL 接口。
- 不把 LL 的固定 8 GiB 通信区、专用 UB 约束强加给默认调用。
- 不在未证明性能更快的前提下强制启用（D09）。

## 2. 现状差异（决定接入形态）

| 维度 | 默认 EPBuffer（native） | 完整 LL（low_latency） |
| --- | --- | --- |
| wire/布局 | 官方 arena + metadata（`dst_buffer_slot_idx`/`dst_gsge_idx`/psum） | 专用路由表：`destination_index`、`relay_read_index`、`token_type`、`source_mask`、`send_counts` |
| handle | 标准 `EPHandle`（psum、src metadata、反向 slot） | 专用字段 + `is_low_latency=True` |
| 路由前提 | 允许 masked（-1）与重复专家 | 要求有效且不重复专家 |
| combine | 支持 bias、权重侧通道、deferred | 无 bias、无权重侧通道、无 deferred |
| 核数 | 可变（`num_sms`） | 固定 32 AI / 64 AIV |
| 协议 | 官方 collective 布局 | packet 协议 + 服务器级 FIRST_HOP 去重 + partial 聚合 |

结论：LL 是一套**不同 wire 协议与 handle 形态**的实现，不能像 native MTE 那样只改传输内核、复用官方布局。接入默认接口必须先解决「布局/handle 统一」与「跨 rank 协议一致」两个前置问题。

## 3. 核心设计决策

### D-A. 布局统一：扩展标准 handle，而非替换
不引入第二套 handle 类型。在标准 `EPHandle` 内部**可选地**持有 LL 路由表（`send_counts`/`token_type`/`destination_index`/`relay_read_index`/`source_mask`），仅当 dispatch 实际走了 LL 去重路径时才填充；对外仍通过标准字段（`topk_idx`、`num_recv_tokens`、`num_recv_tokens_per_expert_list`、`psum_num_recv_tokens_per_expert`、`num_unaligned_recv_tokens_per_expert`）暴露合同。`EPHandle.is_low_latency` 语义改为「本 handle 由 LL 去重路径生成」，标准调用者无需感知。

### D-B. 路由前提（D06）：设备侧检查 + 回退
LL 要求无 masked、无重复专家。默认调用可能传入 masked/重复。方案：
- 在 dispatch 前对 `topk_idx` 做一次轻量设备侧合法性扫描（检测 `-1` 与 token 内重复专家）。
- 合法 → 走 LL 去重；不合法 → 回退 native URMA。
- 回退必须发生在通信开始前（D08：启动后不重试回退）。

### D-C. 跨 rank 协议一致（D05）：全组一致准入
LL 的 packet 协议与官方布局不互通。与 native MTE 的「各 rank 独立决策」不同，LL 去重要求**全组一致**：准入判据必须是全 rank 确定性的（只依赖 shape/容量/路由合法性/开关，不依赖单 rank 的局部 B）。所有 rank 要么都进 LL 去重，要么都走官方路径，否则引入一个小的 all-reduce 共识决定（优先用确定性判据，避免额外 collective）。

### D-D. combine 融合范围（D02/D03）：能力子集
LL combine 无 bias、无权重侧通道、无 deferred。方案：
- `bias=None` 且无权重侧通道且 `defer_epilogue=False` → 可走 LL combine 聚合/融合。
- 其余（bias、权重回传、deferred）保留官方 combine。
- 保持「combine 不把权重乘进 hidden」（D03）。

### D-E. 事件语义（D07）：immediate 优先
LL 是完整操作，不支持 deferred。`defer_epilogue=True` 的调用保留官方路径；`async_with_compute_stream=True` 的 immediate 调用在 LL 完成后返回有效事件。

### D-F. 性能门槛（D09）：默认为关
LL 去重/融合是重方案（额外 packet/relay/partial 开销）。接入后默认**不启用**，先以 `EP_ENABLE_LL_SERVER_DEDUP=1` 显式开关 + 三组/四组性能对照验证收益，再决定默认启用范围。这与现有 `EP_DISABLE_LOW_LATENCY_TRANSPORT` 回退开关并列，互不影响。

## 4. D01–D09 逐项准入实现方案

| 规则 | 实现方案 |
| --- | --- |
| D01 官方调用先合法 | 复用现有 Python/C++ 合法性检查；LL 接入不改报错语义，非法 dtype/容量/核数仍明确报错 |
| D02 功能逐项准入 | dispatch 走 LL 的条件必须显式排除 cached/FP8 非预量化/alignment≠1/累计 stats；combine 走 LL 排除 bias/权重/deferred |
| D03 输出及权重不变 | LL 输出 `recv_x` 逐行等价于官方展开布局；权重不乘 hidden；INT16 FP8 scale 格式不变 |
| D04 handle 可持续 | 扩展标准 handle 持有 LL 路由表，重复 combine、batch 切换后仍有效；禁止无检测混用不同布局 handle（用 `_buffer_token` 校验） |
| D05 跨 rank 协议一致 | 全 rank 确定性准入判据（见 D-C），不依赖单 rank 局部状态 |
| D06 路由前提可保证 | 设备侧路由合法性扫描 + 不合法回退 URMA（见 D-B） |
| D07 融合服从事件语义 | 仅 immediate 走 LL；deferred 保留官方路径（见 D-E） |
| D08 启动后不重试回退 | 通信开始前决定路径；启动后错误按作业错误处理 |
| D09 性能决定启用范围 | 默认关闭，显式开关 + 性能对照后再定（见 D-F） |

## 5. 分阶段实施

### 阶段 1（T08）：dispatch 融合 + 服务器级去重接入默认接口
1. host 侧（`csrc/buffers/ep.hpp`）新增 LL 去重准入判定：`EP_ENABLE_LL_SERVER_DEDUP=1` 且满足 D-A/D-B/D-C 的确定性条件。
2. 路由合法性扫描内核（检测 masked/重复专家），产出可回退判据。
3. 调用 `dispatch_server_dedup_kernel`（或复用其子结构），把 `destination_index`/`relay_read_index`/`token_type`/`source_mask` 映射进标准 handle，并从 LL 输出还原标准 `recv_x`/`recv_topk_weights`/`recv_src_metadata`。
4. cached、alignment、deterministic、异步行为：不满足的路径回退官方。

### 阶段 2（T09）：combine 聚合 + 去重 + 融合接入默认接口
1. 对接阶段 1 的 LL handle 布局与 metadata。
2. 复用 `combine_server_dedup_kernel` + `source_aggregate`，保持不乘权、兼容 `bias=None`，专家贡献恰好一次。
3. `bias`/权重侧通道/deferred 组合回退官方 combine。

## 6. 验收判据（对齐 D01–D09）
- 三组对照（official / 关 MTE 关 LL / 自动含 LL）：默认用例结果逐项一致；LL 命中用例有真实 kernel 路径证据。
- `--profile-paths` 确认 `dispatch_server_dedup_kernel`/`combine_server_dedup_kernel` 实际运行。
- 性能：LL 去重路径在目标 EP（≥32 的多机）上 dispatch/combine 耗时较官方不劣化，才可谈默认启用。
