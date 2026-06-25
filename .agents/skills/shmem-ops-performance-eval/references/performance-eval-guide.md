# SHMEM 性能采集流程与规范

本文规定 `shmem-ops-performance-eval` 在 correctness 通过后如何采集性能、分析瓶颈并输出结构化数据。所有性能结论 **MUST** 来自实际命令输出或明确标注为假设。

## 核心原则

- **性能阶段是强制性阶段**：正确性通过后 **NEVER** 直接停止，**MUST** 进行性能采集、baseline 接入和瓶颈分析
- **有 baseline 时 **MUST** 接入**：有 HCCL/aclnn/拼接 baseline 时，**MUST** 接入并对比，默认达标线为 current ≥ baseline 的 80%
- **无 baseline 时 **MUST** 指标验收**：通信算子测量带宽利用率或端到端时延，能计算带宽利用率时 **NEVER** 低于 20%
- **功能完善不算性能优化**：例如从 Host RMA 改为 Device kernel 是功能完善，不是性能优化
- **并发补齐不算性能优化**：例如从临时 `block_dim=1` 改为 design 要求的多 block，是 correctness 实现补齐
- **只采集当前实现时间不算完成**：**MUST** 有 baseline 对比并达标，或在 metric_only 下完成指标测量并达标

## 1. 前置条件

进入性能阶段前 **MUST** 满足：

- 编译成功
- functional/correctness contract 已通过
- implementation-vs-design review 已通过：编译 target、transport API、launch block_dim、phase、tile/chunk/tail 与 design 一致
- 中等规模 correctness case 已通过，**NEVER** 只依赖 smoke 小 shape
- 输出可见性、同步不变量、tail case 已验证或明确未验证原因
- `design.md` 的 performance contract 已读取

correctness 未通过时，性能数据无效。

## 2. 性能 contract 字段

`design.md` 应提供或由实现阶段补齐：

| 字段 | 说明 |
| --- | --- |
| metric | `latency_us`、`algo_bandwidth_GBps`、`bus_bandwidth_GBps`、`bandwidth_utilization_percent`、`effective_flops`、`compute_utilization_percent`、`cycles` |
| baseline | HCCL（见 baseline-selection.md §1）、aclnn（见 baseline-selection.md §2）、拼接（见 baseline-selection.md §3）、已有 SHMEM example、用户参考实现、或 metric_only |
| baseline_target | 有 baseline 时默认 current ≥ 80% baseline；无 baseline 时写 metric_only 指标目标，通信算子能计算带宽利用率时 **NEVER** 低于 20% |
| cases | shape、dtype、PE count、engine、scope |
| min_scale | 集合通信至少 256MB 级通信数据量；计算/通算融合 hidden size 上千级；无法满足时说明原因 |
| repeats | 预热次数、统计次数 |
| profiler | msprof、torch_npu profiler、SHMEM cycle profiling、example 内部计时 |
| target | 目标改善方向或验收阈值 |

如果 contract 不完整，先补到 `design.md` 或在实现日志中显式写出假设。

## 3. 性能结果表

每个 case 至少输出：

| 字段 | 含义 |
| --- | --- |
| case_id | 唯一 case 名称 |
| shape | 输入输出 shape |
| dtype | 数据类型 |
| n_pes | PE 数 |
| engine | MTE、SDMA、RDMA/RoCE 或 default |
| metric | 主指标 |
| baseline | baseline 数值或 N/A |
| current | 当前实现数值 |
| delta | 与 baseline 差异 |
| target | baseline 80% 达标线或 metric_only 指标目标 |
| pass | 是否达标 |
| notes | 关键现象 |

如果没有 baseline，仍需记录 current、方法、baseline 搜索过程、metric_only 目标和达标判断，不要伪造对比。

通信算子额外输出：

| 字段 | 含义 |
| --- | --- |
| logical_payload_bytes | 算法语义数据量，说明单 PE/全局口径 |
| algo_bandwidth_GBps | `logical_payload_bytes / latency_s / 1e9` |
| bus_bandwidth_GBps | `algo_bandwidth_GBps * bus_factor` |
| peak_bandwidth_GBps | 硬件链路或 fabric 峰值及来源 |
| bandwidth_utilization_percent | `bus_bandwidth_GBps / peak_bandwidth_GBps * 100` |

> bus_factor 是推导参数，不作为性能结果表的主列输出，仅在"通信指标"中单独说明其数值和来源。

计算算子额外输出：

| 字段 | 含义 |
| --- | --- |
| op_count_flops | 有效 FLOPs 公式和数值 |
| compute_latency_us | compute frame 耗时；若使用端到端耗时 **MUST** 标注 |
| effective_flops | `op_count_flops / compute_latency_s` |
| peak_flops | 当前 SoC + dtype 的硬件峰值及来源 |
| compute_utilization_percent | `effective_flops / peak_flops * 100` |

关键片段占比额外输出：

| frame_id | phase | avg_us | max_core_us | count | percent_of_e2e | bottleneck_note |
| --- | --- | --- | --- | --- | --- | --- |

## 4. 性能 case 规模要求

- smoke 小 shape 只用于启动和 sanity check，不作为性能结论
- 集合通信类至少包含 256MB 级通信数据量 case
- 计算或通算融合类至少包含 hidden size 上千级 case
- 达不到规模要求时，最终报告 **MUST** 写"未满足性能规模门禁"，**NEVER** 写成完整性能完成