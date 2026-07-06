---
name: shmem-ops-performance-eval
description: "SHMEM 算子性能采集、baseline 对比、聊天自动输出和瓶颈分析。关键词：性能采集、performance、baseline、bandwidth、steady_bus、自动输出、优化轮次。"
---

# SHMEM 算子性能采集

**Skill类型**：评测型（采集性能数据，对比 baseline，将结构化数据写入 performance_report.md）

> **中文写作要求**：性能报告 **MUST** 使用中文撰写。仅 API 名称、代码片段、指标名（如 `e2e_latency_us`）等技术术语保留英文原文。

正确性验证通过后执行。**仅当** `design.md` `meta.performance_required: true`（Phase 0 #4 用户已确认）时由编排器调用 Phase 6；**NEVER** 在用户未要求性能工作时擅自采集。

性能**优化**（Phase 6.5）**仅当** `meta.performance_auto_optim: true` 且 Phase 6 未达标时由编排器自动调用 `shmem-ops-performance-optim`；`performance_auto_optim: false` 时 eval 只采集与报告，**禁止**擅自改 kernel。

采集性能数据、接入 baseline、分析瓶颈，将结构化数据写入 performance_report.md（模板：[templates/performance-report.md](templates/performance-report.md)）。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/performance-eval-guide.md](references/performance-eval-guide.md) | 采集流程概览、contract 字段、结果表格式、规模要求 |
| [references/baseline-selection.md](references/baseline-selection.md) | baseline 选择策略、HCCL/aclnn 清单、决策树 |
| [references/baseline-compare-workflow.md](references/baseline-compare-workflow.md) | baseline C++ 接入、离线对比、聊天对比表 |
| [references/perf-workflow.md](references/perf-workflow.md) | **性能三阶段采集 + sweep 命令代码段** |
| [references/perf-chat-output-spec.md](references/perf-chat-output-spec.md) | **聊天自动输出规范** |
| [references/platform-perf-spec.md](references/platform-perf-spec.md) | **平台四算子带宽/时延目标表** |
| [references/timing-and-metrics-standard.md](references/timing-and-metrics-standard.md) | 时间打点标准、双指标方案、指标计算公式、Device 打点、计算指标 |
| [references/profiling-tools.md](references/profiling-tools.md) | msprof、SHMEM cycle profiling 使用方法 |
| [references/device-profiling-guide.md](references/device-profiling-guide.md) | SHMEMI_PROF 完整操作指南（kernel 打点 → Host 导出 → 填报告） |
| [templates/performance-report.md](templates/performance-report.md) | 性能报告模板（eval 写入数据的格式规范） |
| [references/skill-performance-review.md](references/skill-performance-review.md) | **维护者备忘**（历史修正记录；Phase 6 运行时不必 Read） |
| [../shmem-ops-design/references/hardware-architecture.md](../shmem-ops-design/references/hardware-architecture.md) | 昇腾硬件参数速查（计算带宽利用率时 **MUST** 参考） |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO`（读仓内 profiling 文档前） |

## 输入

- design.md 的 perf contract（metric、baseline、target_cases、最小规模）
- 已通过正确性验证的算子二进制
- profiler 环境（SHMEMI_PROF_START/END、aclshmemx_get_prof）
- 调用上下文（由 `shmem-ops-performance-optim` OptimStep 5 传入）：

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `target_section` | 写入目标区域 | `round_0`（基线采集）、`round_1_step_3`（Round 1 step 3） |
| `round_label` | 轮次标签 | `Round 0 基线`、`Round 1 机制优化：并发 peer put`、`Round 2 机制优化：double buffer` |
| `is_optimal_step` | 是否为该轮最优 step | `true`（需输出 Device Frame Table）、`false`（仅填 step 表行） |

> 独立调用 eval 时（非 optim 驱动），默认 `target_section = round_0`，仅写入 §2 + §3。

### 工作流

```
步骤 1  选择 baseline + 在 baseline/ 实现 C++ 可执行文件 + 采集 baseline 性能（[BASELINE_PERF]）
步骤 2  采集 SHMEM 性能（[PERF]）+ [perf-workflow.md](references/perf-workflow.md) sweep 找 steady_bus + 离线对比
步骤 3  瓶颈分析 + 写入 performance_report.md + **聊天自动输出**（见 perf-chat-output-spec.md）
```

**聊天自动输出（Hard Gate）**：脚本跑完后 **MUST** 在同轮对话输出 **带宽表 + 时延表**（[platform-perf-spec.md](references/platform-perf-spec.md)），及 sweep/平台区对比，**NEVER** 只报时延或等用户追问。

**参考实现**：`custom-ops/alltoallv/baseline/` + [perf-workflow.md](references/perf-workflow.md)

### 写入规则

根据 `target_section` 和 `is_optimal_step` 决定写入位置：

| target_section | is_optimal_step | 写入位置 | 具体内容 |
| --- | --- | --- | --- |
| `round_0` | — | §2 + §3.1~§3.5 + S 档数据 | Baseline 详细信息、L 档性能结果/通信/计算/Device Frame/性能对比表；S 档数据单独存储供 §6.2 使用 |
| `round_N_step_M` | `false` | §4 Round N Step 表 | 在已有 Step 表中追加一行（仅 L 档数据） |
| `round_N_step_M` | `true` | §4 Round N | Step 表行（仅 L 档）+ 性能验证对比表 + Device Frame Table |
| `round_N_step_M` | `true`（最终轮） | §4 + §5 + §6 + §7 | 上述 + §5 轮次总览行 + §6.1 L 档对比 + §6.2 S 档对比 + §7 结论 |
| — | 仅 S 档 | §6.2 | S 档在 Round 0 采集 baseline S 数据，最终轮重新采集 final S 数据写入 §6.2 对比。**NEVER** 写入 §4 优化记录 |

**节号强制约束**：写入时 **MUST** 使用模板定义的固定节号（§1~§7）和节名。以下行为 **NEVER** 允许：
- 在 §1~§7 之外新增自定义章节（如"采集命令"、"指标公式"、"正确性回归"）
- 辅助信息归入已有章节：采集命令 → §2 Baseline 详细信息；指标公式 → §3.2/§3.3 表头备注
- §4 每个 Round **MUST** 包含完整子结构：Step 表 → 性能验证对比表 → Device Frame Table → 本轮最优配置 → 决策

---

## 步骤 1：Baseline 选择与采集

**MUST** 按以下顺序逐步查找，声称"无 baseline"前每步都要排查（详见 [references/baseline-selection.md](references/baseline-selection.md)）：

1. **HCCL 集合通信算子**：AllGather、AllReduce、ReduceScatter、AllToAll、Broadcast、Reduce、Scatter、Gather（完整清单见 baseline-selection.md §1.1）
2. **aclnn 扩展算子**：aclnnAlltoAllV 等（完整清单见 baseline-selection.md §1.2）
3. **指标测试（无直接 baseline）**：仅测量利用率指标（见 baseline-selection.md §3）

`peak_bandwidth` 按通信模式确定（参考 [hardware-architecture.md §2.6](../shmem-ops-design/references/hardware-architecture.md)）：
- P2P 点对点场景：28 GB/s（单条 HCCS 链路单向）
- 集合通信场景（AllReduce/AllGather/ReduceScatter 等）：聚合带宽，如 910B3 8 卡 full-mesh 为 7 × 28 = 196 GB/s
- 报告中 **MUST** 注明 peak_bandwidth 使用的值和来源

规则：
- 存在 HCCL/aclnn baseline 时 **MUST** 接入
- 达标线 **MUST** 按 [platform-perf-spec.md](references/platform-perf-spec.md) 按算子取值（非统一 80%）
- 无 baseline 时 **MUST** 记录完整搜索过程（逐步排查 HCCL 清单 → aclnn 清单）
- baseline 测试程序 **MUST** 以 C++ 可执行文件方式实现，**NEVER** 使用 Python 脚本
- 选定 baseline 后 **MUST** 立即采集 baseline 性能数据（使用与当前实现相同的 case 和环境），输出 `[BASELINE_PERF]`（含 `e2e_us`、`bus_bandwidth_GBps` 等）
- **MUST** 在 `custom-ops/<op>/baseline/` 放置 C++ baseline 源码与 `baseline/scripts/run_baseline.sh`；算子根 `CMakeLists.txt` 通过 `add_subdirectory(baseline)` 编译
- AllToAllV 类算子：CANN 无独立 `aclnnAlltoAllV` 头文件时，**MUST** 回退 `HcclAlltoAllV` 并在 `baseline_search` 记录原因（见 baseline-compare-workflow.md §5）
- **MUST** 按 [perf-workflow.md](references/perf-workflow.md) 分三阶段采集（**NEVER** 同脚本混跑 HCCL+SHMEM）

---

## 步骤 2：性能采集、计算与瓶颈分析

**时间打点标准**：严格按照 [references/timing-and-metrics-standard.md](references/timing-and-metrics-standard.md) 执行。SHMEM 算子 **MUST** 报告双指标：

| 指标 | 含义 | 起点 | 终点 | 用途 |
| --- | --- | --- | --- | --- |
| `e2e_latency_us` | 端到端延迟（含输入到对称堆的拷贝 / staging copy） | 输入到对称堆的拷贝前 | stream sync 后 | 与 baseline 公平对比 |
| `kernel_latency_us` | 纯 kernel 执行时间 | kernel launch 前 | stream sync 后 | 内部优化定位 |

与 baseline 对比时 **latency 参考 e2e_us**，**带宽达标与 Round 间对比 MUST 用 steady_bus_bandwidth_GBps**（kernel 口径）。内部优化定位用 kernel_latency_us。

**MUST** 记录：

| 字段 | 内容 |
| --- | --- |
| case | shape、dtype、PE count、engine、scope |
| command | 完整测试命令和环境 |
| metric | e2e_latency_us、kernel_latency_us、algo_bandwidth_GBps、bus_bandwidth_GBps、**steady_bus_bandwidth_GBps**、bandwidth_utilization_percent |
| device frame | phase、cycles、avg_us、percent_of_e2e |
| compute | effective_flops、compute_utilization_percent（标注 N/A） |

通信指标计算公式（**MUST** 在报告中写明 logical_payload_bytes 口径；bus_factor 仅在通信指标子节说明，不作为主表列；详见 [timing-and-metrics-standard.md](references/timing-and-metrics-standard.md) section 4）：

```text
algo_bandwidth_GBps = logical_payload_bytes / latency_s / 1e9
bus_bandwidth_GBps  = algo_bandwidth_GBps * bus_factor
bandwidth_utilization_percent = bus_bandwidth_GBps / peak_bandwidth_GBps * 100
```

注意：`algo_bandwidth` **NEVER** 乘 2，统一按 input size 计算（NCCL `algBw` 惯例）。与 baseline 对比时 latency 使用 `e2e_latency_us`。

性能 case 要求（规模分档见 [shmem-ops-testcase-gen/references/testcase-scale-standard.md](../shmem-ops-testcase-gen/references/testcase-scale-standard.md)）：
- 性能采集 **MUST** 覆盖 **8PE S 档 + 8PE L 档** 两种规模，**NEVER** 只采 L 档
- L 档：集合通信 ≥256MB 数据量（全 PE 总量）
- S 档：按 testcase-scale-standard.md 定义
- **NEVER** 只用 smoke 小 shape

有 baseline 时，计算 delta（当前实现 vs baseline）和利用率；无 baseline 时，计算 metric_only 达标判断。

**瓶颈分析**：基于 profiler frame table 判断：
- 哪个 phase 占比最高
- 是否存在不合理的 wait/sync 占比
- 带宽利用率是否符合预期

---

## 输出字段规范

eval 输出以下结构化数据，写入 performance_report.md。缺失任一项视为不完整。

### **MUST** 输出

| 输出项 | 字段 | 说明 |
| --- | --- | --- |
| Baseline 信息 | type, source_api, search_process, measurement | 逐步排查记录、baseline 测试数据 |
| 性能结果表 | case_id, shape, dtype, n_pes, e2e_us, kernel_us, algo_GBps, bus_GBps, utilization% | 每个采集 case 的完整指标 |
| 通信指标 | logical_payload_bytes, bus_factor, peak_bandwidth_GBps, bandwidth_utilization_percent | 通信算子 **MUST** |
| 计算指标 | op_count_flops, compute_latency_us, effective_flops, peak_flops, compute_utilization_percent | 标注 "N/A" |
| Device Frame Table | frame_id, phase, avg_us, max_core_us, count, percent_of_e2e, bottleneck_note | **MUST** 存在 |
| 瓶颈判断 | 主要耗时来源、带宽利用率是否达标、计算利用率是否达标 | 基于 frame table 分析 |
| Delta vs Baseline | steady_bus 对比、达标判断 | 有 baseline 时 **MUST** |
| 聊天自动输出 | sweep 表、平台区对比表、（optim 驱动时）轮次 Δ% 表 | **MUST**，见 perf-chat-output-spec.md |

### 调用方差异

| 调用方 | target_section | 聊天 MUST 额外输出 |
| --- | --- | --- |
| Phase 6 独立 eval | `round_0` | §2 perf-chat-output-spec：Sweep 表 + 平台区对比 |
| optim OptimStep 5（每 step） | `round_N_step_M` | 无（仅写报告 Step 行） |
| optim OptimStep 5（round 最优） | `round_N_step_M` + `is_optimal_step=true` | §3 perf-chat-output-spec：本轮最优表 + 更新累计总览 |
| optim 全部完成 | 最终轮最优 | §4 perf-chat-output-spec：最终对比表 |

### 输出格式

性能结果以表格形式在聊天界面展示，关键指标 **MUST** 摘要输出，**NEVER** 仅输出路径。

**有 baseline 时，聊天回复 MUST 包含带宽表与时延表**（steady_bus GB/s、比值、目标、达标），格式见 [platform-perf-spec.md](references/platform-perf-spec.md) 与 [perf-chat-output-spec.md §2](references/perf-chat-output-spec.md)。**NEVER** 只粘贴单行 `[PERF]` 或只报 e2e 时延。

示例（结构示意，数值须来自实际采集）：

| 指标 | SHMEM | HCCL baseline | SHMEM/baseline |
| --- | --- | --- | --- |
| bus_GBps (M_fp16_8pe) | 30.86 | 10.61 | 290.7% PASS |

**稳定带宽**：对比与达标 **MUST** 使用 `steady_bus_bandwidth_GBps`（kernel 口径），且 **MUST** 经 [perf-workflow.md §2](references/perf-workflow.md) sweep 确认 per-PE payload ≥ 64MB 且相邻点变化 ≤5%。

---

## 检查点

- [ ] baseline 已选择或记录搜索过程
- [ ] 性能数据已采集（或标记无硬件原因）
- [ ] 包含 L 档（大规模）case
- [ ] 瓶颈已分析
- [ ] 输出字段规范中全部 **MUST** 项已输出
- [ ] 性能数据已写入 performance_report.md
- [ ] 已实现 baseline C++ 程序且 `[BASELINE_PERF]` 已采集
- [ ] 聊天已按 [perf-chat-output-spec.md](references/perf-chat-output-spec.md) 自动输出（非用户追问才补）
- [ ] optim 驱动时：每轮最优 step 后已输出 Round Δ% 表与累计总览

---

## 反模式（NEVER DO THESE）

- ❌ 有可用 baseline 却不接入
- ❌ 只采集时间不分析瓶颈
- ❌ 用 smoke 小 shape 作为性能结论
- ❌ 没有硬件却编造性能数据
- ❌ 通信算子不测量带宽利用率
- ❌ 只输出 algo_bandwidth 不输出 bus_bandwidth 和 utilization
- ❌ 报告中缺少 Device Frame Table
- ❌ baseline 字段写 `"none"` 但不记录搜索过程
- ❌ SHMEM 计时不含输入到对称堆的拷贝（staging copy）（只计 kernel 时间与 baseline 对比）
- ❌ 只报告单一延迟指标，不区分 e2e_latency_us 和 kernel_latency_us
- ❌ algo_bandwidth 公式乘 2（应按 NCCL algBw 惯例，不乘 2）
- ❌ 输出指标不完整
- ❌ 将用户 input 直接构造在对称内存上以跳过输入到对称堆的拷贝（staging copy）
- ❌ 写入 performance_report.md 时新增模板未定义的章节（如"采集命令"、"指标公式"、"正确性回归"），所有信息必须归入 §1~§7 对应子节
- ❌ custom-ops 性能采集在文档/回复中首选裸 `cd custom-ops/<op> && ...`（用 [perf-workflow.md](references/perf-workflow.md)）
- ❌ 性能测试完成后不在聊天中自动输出结果，等用户引导
- ❌ Phase 6.5 只在第 5 轮结束才报性能变化
- ❌ 只报 SHMEM `[PERF]` 不报 baseline `[BASELINE_PERF]` 及对比表
- ❌ 无离线对比流程却声称已完成 baseline 对比
- ❌ **同一 shell 内混跑 HCCL baseline 与 SHMEM perf**（须分阶段 + 离线 compare）
- ❌ **用 M/S 档或单点 e2e 带宽作最终性能结论**（带宽仍随 payload 上升 = 未饱和）
- ❌ 未跑 sweep / 倍增 sweep 即声称测得「稳定带宽」
- ❌ 达标对比用 `bus_bandwidth_GBps`(e2e) 而不用 `steady_bus_bandwidth_GBps`(kernel)
- ❌ AllToAllV 未排查 `HcclAlltoAllV` 就直接写 metric_only
- ❌ baseline 与 SHMEM 使用不同 gen_data / sendcounts
