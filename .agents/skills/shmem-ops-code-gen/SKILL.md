---
name: shmem-ops-code-gen
description: "根据 design.md 生成 SHMEM 算子代码、目录结构、CMake 和 README。关键词：代码生成、code-gen、实现、kernel、main.cpp。"
---

# SHMEM 算子代码生成

**Skill类型**：代码生成型（读取设计文档，输出可编译代码）

> **中文写作要求**：README.md 等交付文档必须使用中文撰写。仅 API 名称、代码片段、命令行示例等技术术语保留英文原文。

消费通过质量门禁的 `design.md`，生成 SHMEM 通信算子或通算融合算子的完整代码。不做需求设计（回 `shmem-ops-design`），不做正确性验证（交给 `shmem-ops-correctness-eval`），不做性能优化（交给 `shmem-ops-performance-optim`）。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/api.md](references/api.md) | SHMEM API 选择参考 |
| [references/code-patterns.md](references/code-patterns.md) | Host/Device 代码组织、RMA 模式、chunk/tail |
| [references/atomic-add-pattern.md](references/atomic-add-pattern.md) | `SetAtomicAdd<T>()` 累加：安全顺序、边界、风险。实现 reduce/累加时必读 |
| [references/code-style.md](references/code-style.md) | C/C++ 代码规范 |
| [references/readme-spec.md](references/readme-spec.md) | README.md 格式规范 |

模板（所有模板以 fenced code block 承载在 GUIDE.md 中，Agent 提取 code block 写入目标路径并替换占位符）：

| 模板 | 适用场景 |
| --- | --- |
| [templates/communication/GUIDE.md](templates/communication/GUIDE.md) | 纯通信算子（CMake + main.cpp + kernel + scripts） |
| [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md) | 通算融合算子（CMake + kernel AIC/AIV + main.cpp + scripts） |

必要时检查本地 `shmem/` 仓库的 `include/`、`src/`、`examples/`。真实代码优先于记忆。

---

## 输入门禁

开始生成前必须验证 `design.md`：

- [ ] 存在完整的 Canonical DSL `yaml` 代码块
- [ ] `source.user_confirmations` 记录了确认的 `op_name` 和 `dtypes`
- [ ] capability mapping 覆盖六类：lifecycle、memory、transport、sync、compute、scheduler
- [ ] DSL `schedule` 的 core_partition/tiling/phases 足够具体
- [ ] compile contract 写明 `cann_env` 和 `build_mode`
- [ ] **Design Review Before Handoff（section 5）已填写且无 FAIL 项**
- [ ] DSL `performance.baseline` 不为空或 `"none"`（必须为具体 baseline 来源——HCCL/aclnn/拼接/metric_only，且附 `baseline_search` 搜索记录）

门禁失败时停止，要求先用 `shmem-ops-design` 修订。

### 门禁执行规则

门禁检查是阻断条件，不是建议。执行方式：

1. 逐项检查上述 checklist，结果必须显式输出给用户（列表形式）
2. 任一项 FAIL → 立即停止，向用户报告缺失项，要求回 `shmem-ops-design` 补齐
3. 不得以"先跑通再补"、"设计基本够用"、"benchmark 不需要完整设计"等理由跳过 FAIL 项
4. 如果 design.md 缺少 Canonical DSL 的 `schedule` / `correctness` / `performance` section，直接判定 FAIL
5. 如果 Design Review Before Handoff（section 5）不存在或有 FAIL 项，直接判定 FAIL

---

## 工作流

```
步骤 1  提取设计契约
步骤 2  选择模板和参考 example
步骤 3  制定实现计划
步骤 4  渐进式代码生成
步骤 5  调用 shmem-ops-compile-debug 编译验证
```

---

## 步骤 1：提取设计契约

从 `design.md` 提取：

| 内容 | DSL 字段 |
| --- | --- |
| 算子身份 | `meta.op_name`、`op_kind`、target SoC、scope |
| 接口 | inputs/outputs、dtype、shape、visibility |
| 语义 | local_compute、communication、finalize |
| 拓扑 | team、peer_model、addressing |
| 内存 | buffers、symmetric_layout、signal/state |
| 调度 | phases、tile/chunk/tail、core_partition、overlap |
| 正确性 | oracle、tolerance、invariants、case_matrix |
| 性能 | metric、baseline、target_cases |

---

## 步骤 2：选择模板和参考 example

| `op_kind` 或语义 | 模板目录 |
| --- | --- |
| transport / collective / 纯 put/get/exchange | `templates/communication` |
| fused_compute_comm / local compute + communication | `templates/fused-compute` |

选定模板后，从对应 `GUIDE.md` 提取 fenced code block，写入 `design.md` 指定的目标路径，替换 `<op_name>`/`<OpName>`/`<OP_NAME>` 占位符。

参考 example 选择：
- allgather / put-get / SDMA / RDMA → `examples/allgather`、`examples/sdma`、`examples/rdma_demo`
- matmul allreduce / reduce-scatter → `examples/matmul_allreduce`、`examples/matmul_reduce_scatter`
- KV / dispatch combine → `examples/kv_shuffle`、`examples/dispatch_gmm_combine`

选定后记录路径和复用理由。

---

## 步骤 3：制定实现计划

编码前写出：
- 目标目录和文件列表
- 复用的 API、example、template
- `main.cpp` 与 Host helper 模块的职责边界
- 构建模式和编译命令
- README.md 覆盖范围

---

## 步骤 4：渐进式代码生成

按"最小正确路径 → 完整正确性 → 性能路径"顺序：

1. 目录结构、CMake、Host 入口、kernel wrapper
2. lifecycle + memory：初始化、symmetric allocation、释放
3. transport + sync：put/get、signal/wait、barrier
4. compute + finalize：local compute、dtype cast、accumulation
     - **通算融合**：AIC 必须使用 CATLASS BlockMmad 实现 matmul，禁止在 AIV 上用标量/向量运算替代
     - **纯通信**：按 design.md 实现 local compute（如有）
      - **GM 累加方式选择**：当 kernel 需要将远端数据累加到 output 时，按 [references/atomic-add-pattern.md](references/atomic-add-pattern.md) §12 决策优先级表选择。摘要如下：
>        1. **MTE 批量 `SetAtomicAdd<T>()` + `mte_get_nbi`**（首选）：多个 remote source 数据聚合到同一 destination 时，所有 source 的 get 全部 in-flight，MTE3 硬件累加。适用于 reduce-scatter / allreduce RS 阶段等场景
>        2. **统一 API `shmem_*_atomic_add`**（MTE 标量）：单元素远端累加（signal / flag / counter），引擎自动分发
>        3. **SDMA fallback**：MTE 不可用时，chunk 分拆 + 核间并行 + 核内串行 get+UB 累加
>        4. **UB 向量化累加**：单 remote source 或无 remote 传输场景，get 到 UB → `AscendC::Add` → DataCopy 写回
5. scheduler：phase、tile/chunk/tail、core partition、overlap
6. README.md（按 references/readme-spec.md）

### 关键约束

- **算子必须在 Device 执行**，禁止 Host RMA 作为主通信路径
- **通算融合算子必须使用 AIC + CATLASS 高性能计算**：AIC 侧必须使用 CATLASS BlockMmad（或同等高性能实现）执行 matmul/compute，AIV 侧负责通信（CommBlockEpilogue）。绝对禁止在 AIV 上用标量/向量点乘（如逐元素 for 循环乘加）替代 AIC + CATLASS 计算——即使"为了正确性先跑通"也不允许，因为这会产生无法优化的低性能基线
- **main.cpp 只做单 PE Host 编排**：参数解析、lifecycle、I/O、launch、cleanup
- 复杂 Host 逻辑拆到独立 `.cpp/.h`（如 `op_host_plan.cpp`）
- 跨 PE 传输必须使用 `aclshmem_*` 或 `aclshmemx_*` 接口
- symmetric allocation 顺序和大小在所有 PE 一致
- `block_dim=1` 仅临时调试用；首版 correctness 必须落地 design 的并发
- 新增核心能力必须对应 gap analysis

### 性能输出要求

main.cpp 的 `--perf` 模式必须输出双指标延迟和带宽（严格按照 [shmem-ops-performance-eval/references/timing-and-metrics-standard.md](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md) 执行）：

| 指标 | 说明 | 公式 |
| --- | --- | --- |
| `e2e_us` | 端到端延迟（含输入到对称堆的拷贝 + barrier + kernel） | 从输入到对称堆的拷贝前到 stream sync 后 |
| `kernel_us` | 纯 kernel 延迟 | 从 kernel launch 前到 stream sync 后 |
| `algo_bandwidth_GBps` | 算法带宽（基于 e2e_us） | `logical_payload_bytes / (e2e_us * 1e-6) / 1e9` |
| `bus_bandwidth_GBps` | 总线带宽 | `algo_bandwidth * bus_factor` |
| `bandwidth_utilization_pct` | 带宽利用率 | `bus_bandwidth / peak_bandwidth * 100`（peak_bandwidth 按通信模式确定，见下文） |

`bus_factor` 按算子语义确定（不区分拓扑，NCCL 惯例的通信量标准化系数）：AllReduce: `2*(n-1)/n`，ReduceScatter/AllGather: `(n-1)/n`，AllToAll/Shuffle: `(n-1)/n`，Broadcast/P2P: 1。

`peak_bandwidth` 按通信模式确定（参考 [hardware-architecture.md §2.6](../shmem-ops-design/references/hardware-architecture.md)）：
- P2P 点对点：28 GB/s（单条 HCCS 链路单向）
- 集合通信（AllReduce/AllGather 等）：聚合带宽，如 910B3 8 卡 full-mesh 为 7 × 28 = 196 GB/s

perf 循环结构要求：
- **e2e 循环**：每轮包含输入到对称堆的拷贝 + barrier + kernel launch，模拟真实调用模式
- **kernel-only 循环**：输入到对称堆的拷贝已完成，只测 kernel launch，用于内部优化定位
- **禁止跳过输入到对称堆的拷贝**：真实使用场景中 input 来自用户 Device GM buffer，性能测试 **MUST** 包含完整的 `input(GM) → 输入到对称堆的拷贝（MTE / SDMA / aclrtMemcpy 等均可） → symmetric memory → kernel` 流程。输入到对称堆的拷贝 **MUST** 计入 `e2e_latency_us`，不得在性能循环前预拷贝一次使 e2e 口径缩水

注意：`algo_bandwidth` 不乘 2，统一按 input size 计算（NCCL `algBw` 惯例）。`logical_payload_bytes` 必须在输出中注明口径（单 PE 还是全局）。

如果算子包含 `SHMEMI_PROF_START/END` 打点，`--perf` 模式还应调用 `aclshmemx_show_prof()` 输出 Device 帧数据。

---

## 步骤 5：编译验证

调用 `shmem-ops-compile-debug`，传入 compile contract。持续修复直到编译通过或确认阻塞。

---

## 最终目标目录结构

最终交付目录结构 **MUST** 严格遵循以下布局。`shmem-ops-code-gen` 本阶段 **MUST** 生成 `CMakeLists.txt`、`README.md`、`src/` 下算子源码，以及有 baseline 时的 `baseline/` 源码和编译配置；`docs/` 下的阶段报告由对应 skill 生成或在后续阶段补齐。所有 `.md` 文档归入 `docs/`，所有算子源码 `.cpp/.h` 归入 `src/`，所有 baseline 源码和编译配置归入 `baseline/`，测试脚本归入 `scripts/`。不允许文档散放在算子根目录，不允许算子源文件散放在根目录，**NEVER** 将 baseline 源码放在 `src/` 下：

```
op_name/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── design.md                  # shmem-ops-design
│   ├── review-report.md           # shmem-ops-code-review
│   ├── correctness_report.md      # shmem-ops-correctness-eval
│   ├── performance_report.md      # shmem-ops-performance-eval / shmem-ops-performance-optim
│   └── case_matrix_report.md      # shmem-ops-testcase-gen
├── src/
│   ├── main.cpp
│   ├── op_name_kernel.cpp
│   ├── op_name_kernel.h
│   ├── op_host_plan.cpp (可选)
│   └── op_host_plan.h (可选)
├── scripts/
│   ├── gen_data.py
│   ├── check_result.py
│   ├── run.sh
│   └── run_case_matrix.py
└── baseline/                       # 有 baseline 时 MUST 存在
    ├── CMakeLists.txt              # 独立的 baseline 编译 target
    ├── src/
    │   └── op_name_baseline.cpp    # HCCL/aclnn/拼接 baseline 源码
    └── scripts/
        └── run_baseline.sh         # baseline 运行脚本
```

---

## 检查点

- [ ] 目录结构符合规范：`docs/` 承载所有阶段产出的 `.md`、`src/` 含全部算子 `.cpp/.h`、`baseline/` 含全部 baseline 源码和编译配置、`scripts/` 含全部测试脚本
- [ ] CMake 生成，target 命名与 op_name 一致
- [ ] 代码遵循 design.md 的 phase/transport/sync/partition
- [ ] main.cpp 不含复杂逻辑
- [ ] main.cpp `--perf` 模式输出双指标延迟（e2e_us、kernel_us）和 algo_bandwidth_GBps、bus_bandwidth_GBps
- [ ] e2e_latency_us 口径包含输入到对称堆的拷贝（未预拷贝缩水）
- [ ] kernel 包含 `SHMEMI_PROF_START/END` 打点，覆盖主要耗时 phase（copy_in/remote_put_get/signal_wait/local_compute/finalize）
- [ ] README.md 覆盖编译、运行、校验入口
- [ ] 编译成功（或记录环境阻塞）
- [ ] 记录所有生成/修改的文件

---

## 反模式（NEVER DO THESE）

- ❌ Host RMA 作为 correctness 实现
- ❌ main.cpp 包含 route/payload/tiling/golden 逻辑
- ❌ main.cpp fork/spawn 多 PE
- ❌ DataCopy 直接写远端地址
- ❌ 跳过 Device kernel 实现
- ❌ 不读 design.md 就生成代码
- ❌ design.md 门禁有 FAIL 项仍继续生成代码（"先跑通再补设计"是最常见的违规路径）
- ❌ GM 标量循环累加作为性能路径交付（按 atomic-add-pattern.md §12 决策表选择正确方式）
- ❌ 修改 SHMEM 核心库却无 gap analysis 授权
- ❌ 通算融合算子在 AIV 上用标量/向量点乘替代 AIC + CATLASS 高性能计算（即使以"正确性验证"为由也不允许——必须从一开始就使用 CATLASS BlockMmad + CommBlockEpilogue 的 AIC/AIV 分工模式）
- ❌ 将输入到对称堆的拷贝排除在 e2e 性能循环外（性能循环前预拷贝使 e2e 口径缩水）
- ❌ `.md` 文档散放在算子根目录（必须归入 `docs/`）
- ❌ `.cpp/.h` 源文件散放在算子根目录（必须归入 `src/`）
- ❌ baseline 源码放在 `src/` 下或算子根目录（必须归入 `baseline/src/`，编译 target 必须在 `baseline/CMakeLists.txt`）
