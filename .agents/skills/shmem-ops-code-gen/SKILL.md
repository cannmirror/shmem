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
| [references/internal-api-boundary.md](references/internal-api-boundary.md) | **禁止** `aclshmemi_*`、deprecated barrier、quiet 误用 |
| [references/api.md](references/api.md) | SHMEM API 选择参考 |
| [references/code-patterns.md](references/code-patterns.md) | Host/Device 代码组织、RMA 模式、chunk/tail |
| [references/atomic-add-pattern.md](references/atomic-add-pattern.md) | `SetAtomicAdd<T>()` 累加：安全顺序、边界、风险。实现 reduce/累加时必读 |
| [references/code-style.md](references/code-style.md) | C/C++ 代码规范 |
| [references/readme-spec.md](references/readme-spec.md) | README.md 格式规范 |
| [../shmem-ops-compile-debug/references/shmem-repo-docs-index.md](../shmem-ops-compile-debug/references/shmem-repo-docs-index.md) | 仓内 `docs/`（**只读**）、`install/shmem/include/`、examples 参考 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | 定位 `SHMEM_REPO`（**读仓内文件前**） |

读 `${SHMEM_REPO}/docs/` **只读**；**MUST** 先定位 `SHMEM_REPO`（shmem-repo-resolution），**NEVER** 向 `docs/` 追加或修改内容。

模板（分文件承载，按生成步骤按需读取，**NEVER** 一次读完全部模板）：

| 文件 | 阅读时机 | 内容 |
| --- | --- | --- |
| [templates/communication/GUIDE.md](templates/communication/GUIDE.md) | 步骤 2 选模板时 | 索引、路径映射、约束 |
| [templates/communication/templates-cmake-main.md](templates/communication/templates-cmake-main.md) | 步骤 4 子步骤 1 | `CMakeLists.txt`、`main.cpp` |
| [templates/communication/templates-kernel.md](templates/communication/templates-kernel.md) | 步骤 4 子步骤 2–4 | `*_kernel.h`、`*_kernel.cpp` |
| [templates/communication/templates-scripts.md](templates/communication/templates-scripts.md) | 步骤 4 子步骤 5（若需补齐 scripts） | `gen_data.py`、`check_result.py`、`scripts/run.sh` |
| [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md) | `op_kind=fused_compute_comm` 时步骤 2–4 | 通算融合（CMake + AIC/AIV kernel + main.cpp + scripts） |

必要时先定位 `SHMEM_REPO`，再查阅仓内文档与头文件（见 shmem-repo-docs-index、shmem-repo-resolution）：`${SHMEM_REPO}/docs/`、`${SHMEM_REPO}/install/shmem/include/`、`${SHMEM_REPO}/examples/`。真实代码与文档优先于记忆。

---

## 输入门禁

开始生成前必须验证 `design.md`：

- [ ] 存在完整的 Canonical DSL `yaml` 代码块
- [ ] `source.user_confirmations` 记录了确认的 `op_name` 和 `dtypes`
- [ ] capability mapping 覆盖六类：lifecycle、memory、transport、sync、compute、scheduler
- [ ] DSL `schedule` 的 core_partition/tiling/phases 足够具体
- [ ] compile contract 写明 `cann_env` 和 `build_mode`
- [ ] **Design Review Before Handoff（section 5）已填写且无 FAIL 项**
- [ ] DSL `performance.baseline` 不为空或 `"none"`（必须为具体 baseline 来源——HCCL/aclnn/metric_only，且附 `baseline_search` 搜索记录）

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
| `transport` / `collective` / `compute` / 纯 put/get/exchange | `templates/communication` |
| `fused_compute_comm`（Matmul/GEMM + 跨 PE 通信，AIC/AIV CoC） | `templates/fused-compute` |

> **`compute`**：单 PE 或 Device 内本地计算、无跨 PE 通信语义时仍用 `templates/communication`（通常仅 Host + 空/轻量 kernel）；**禁止**因存在 `local_compute` 字段就路由到 fused-compute。  
> **`fused_compute_comm`**：必须同时含 Cube matmul（CATLASS BlockMmad）与 SHMEM 跨 PE 通信（CommBlockEpilogue），见 [core-allocation.md §4](../shmem-ops-design/references/core-allocation.md)。

选定模板后，**仅读取当前生成步骤对应的模板文件**（见上表）。纯通信按 [templates/communication/GUIDE.md](templates/communication/GUIDE.md) 分步读取子模板；通算融合从 [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md) 按章节标题提取 fenced code block，写入目标路径，替换 `<op_name>`/`<OpName>`/`<OP_NAME>` 占位符。

参考 example 选择：
- allgather / put-get / SDMA / RDMA → `${SHMEM_REPO}/examples/allgather`、`sdma`、`rdma_demo`（先定位 `SHMEM_REPO`）
- matmul allreduce / reduce-scatter → `${SHMEM_REPO}/examples/matmul_allreduce`、`matmul_reduce_scatter`
- KV / dispatch combine → `${SHMEM_REPO}/examples/kv_shuffle`、`dispatch_gmm_combine`

选定后记录路径和复用理由。

---

## 步骤 3：制定实现计划

编码前写出：
- 目标目录和文件列表（**默认** `custom-ops/<op_name>/`，非 `examples/`）
- 复用的 API、example、template
- `main.cpp` 与 Host helper 模块的职责边界
- 构建模式（`independent_project` 默认）与编译命令（[custom-ops-entrypoints.md](../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1 编译）
- README.md 覆盖范围

---

## 步骤 4：渐进式代码生成

**模板分支（MUST）**：
- `meta.op_kind == fused_compute_comm` → **只读** [templates/fused-compute/GUIDE.md](templates/fused-compute/GUIDE.md)，按章节标题提取代码块；**NEVER** 读 `templates/communication/templates-*.md`
- `transport` / `collective` / `compute` → 只读 communication 子模板（下表）

按"最小正确路径 → 完整正确性 → 性能路径"顺序。**每子步骤只读当前分支对应模板文件**：

1. **CMake + Host**（communication：`templates-cmake-main.md`；fusion：fused-compute GUIDE 对应章节）
2. **Kernel 声明**（communication：`templates-kernel.md`；fusion：fused-compute GUIDE）：`src/<op_name>_kernel.h`
3. **lifecycle + memory**（同上）：初始化、symmetric allocation、释放
4. **transport + sync + compute**（communication 分支；fusion 在 fused-compute kernel 章节实现）：
     - **通算融合**：AIC 必须使用 CATLASS BlockMmad 实现 matmul，AIV 负责 CommBlockEpilogue；禁止在 AIV 上用标量/向量运算替代 Cube 计算
     - **纯通信**：按 design.md 实现 local compute（如有）
      - **GM 累加方式选择**：当 kernel 需要将远端数据累加到 output 时，按 [references/atomic-add-pattern.md](references/atomic-add-pattern.md) §12 决策优先级表选择
5. **测试脚本**（communication：`templates-scripts.md`；fusion：fused-compute GUIDE；若 Phase 2 已生成则跳过）
6. **scheduler**（communication：kernel.cpp；fusion：fused-compute kernel 章节）：phase、tile/chunk/tail、core partition、overlap
7. **README.md**（按 references/readme-spec.md）

### 关键约束

- **算子必须在 Device 执行**，禁止 Host RMA 作为主通信路径
- **main.cpp 只做单 PE Host 编排**：参数解析、lifecycle、I/O、launch、cleanup
- 复杂 Host 逻辑拆到独立 `.cpp/.h`（如 `op_host_plan.cpp`）
- 跨 PE 传输必须使用 `aclshmem_*` 或 `aclshmemx_*` 接口
- symmetric allocation 顺序和大小在所有 PE 一致
- **通算融合算子必须使用 AIC + CATLASS 高性能计算**：AIC 侧 BlockMmad（或同等实现），AIV 侧 CommBlockEpilogue；禁止在 AIV 上用标量/向量点乘替代 AIC 计算
- `block_dim=1` 仅临时调试用；首版 correctness 必须落地 design 的并发
- 新增核心能力必须对应 gap analysis

### 性能输出要求

main.cpp 的 `--perf` 模式必须输出双指标延迟和带宽（严格按照 [shmem-ops-performance-eval/references/timing-and-metrics-standard.md](../shmem-ops-performance-eval/references/timing-and-metrics-standard.md) 执行）：

| 指标 | 说明 | 公式 |
| --- | --- | --- |
| `e2e_us` | 端到端延迟（含输入到对称堆的拷贝 + barrier + kernel） | 从输入到对称堆的拷贝前到 stream sync 后 |
| `kernel_us` | 纯 kernel 延迟 | 从 kernel launch 前到 stream sync 后 |
| `algo_bandwidth_GBps` | 算法带宽（基于 e2e_us，参考） | `logical_payload_bytes / (e2e_us * 1e-6) / 1e9` |
| `bus_bandwidth_GBps` | 总线带宽（e2e 参考） | `algo_bandwidth * bus_factor` |
| `steady_bus_bandwidth_GBps` | **达标主指标**（kernel 口径） | `logical_payload_bytes / (kernel_us * 1e-6) / 1e9 * bus_factor` |
| `bandwidth_utilization_pct` | 带宽利用率（基于 e2e bus，参考） | `bus_bandwidth / peak_bandwidth * 100`（peak_bandwidth 按通信模式确定，见下文） |

`bus_factor` 按算子语义确定（不区分拓扑，NCCL 惯例的通信量标准化系数）：AllReduce: `2*(n-1)/n`，ReduceScatter/AllGather: `(n-1)/n`，AllToAll/Shuffle: `(n-1)/n`，Broadcast/P2P: 1。

`peak_bandwidth` 按通信模式确定（参考 [hardware-architecture.md §2.6](../shmem-ops-design/references/hardware-architecture.md)）：
- P2P 点对点：28 GB/s（单条 HCCS 链路单向）
- 集合通信（AllReduce/AllGather 等）：聚合带宽，如 910B3 8 卡 full-mesh 为 7 × 28 = 196 GB/s

perf 循环结构要求：
- **e2e 循环**：每轮包含输入到对称堆的拷贝 + barrier + kernel launch，模拟真实调用模式
- **kernel-only 循环**：输入到对称堆的拷贝已完成，只测 kernel launch，用于内部优化定位
- **禁止跳过输入到对称堆的拷贝**：真实使用场景中 input 来自用户 Device GM buffer，性能测试 **MUST** 包含完整的 `input(GM) → 输入到对称堆的拷贝（MTE / SDMA / aclrtMemcpy 等均可） → symmetric memory → kernel` 流程。输入到对称堆的拷贝 **MUST** 计入 `e2e_latency_us`，不得在性能循环前预拷贝一次使 e2e 口径缩水

注意：`algo_bandwidth` 不乘 2，统一按 input size 计算（NCCL `algBw` 惯例）。`logical_payload_bytes` 必须在输出中注明口径（单 PE 还是全局）。**Phase 6 达标与 Round 对比 MUST 用 `[PERF]` 行的 `steady_bus_bandwidth_GBps`**，不得用 e2e 带宽。

如果算子包含 `SHMEMI_PROF_START/END` 打点，`--perf` 模式还应调用 `aclshmemx_get_prof(nullptr, true)` 输出 Device 帧数据。

---

## 步骤 5：编译验证

调用 `shmem-ops-compile-debug`，传入 compile contract。持续修复直到编译通过或确认阻塞。

---

## 最终目标目录结构

**默认根路径**：`custom-ops/<op_name>/`（独立工程）。in-tree 时为 `examples/<op_name>/`。

最终交付目录结构 **MUST** 严格遵循以下布局（以下以 `<op_root>/` 表示算子根目录）：

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
│   ├── run_case_matrix.py
│   ├── perf.sh                      # Phase 6；实现见 perf-workflow.md
│   └── perf_compare.sh              # 有 baseline 时；或统一用 perf-workflow §1 阶段 C
└── baseline/                       # 有 baseline 时 MUST 存在
    ├── CMakeLists.txt              # 独立的 baseline 编译 target（add_subdirectory）
    ├── src/
    │   └── op_name_baseline.cpp    # HCCL/aclnn baseline 源码
    └── scripts/
        └── run_baseline.sh         # baseline 运行脚本，输出 [BASELINE_PERF]
```

---

## 检查点

- [ ] 目录结构符合规范：`docs/` 承载所有阶段产出的 `.md`、`src/` 含全部算子 `.cpp/.h`、`baseline/` 含全部 baseline 源码和编译配置、`scripts/` 含全部测试脚本
- [ ] CMake 生成，target 命名与 op_name 一致
- [ ] 代码遵循 design.md 的 phase/transport/sync/partition
- [ ] 默认 unified single path；无 profiling ≥5% 收益证据则不引入 size 分支（见 code-patterns.md §6.4）
- [ ] main.cpp 不含复杂逻辑
- [ ] main.cpp `--perf` 模式输出双指标延迟（e2e_us、kernel_us）、algo/bus 带宽及 **`steady_bus_bandwidth_GBps`**
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
- ❌ 一次性读取全部 communication 模板文件（必须按步骤只读当前子文件）
- ❌ design.md 门禁有 FAIL 项仍继续生成代码（"先跑通再补设计"是最常见的违规路径）
- ❌ GM 标量循环累加作为性能路径交付（按 atomic-add-pattern.md §12 决策表选择正确方式）
- ❌ 修改 SHMEM 核心库却无 gap analysis 授权
- ❌ 将输入到对称堆的拷贝排除在 e2e 性能循环外（性能循环前预拷贝使 e2e 口径缩水）
- ❌ `.md` 文档散放在算子根目录（必须归入 `docs/`）
- ❌ `.cpp/.h` 源文件散放在算子根目录（必须归入 `src/`）
- ❌ baseline 源码放在 `src/` 下或算子根目录（必须归入 `baseline/src/`，编译 target 必须在 `baseline/CMakeLists.txt`）
- ❌ 用户未要求 in-tree 时将算子生成到 `examples/`（默认 `custom-ops/<op_name>/`）
- ❌ 通算融合算子在 AIV 上用标量/向量点乘替代 AIC + CATLASS 高性能计算（即使以"正确性验证"为由也不允许——须使用 CATLASS BlockMmad + CommBlockEpilogue 的 AIC/AIV 分工模式）
