---
name: shmem-ops-dev
description: "SHMEM 通信算子与通算融合算子端到端开发编排器。当用户需要从需求、伪代码或参考实现完成一个 SHMEM 算子时使用。关键词：shmem、通信算子、通算融合、端到端、算子开发、全流程。"
---

# SHMEM 算子端到端开发编排

**Skill类型**：流程导向型（八阶段工作流，子技能串行编排）

本 skill 编排八个子 skill，驱动 SHMEM 通信算子或通算融合算子从需求到可交付状态。

## 核心原则

0. **中文写作**：所有交付的 `.md` 文档（design.md、review-report.md、performance_report.md 等）必须使用中文撰写。仅 API 名称、代码片段、变量名、数学公式等技术术语保留英文原文，其余描述、分析、结论一律使用中文
1. **阶段串行（含条件性子阶段）**：需求确认 → 设计 → 用例生成 → 代码生成 → 编译正确性 → 代码走读 → 性能采集 → 达标进入最终交付；未达标进入性能优化（条件性 Phase 6.5，最多 5 轮）→ 最终交付，严格顺序执行
2. **子技能执行**：每个阶段 **MUST** 调用对应子 skill，不得自行实现
3. **阶段门控**：前一阶段检查点全部通过后才进入下一阶段
4. **设计驱动编码**：代码生成依赖设计文档中的 Canonical DSL 和六类模块设计
5. **先 correctness 再 performance**：功能或正确性未通过时，不做性能优化
6. **性能优化上限 5 轮**：每轮记录假设、改动、正确性复测和性能结果
7. **结果可视化**：关键结果在聊天界面展示，不要仅输出路径
8. **目录结构强制**：所有 `.md` 文档归入 `docs/`，所有 `.cpp/.h` 源文件归入 `src/`，测试脚本归入 `scripts/`。不得扁平散放
9. **不静默修改核心库**：只有 gap analysis 明确授权时才允许改 SHMEM 核心能力

## 可用子 Skill 清单

| Skill | 路径 | 职责 |
| --- | --- | --- |
| `shmem-ops-design` | `shmem-ops-design/SKILL.md` | 将需求转化为 design.md（Canonical DSL + 契约） |
| `shmem-ops-testcase-gen` | `shmem-ops-testcase-gen/SKILL.md` | 生成 case matrix、golden/checker、测试脚本 |
| `shmem-ops-code-gen` | `shmem-ops-code-gen/SKILL.md` | 根据 design.md 生成算子代码、CMake、README |
| `shmem-ops-compile-debug` | `shmem-ops-compile-debug/SKILL.md` | 编译、运行、失败分类和调试 |
| `shmem-ops-code-review` | `shmem-ops-code-review/SKILL.md` | 实现与设计一致性走读 |
| `shmem-ops-correctness-eval` | `shmem-ops-correctness-eval/SKILL.md` | 正确性契约验证和报告 |
| `shmem-ops-performance-eval` | `shmem-ops-performance-eval/SKILL.md` | 性能采集、baseline 对比、瓶颈分析 |
| `shmem-ops-performance-optim` | `shmem-ops-performance-optim/SKILL.md` | 性能优化迭代（最多 5 轮） |

## 工作流总览

```
Phase 0       Phase 1       Phase 2         Phase 3        Phase 4            Phase 5       Phase 6            ┌ 达标 → Phase 7
需求环境确认 ──▶ 设计文档 ──▶ 用例生成 ──▶ 代码生成 ──▶ 编译+正确性 ──▶ 代码走读 ──▶ 性能采集 ──▶ ┤
—              design     testcase-gen   code-gen    compile-debug    code-review  performance   │ 未达标 → Phase 6.5 性能优化 → Phase 7
                                                     + correctness                  -eval         └

输入: 算子需求 + 环境                            输出: 可交付算子 + 测试 + 性能报告
```

## 反模式清单（NEVER DO THESE）

- ❌ 跳过设计阶段直接写代码
- ❌ 跳过用例生成，直接进入代码生成
- ❌ 自行实现算子代码，不调用子 skill
- ❌ correctness 失败时做性能优化
- ❌ 未经用户明确要求，性能优化超过 5 轮
- ❌ 将 `block_dim=1` 补齐到设计并发当作性能优化轮次
- ❌ 有可用 baseline（HCCL/aclnn/拼接）但不接入
- ❌ 编造硬件测试或 profiler 结果
- ❌ 修改 SHMEM 核心库却无 gap analysis 授权
- ❌ 引用不存在的子 skill

---

## Phase 0：需求与环境确认

**调用 Skill**：—

### 执行内容

确认本次任务入口和环境：

| 输入形态 | 处理方式 |
| --- | --- |
| 只有自然语言需求 | 收集最小需求后进入 Phase 1 |
| 有伪代码或异构参考实现 | 进入 Phase 1，设计阶段归一化 |
| 已有 design.md | 直接进入 Phase 1 质量门禁 |
| 用户说"继续开发" | 按中断恢复矩阵从最早未完成阶段恢复 |

最小确认项：
- `op_name`（可用作目录名和符号前缀）
- 功能语义（local compute、communication、final result）
- 目标 SoC 和 dtype
- CANN `set_env.sh` 路径（用户指定，禁止硬编码默认路径）
- Python 可执行文件或环境激活命令
- SHMEM 仓库路径（默认 `shmem/`）

### 检查点

- [ ] op_name 已确认
- [ ] 功能语义明确
- [ ] CANN 环境路径已确定（或标记未提供）
- [ ] Python 环境已确定（或标记未提供）

**全部通过 → 进入 Phase 1**

---

## Phase 1：设计文档

**调用 Skill**：`shmem-ops-design`

### 执行内容

```
1. 如果无 design.md，使用 shmem-ops-design 生成
2. 如果已有 design.md，读取并检查质量门禁
3. 设计文档必须包含：Canonical DSL、capability mapping、gap analysis、
   correctness invariants、compile/test/perf contract
```

### 检查点

- [ ] design.md 存在且非占位
- [ ] 完整 Canonical DSL yaml block（含 schedule、correctness、performance section）
- [ ] DSL `topology` 含 deployment、拓扑类型、链路带宽（来自设计前确认）
- [ ] 六类模块完整（lifecycle/memory/transport/sync/compute/scheduler）
- [ ] capability mapping 分类完整
- [ ] correctness invariants 可测试
- [ ] compile/test/perf contract 可执行
- [ ] Design Review Before Handoff（section 5）六项检查已填写且无 FAIL
- [ ] 门禁检查结果已显式输出给用户

Phase 1 的 design.md 门禁必须在进入 Phase 2 前完成。门禁结果中有任何 FAIL 或缺失 section，必须先修订 design.md 再继续，不允许带缺陷进入后续阶段。

**全部通过 → 进入 Phase 2**

---

## Phase 2：测试用例生成

**调用 Skill**：`shmem-ops-testcase-gen`

### 执行内容

```
1. 读取 design.md 的 correctness contract 和 invariants
2. 生成 case matrix（smoke/contract/tail/repeats/gap/medium-scale）
3. 生成 gen_data.py、check_result.py、run.sh
4. 生成 golden 文件
```

### 检查点

- [ ] case matrix 覆盖完整
- [ ] gen_data.py 可独立运行
- [ ] check_result.py 返回明确退出码
- [ ] 包含中等规模 case（集合通信≥256MB；融合 hidden≥1000）

**全部通过 → 进入 Phase 3**

---

## Phase 3：代码生成

**调用 Skill**：`shmem-ops-code-gen`

### 执行内容

```
1. 读取通过门禁的 design.md
2. 选择模板和参考 example
3. 渐进式代码生成（lifecycle → transport → compute → scheduler）
4. 生成 README.md
5. 调用 shmem-ops-compile-debug 编译验证
```

### 检查点

- [ ] 代码按 design 的 phase/transport/sync/partition 实现
- [ ] 目录结构符合规范：`docs/` 含全部 `.md`、`src/` 含全部 `.cpp/.h`、`scripts/` 含全部测试脚本
- [ ] main.cpp 不含复杂逻辑
- [ ] README.md 覆盖编译、运行、校验入口
- [ ] e2e_latency_us 口径包含输入到对称堆的拷贝（未预拷贝缩水）
- [ ] 编译成功（或记录环境阻塞）

**全部通过 → 进入 Phase 4**

---

## Phase 4：编译与正确性闭环

**调用 Skill**：`shmem-ops-compile-debug` + `shmem-ops-correctness-eval`

### 执行内容

```
1. 使用 shmem-ops-compile-debug 按 compile contract 构建
2. 编译失败持续修复直到通过或确认阻塞
3. 编译成功后，使用 shmem-ops-correctness-eval 执行 case matrix
4. 失败分类：design bug → Phase 1；code bug → Phase 3；env → 记录阻塞
```

### 检查点

- [ ] 编译成功
- [ ] 全部 case 通过（或分类标记未通过原因）
- [ ] 中等规模 case 已验证（或标记未满足）
- [ ] invariants 已验证

**全部通过 → 进入 Phase 5**

---

## Phase 5：实现与设计一致性走读

**调用 Skill**：`shmem-ops-code-review`

### 执行内容

```
1. 对比 design.md 和实现代码
2. 逐项检查：CMake target、block_dim、transport API、tile/chunk/tail、offset、性能路径
3. 输出走读结果表
```

### 检查点

- [ ] 走读全部 PASS
- [ ] 无固定单核 launch（除非 design 明确要求）
- [ ] 无未接入的高性能 kernel

**全部通过 → 进入 Phase 6**

---

## Phase 6：性能采集

**调用 Skill**：`shmem-ops-performance-eval`

### 执行内容

```
1. 读取 perf contract
2. 选择 baseline（CANN/拼接/metric-only）
3. 采集当前实现性能
4. 计算 delta 和利用率
5. 瓶颈分析
6. 输出性能报告
```

### 检查点

- [ ] baseline 已选择或记录搜索过程
- [ ] 性能数据已采集
- [ ] 包含 L 档（大规模）case
- [ ] 瓶颈已分析
- [ ] 有 baseline 时：达到 80% → Phase 7 交付；未达 80% → 进入性能优化
- [ ] 无 baseline 时：满足指标 → Phase 7；不满足 → 进入性能优化

**达标 → 进入 Phase 7（交付）；未达标 → 进入性能优化**

---

## Phase 6.5：性能优化（最多 5 轮）

**调用 Skill**：`shmem-ops-performance-optim`

### 执行内容

```
每轮：
1. hypothesis（瓶颈假设）
2. 单一方向改动
3. 编译（shmem-ops-compile-debug）
4. 正确性复测（shmem-ops-correctness-eval）
5. 性能采集（shmem-ops-performance-eval，optim 调用 eval 时 MUST 传递 `target_section` / `round_label` / `is_optimal_step` 上下文）
6. 决策：keep / revise / revert
```

### 停止条件

- 已完成 5 轮（唯一正常停止条件——即使已达标也 **MUST** 跑满 5 轮）
- 5 轮后仍未达标 → 输出差距和瓶颈，进入 Phase 7 交付
- 用户明确要求继续优化 → 可超过 5 轮，每额外轮次遵循完整流程
- 需修改核心库但无 gap analysis 授权

### 硬性限制

- `block_dim=1` → 设计并发 **不计入**优化轮次
- 正确性失败的轮次不采纳性能结果
- 无瓶颈分析不得开始优化

**完成 → 进入 Phase 7**

---

## Phase 7：最终交付

**调用 Skill**：—

### 执行内容

输出最终交付摘要，必须包含：

- 算子名称和 `docs/design.md` 路径
- 目录结构检查（`docs/` 包含全部 `.md`、`src/` 包含全部 `.cpp/.h`、`scripts/` 包含全部测试脚本）
- 修改文件列表
- 编译命令（完整可复现）
- 测试命令（完整可复现）
- 正确性验证结果表
- 性能采集结果和 baseline 对比（`e2e_latency_us` 必须包含输入到对称堆的拷贝时间）
- 优化轮次摘要（如有）
- 未完成项、未验证项、环境阻塞或风险

不要只输出路径；关键结果必须在回复中摘要展示。

---

## 阶段间数据流

```
Phase 0 输出                Phase 1 输入
  op_name、env 确认   ────▶   需求语义、SoC、dtype

Phase 1 输出                Phase 2 输入
  design.md (完整)    ────▶   correctness contract、invariants

Phase 2 输出                Phase 3 输入
  case matrix         ────▶   design.md
  gen_data/checker             复用计划、模板选择

Phase 3 输出                Phase 4 输入
  算子代码、CMake     ────▶   compile contract
  README.md                    test contract

Phase 4 输出                Phase 5 输入
  编译通过            ────▶   design.md vs 实现代码
  correctness PASS

Phase 5 输出                Phase 6 输入
  走读 PASS           ────▶   perf contract
                               baseline 策略

Phase 6 输出                Phase 6.5/7 输入
  性能报告            ────▶   达标判断
  瓶颈分析                     优化或交付
```

## 状态跟踪表

| Phase | 前置条件 | 调用 Skill | 关键产出物 |
| --- | --- | --- | --- |
| 0 | 无 | — | op_name + 环境确认 |
| 1 | Phase 0 | `shmem-ops-design` | design.md（Canonical DSL + 契约） |
| 2 | Phase 1 | `shmem-ops-testcase-gen` | case matrix + scripts |
| 3 | Phase 2 | `shmem-ops-code-gen` | 算子代码 + README |
| 4 | Phase 3 | `shmem-ops-compile-debug` + `shmem-ops-correctness-eval` | 编译通过 + correctness PASS |
| 5 | Phase 4 | `shmem-ops-code-review` | 走读 PASS |
| 6 | Phase 5 | `shmem-ops-performance-eval` | 性能报告 |
| 6.5 | Phase 6 未达标 | `shmem-ops-performance-optim` | 优化轮次记录 |
| 7 | Phase 6 达标 或 6.5 完成 | — | 最终交付摘要 |

## 错误恢复

### 中断恢复矩阵

| 检测条件 | 判定阶段 | 恢复动作 |
| --- | --- | --- |
| 无 design.md | Phase 1 未完成 | 使用 `shmem-ops-design` 生成 |
| design.md 缺 Canonical DSL | Phase 1 未完成 | 修订设计并重跑门禁 |
| 无测试脚本 | Phase 2 未完成 | 使用 `shmem-ops-testcase-gen` |
| 有 design 无实现代码 | Phase 3 未完成 | 使用 `shmem-ops-code-gen` |
| 代码存在但编译失败 | Phase 4 未完成 | 从编译调试恢复 |
| 编译通过但 correctness 未过 | Phase 4 未完成 | 从正确性调试恢复 |
| correctness 通过但无走读 | Phase 5 未完成 | 执行 `shmem-ops-code-review` |
| 走读通过但无性能数据 | Phase 6 未完成 | 执行性能采集 |
| 有性能数据但优化未完成 | Phase 6.5 未完成 | 从下一轮优化继续 |

### 编译/测试失败

由 `shmem-ops-compile-debug` 内部处理，持续修复直到正确性通过或确认为环境阻塞/设计缺陷。
