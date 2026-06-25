---
name: shmem-ops-code-review
description: "SHMEM 算子实现与设计一致性走读，生成最终交付的 review-report.md。关键词：code review、design review、一致性检查、走读、交付报告。"
---

# SHMEM 算子实现与设计一致性走读

**Skill类型**：检查型（对比 design.md 和实现代码，输出走读报告）

> **中文写作要求**：review-report.md 必须使用中文撰写。仅 API 名称、代码片段、检查项英文缩写等技术术语保留英文原文。

正确性通过前后都必须执行。通过前用于补齐实现，通过后作为进入性能阶段的门禁。**最终交付时必须生成完整的 `review-report.md`**。

## 工作流

```
步骤 1  读取 design.md 和实现代码
步骤 2  逐项检查（CMake/block_dim/transport/sync/compute/style）
步骤 3  输出走读结果表
步骤 4  生成 review-report.md（含正确性全量结果 + 性能摘要 + 交付物清单）
```

---

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/code-review-checklist.md](references/code-review-checklist.md) | 分类检查项、错误/正确做法示例 |
| [references/review-report-template.md](references/review-report-template.md) | 走读报告输出模板（必须严格按此模板生成） |
| [../shmem-ops-code-gen/references/code-style.md](../shmem-ops-code-gen/references/code-style.md) | 代码风格规范，走读时必须对照检查 |

---

## 检查项

必须逐项检查并记录结果：

| 检查项 | 严重级别 | 标准 |
| --- | --- | --- |
| CMake target | P0 | 包含 design 声明的所有 kernel、Host helper、transport 源文件 |
| launch block_dim | P0 | 来自 design 或参数化；固定 `<<<1>>>` 不能作为交付（除非 design 明确要求单核） |
| transport API | P0 | design 声明的 API 实际出现在编译路径中 |
| 跨 PE 传输 | P0 | 使用 `aclshmem_*` / `aclshmemx_*`，不用 DataCopy 写远端 |
| signal/wait 配对 | P0 | 每个 signal 有对应 wait，magic/epoch 策略一致 |
| symmetric 分配顺序 | P0 | 所有 PE 的 symmetric malloc 调用顺序和大小完全一致 |
| AIC+CATLASS（通算融合） | P0 | 通算融合必须使用 AIC+CATLASS BlockMmad 计算，禁止 AIV 标量点乘替代 |
| phase 边界同步 | P1 | 与 design 一致 |
| tile/chunk/tail | P1 | 按 design 支持并发；非对齐 case 不永久降级为单核 |
| main.cpp 边界 | P1 | 不含复杂计算逻辑；复杂 Host 逻辑已拆模块 |
| 性能关键路径 | P1 | 无设计未说明的多余 GM scratch、全局 barrier 或串行 phase |
| e2e 计时口径 | P0 | `e2e_latency_us` 必须包含输入到对称堆的拷贝 + barrier + kernel；`kernel_latency_us` 只包含 kernel launch。两者不得相等（除非输入到对称堆的拷贝时间可忽略且代码中有注释说明）。严禁在性能循环前预拷贝对称堆使 e2e 口径缩水 |
| 代码风格 | P2 | 对照 `code-style.md` 检查 |

---

## 走读报告要求

走读完成后必须在算子目录生成 `docs/review-report.md`，**严格按照** [references/review-report-template.md](references/review-report-template.md) 格式。

报告必须包含以下 6 个 section，缺少任何一个视为不完整：

### Section 1：设计一致性检查
- 逐项列出检查结果，标注严重级别（P0/P1/P2）
- P0 项任一 FAIL → 总体 FAIL

### Section 2：正确性验证结果
- **汇总表**：总用例数、通过数、失败数、通过率
- **全量 case 结果表**：必须列出 case matrix 中的**全部 case**，不得省略或只列部分
- **invariant 验证表**：逐项标注验证结果

### Section 3：性能评估摘要
- 引用 `performance_report.md` 路径
- 摘录最终对比表（baseline vs final + Δ%）
- 标注达标状态

### Section 4：已知限制与风险

### Section 5：结论
- 设计一致性、正确性、性能三项独立判定
- 总体结论：可交付 / 需修复

### Section 6：交付物清单
- 列出所有必须交付的文件及状态

### 报告自检

生成 review-report.md 后，逐项自检：

1. 是否包含全部 6 个 section？缺少任一个 → 补齐后再交付
2. Section 2 是否列出了 case matrix 中的**全部** case？只列部分 → 补齐
3. Section 3 是否引用了 performance_report.md 并摘录了最终对比表？
4. Section 6 是否列出了全部交付物及其状态？

自检不通过不得声称走读完成。

---

## 最终交付物（必须全部齐备）

| 交付物 | 说明 |
| --- | --- |
| `docs/design.md` | 设计文档（来自 shmem-ops-design） |
| 算子代码 | `src/` + `CMakeLists.txt`（来自 shmem-ops-code-gen） |
| 测试脚本 | `scripts/gen_data.py` + `check_result.py` + `run.sh` + `run_case_matrix.py`（来自 shmem-ops-testcase-gen） |
| `docs/review-report.md` | 走读报告（本 skill 生成，包含正确性全量结果） |
| `docs/performance_report.md` | 性能优化报告（来自 shmem-ops-performance-optim） |
| `docs/correctness_report.md` | 正确性验证报告（来自 shmem-ops-correctness-eval） |
| `docs/case_matrix_report.md` | Case Matrix 执行报告（来自 shmem-ops-testcase-gen） |

**缺少任一交付物不得声称算子完成。**

---

## 门控规则

- 任一 P0 检查项 FAIL → 不能进入性能采集
- 正确性未全部通过 → 不能进入性能采集
- FAIL 时回 `shmem-ops-code-gen` 修正实现，或回 `shmem-ops-design` 修订设计

---

## 检查点

- [ ] 所有检查项已逐项检查并记录
- [ ] P0 项全部 PASS
- [ ] 正确性全量 case 结果已列出（不得省略）
- [ ] invariant 验证已列出
- [ ] 性能摘要引用了 performance_report.md
- [ ] review-report.md 已按模板格式生成
- [ ] 交付物清单完整
- [ ] baseline 源码在 `baseline/src/` 下、编译 target 在 `baseline/CMakeLists.txt`，未混入算子 `src/` 或根目录

---

## 反模式（NEVER DO THESE）

- ❌ 不读 design.md 直接走读代码
- ❌ 只看代码能否编译通过，不检查与设计的语义一致性
- ❌ 发现 `block_dim=1` 但不标记为 FAIL
- ❌ 发现 `DataCopy` 远端地址但不标记为 FAIL
- ❌ 不对照 `code-style.md` 检查代码风格
- ❌ 正确性结果只列出部分 case（必须列出全部 case matrix 中的 case）
- ❌ 走读报告中不包含正确性全量结果
- ❌ 走读报告中不包含性能摘要
- ❌ 不生成 review-report.md 就声称走读完成
- ❌ 交付物缺少 performance_report.md 却声称算子完成
- ❌ 发现输入到对称堆的拷贝不在 e2e 计时范围内却不标记为 FAIL（必须标记为 P0 FAIL）
- ❌ review-report.md 不在 `docs/` 子目录下
- ❌ baseline 源码混入算子 `src/` 或散放在算子根目录（必须归入 `baseline/src/`）
- ❌ 交付物清单声明的文件在磁盘上不存在
