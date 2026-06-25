# SHMEM 算子走读报告模板

走读完成后按以下格式输出报告，保存为算子目录下的 `docs/review-report.md`。

## 走读信息

| 字段 | 内容 |
| --- | --- |
| 算子名称 | |
| design.md | [路径] |
| 代码目录 | [op_dir] |
| 走读时间 | [日期] |
| SoC | |
| PE 数 | |

## 1. 设计一致性检查

| 序号 | 检查项 | 严重级别 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | CMake 构建模式与 design 一致 | P0 | PASS/FAIL | |
| 2 | 跨 PE 搬运使用 SHMEM 数据面接口 | P0 | PASS/FAIL | |
| 3 | Transport API 与 design 一致 | P0 | PASS/FAIL | |
| 4 | Signal/Wait 配对正确 | P0 | PASS/FAIL | |
| 5 | Phase 边界同步与 design 一致 | P1 | PASS/FAIL | |
| 6 | Block dim 与 design core_partition 一致 | P0 | PASS/FAIL | |
| 7 | Tile/Chunk/Tail 处理正确 | P1 | PASS/FAIL | |
| 8 | main.cpp 不含复杂逻辑 | P1 | PASS/FAIL | |
| 9 | 高性能 kernel 路径已接入（通算融合必须使用 AIC+CATLASS） | P0 | PASS/FAIL | |
| 10 | 无多余 GM scratch 中转 | P2 | PASS/FAIL | |
| 11 | Symmetric allocation 顺序和大小所有 PE 一致 | P0 | PASS/FAIL | |
| 12 | 代码风格符合 code-style.md | P2 | PASS/FAIL | |

严重级别：P0=致命（必须修复）、P1=严重（强烈建议修复）、P2=建议（优化项）。

## 2. 正确性验证结果

### 2.1 汇总

| 指标 | 值 |
| --- | --- |
| 总用例数 | |
| 通过数 | |
| 失败数 | |
| 通过率 | % |
| 阻塞（未执行）数 | |

### 2.2 全量 Case 结果

**必须列出 case matrix 中的全部 case**，不得省略。

| Case ID | category | scale | n_pes | shape | dtype | Result | MaxAE | MaxRE | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_2pe_fp16 | functional | XS | 2 | (128, 64) | fp16 | PASS | 0.0 | 0.0 | |
| small_2pe_fp16 | functional | S | 2 | (512, 256) | fp16 | PASS | 0.0 | 0.0 | |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 2.3 Invariant 验证

| Invariant | 验证方式 | 结果 | 说明 |
| --- | --- | --- | --- |
| symmetric allocation 顺序一致 | rank pattern | PASS/FAIL/未验证 | |
| put/get 语义正确 | CPU golden | PASS/FAIL/未验证 | |
| signal/wait 完成语义 | 多 PE 执行 | PASS/FAIL/未验证 | |
| output visibility | 比较输出 | PASS/FAIL/未验证 | |
| tail/不均衡分片 | boundary case | PASS/FAIL/未验证 | |

## 3. 性能评估摘要

| 指标 | 值 |
| --- | --- |
| 性能报告 | [performance_report.md 路径] |
| 基线类型 | HCCL / 拼接 / metric_only |
| 达标状态 | 达标 / 未达标 |
| 优化轮次 | /5 |

### 性能关键指标

| 指标 | Baseline | Final | Δ% |
| --- | --- | --- | --- |
| e2e_latency_us | | | |
| kernel_latency_us | | | |
| algo_bandwidth_GBps | | | |
| bus_bandwidth_GBps | | | |

（完整优化记录见 performance_report.md）

## 4. 已知限制与风险

| 编号 | 类别 | 描述 | 严重级别 | 状态 |
| --- | --- | --- | --- | --- |
| R1 | | | P0/P1/P2 | 已修复/待修复/接受风险 |

## 5. 结论

- **设计一致性**：PASS / FAIL（P0 项全部通过为 PASS）
- **正确性验证**：PASS / FAIL（全部 case 通过为 PASS）
- **性能达标**：达标 / 未达标
- **总体结论**：**可交付** / **需修复**（设计一致性 PASS + 正确性 PASS = 可交付）

### FAIL 项修复建议

（如有 FAIL 项，逐项列出修复建议）

## 6. 交付物清单

| 交付物 | 路径 | 状态 |
| --- | --- | --- |
| design.md | docs/ | ✅ / ❌ |
| 算子代码 | src/ | ✅ / ❌ |
| 测试脚本 | scripts/ | ✅ / ❌ |
| review-report.md（本文档） | docs/ | ✅ / ❌ |
| performance_report.md | docs/ | ✅ / ❌ |
