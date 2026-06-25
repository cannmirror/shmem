# 参考资料索引

本目录包含 `shmem-ops-performance-eval` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [performance-eval-guide.md](performance-eval-guide.md) | 采集流程概览、contract 字段、结果表格式、规模要求 | 步骤 1-6 全过程 |
| [baseline-selection.md](baseline-selection.md) | baseline 选择策略、HCCL/aclnn 清单、拼接方案、决策树 | 步骤 1 baseline 选择时 |
| [timing-and-metrics-standard.md](timing-and-metrics-standard.md) | 时间打点标准、指标计算公式、Device 打点、计算指标、采集方法 | 步骤 2-4 性能采集和对比时 |
| [profiling-tools.md](profiling-tools.md) | msprof、SHMEM cycle profiling 使用方法 | 步骤 2 性能采集时 |
| [device-profiling-guide.md](device-profiling-guide.md) | SHMEMI_PROF 完整操作指南 | 步骤 2 Device 打点采集时 |

跨 skill 参考：

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [hardware-architecture.md](../../shmem-ops-design/references/hardware-architecture.md) | 昇腾硬件参数速查（核数、P2P 带宽等） | 计算带宽利用率时 |

模板：

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [performance-report.md](../templates/performance-report.md) | 性能报告模板（7 段式） | 输出报告时 |
