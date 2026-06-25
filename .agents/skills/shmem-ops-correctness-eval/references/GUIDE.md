# References Guide

本目录为 `shmem-ops-correctness-eval` 提供正确性验证参考资料索引。

当前正确性验证复用 testcase-gen 的契约和精度标准，索引保留在本 skill 本地，避免执行时遗漏跨 skill 参考。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [correctness.md](../../shmem-ops-testcase-gen/references/correctness.md) | golden 构造策略、invariant 到测试映射、失败分类依据 | 执行 case matrix 和验证 invariants 时 |
| [precision-standard.md](../../shmem-ops-testcase-gen/references/precision-standard.md) | OpTypes 分类、rtol/atol 取值、双统计判定 | 判断 dtype 精度和生成/复核 checker 结果时 |
