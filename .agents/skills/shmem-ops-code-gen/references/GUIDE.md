# References Guide

本目录包含 `shmem-ops-code-gen` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [api.md](api.md) | SHMEM API 分类和选择参考 | 选择 lifecycle/memory/transport/sync API 时 |
| [code-patterns.md](code-patterns.md) | Host/Device 代码组织模式 | 步骤 4 代码生成时 |
| [atomic-add-pattern.md](atomic-add-pattern.md) | `SetAtomicAdd<T>()` 累加：安全顺序、边界、风险 | 实现 reduce/累加时 |
| [code-style.md](code-style.md) | C/C++ 代码规范和审查清单 | 代码生成后审查时 |
| [readme-spec.md](readme-spec.md) | 算子 README.md 格式要求 | 生成 README 时 |
