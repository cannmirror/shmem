# SHMEM Agent Assets

AI Agent 配置资产，用于辅助 SHMEM 通信算子与通算融合算子的开发。

> **声明**：本目录下的 Skills 仅用于辅助 AI Agent 生成 SHMEM 算子代码与文档，**不作性能保证、泛化性保证或生产可用性承诺**。生成的代码需经人工审查、编译验证和充分测试后方可用于生产环境。

## 目录结构

```
.agents/
├── README.md           ← 本文件
└── skills/             ← SHMEM 算子开发 Skill 工具集
    ├── README.md       ← Skill 全景文档（工作流 / Phase / 产物 / 布局）
    ├── shmem-ops-dev/       ← 编排器：Phase 0-7 全流程调度
    ├── shmem-ops-design/   ← Phase 1：算子设计 → design.md
    ├── shmem-ops-testcase-gen/  ← Phase 2：用例生成 → case matrix + golden/checker
    ├── shmem-ops-code-gen/      ← Phase 3：代码生成 → src/ + CMakeLists.txt
    ├── shmem-ops-compile-debug/ ← Phase 4：编译与调试
    ├── shmem-ops-correctness-eval/ ← Phase 4：正确性验证
    ├── shmem-ops-code-review/     ← Phase 5：设计-代码一致性走读
    ├── shmem-ops-performance-eval/   ← Phase 6：性能基线采集
    └── shmem-ops-performance-optim/  ← Phase 6.5：性能优化迭代（5 轮，条件性）
```

## 工作流

9 个 Skill 按 Phase 0-7 状态机串行编排，覆盖算子从需求到交付的全流程：

```
Phase 0 需求确认 → 1 设计 → 2 用例 → 3 代码 → 4 编译+正确性 → 5 走读 → 6 性能基线 → 6.5 性能优化（条件性） → 7 最终交付
```

阶段门控：前一阶段检查点全部通过后才进入下一阶段。详见 [skills/README.md](skills/README.md)。

## 快速开始

用自然语言描述算子需求，编排器自动按 Phase 0-7 推进：

> 基于 shmem 实现 alltoallv 算子，数据类型 int/fp16，单机 8 卡，full-mesh，p2p 带宽 28GB/s

或结合伪代码输入：

> 基于这个 moe dispatch 伪代码，在 custom-ops（自定义独立算子工程）目录下实现 moe dispatch 算子，数据类型 fp16，单机 8 卡 full-mesh

## 边界声明

| 项目 | 说明 |
|------|------|
| **用途** | 仅辅助 Agent 生成 SHMEM 算子代码与文档，不替代人工开发 |
| **性能保证** | 不承诺生成代码的性能指标（带宽、延迟等）达到任何特定标准 |
| **正确性保证** | 不承诺生成代码一次性通过正确性验证，需人工审查与迭代 |
| **生产可用性** | 生成代码仅供参考与学习，生产部署前须完成完整测试与审查 |
| **影响范围** | Skills 不修改 SHMEM 核心库（`src/`、`include/`），生成内容应限定在用户指定的算子工程或示例目录 |
| **构建/运行** | Skills 不参与 SHMEM 库本身的构建、打包或运行流程 |

本目录内容为 Markdown 工作流资产，不属于 SHMEM 构建、打包、运行或公共 API 的一部分。
