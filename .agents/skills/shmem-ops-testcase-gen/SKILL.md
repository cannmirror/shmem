---
name: shmem-ops-testcase-gen
description: "生成 SHMEM 算子正确性测试计划、case matrix、golden/checker 和测试脚本。关键词：测试用例、case matrix、golden、checker、gen_data、check_result。"
---

# SHMEM 算子测试用例生成

**Skill类型**：文档与脚本生成型（读取设计文档，输出测试计划和验证脚本）

> **中文写作要求**：测试计划和 case matrix 文档必须使用中文撰写。仅 Case ID、代码片段、API 名称等技术术语保留英文原文。

将 `design.md` 中的 correctness contract 和 invariants 转化为可执行的测试计划、数据生成脚本和精度验证脚本。测试计划是实现约束，必须在代码生成前完成。

## 必读资料

| 文件 | 用途 |
| --- | --- |
| [references/correctness.md](references/correctness.md) | 测试规则——golden 构造策略、case matrix 类别、invariant→test 映射 |
| [references/precision-standard.md](references/precision-standard.md) | 精度标准——OpTypes 分类、rtol/atol 取值表、双统计判定、双级兜底 |
| [references/test-structure-template.md](references/test-structure-template.md) | 代码模板——gen_data.py / check_result.py / run.sh（含 timeout）/ main.cpp 边界规则 |
| [references/testcase-scale-standard.md](references/testcase-scale-standard.md) | 规模分档（XS/S/M/L）、边界条件、dtype 覆盖、case matrix 模板、最小 case 数 ≥ 20 |

## 输入

- `design.md` 的 `correctness` 字段：oracle、rtol/atol、invariants、case_matrix
- `design.md` 的 `interface` 字段：inputs/outputs/dtype/shape
- `design.md` 的 `semantics` 字段：local_compute、communication、finalize

## 工作流

```
步骤 1  提取 correctness contract
步骤 2  生成 case matrix
步骤 3  生成数据生成脚本 (gen_data.py)
步骤 4  生成精度验证脚本 (check_result.py)
步骤 5  生成运行脚本 (run.sh)
```

---

## 步骤 1：提取 correctness contract

从 `design.md` 中提取：
- oracle 方案（rank pattern / CPU numpy / PyTorch / file-based）
- tolerance（rtol/atol，按 dtype 区分）
- invariants 列表
- case matrix 规模要求

---

## 步骤 2：生成 case matrix

必须按 [references/testcase-scale-standard.md](references/testcase-scale-standard.md) 的规模分档生成，覆盖 XS/S/M/L 四档 + 边界 + 压力，总 case 数 ≥ 20。类别要求详见 [references/correctness.md](references/correctness.md) §3。

---

## 步骤 3：生成 gen_data.py

职责：
- 固定随机种子（`np.random.seed(42)`），确保可复现
- 为每个 PE 生成独立输入文件（`input_pe{rank}.bin`）
- 计算 golden 输出（使用高精度 float32）
- 保存 golden 文件（`golden_pe{rank}.bin` 或 `golden.bin`）
- 保存配置文件（shape、dtype、PE 数等）

Golden 构造方法（按算子类型选择）：
- **通信算子**：rank pattern — 直接按通信语义构造期望输出
- **计算/融合算子**：CPU numpy/PyTorch 参考实现

---

## 步骤 4：生成 check_result.py

职责：
- 读取算子输出和 golden 文件
- **按 `op_kind` 和 dtype 选择 tolerance**（不能一律使用 `1e-5`）
- 每个 PE 计算并打印 MaxAE、MaxRE、MeanRE 和 tolerance 阈值
- 打印详细 mismatch（PE、index、actual、expected、error）
- 返回明确退出码（0=通过，非0=失败）

Tolerance 选择规则详见 [references/precision-standard.md](references/precision-standard.md)。run.sh 调用 check_result.py 时必须显式传 `--rtol` 和 `--atol`，不依赖脚本内默认值。

---

## 步骤 5：生成 run.sh

多 PE 启动脚本，每个 PE 一个独立进程。

---

## 输出产物

```
op_name/
├── scripts/
│   ├── gen_data.py
│   ├── check_result.py
│   ├── run.sh
│   └── run_case_matrix.py
└── data/              (gen_data.py 生成)
    ├── input_pe0.bin
    ├── input_pe1.bin
    ├── golden.bin
    └── config.json
```

case matrix 执行报告输出到 `docs/case_matrix_report.md`。

---

## 检查点

- [ ] case matrix 覆盖 XS/S/M/L 四档规模（见 testcase-scale-standard.md）
- [ ] case matrix 覆盖所有类别（functional + boundary + stress + performance）
- [ ] 总 case 数 ≥ 20
- [ ] gen_data.py 可独立运行，输出确定性数据
- [ ] check_result.py 返回明确退出码
- [ ] run.sh 支持多 PE 启动
- [ ] 包含中等规模 case（M 档）和大规模 case（L 档），或标记未满足原因
- [ ] golden 逻辑与 design 语义一致

---

## 反模式（NEVER DO THESE）

- ❌ golden 逻辑放在 main.cpp 中
- ❌ 精度验证放在 main.cpp 中
- ❌ 只有 smoke case 没有 medium scale
- ❌ 只覆盖单一规模档次（必须覆盖 XS/S/M/L 四档）
- ❌ 总 case 数不足 20
- ❌ golden 用低精度计算导致误差累积
- ❌ 测试脚本依赖未声明的环境变量
- ❌ transport/纯搬运算子使用 relaxed tolerance（应为 bitwise exact）
- ❌ check_result.py 只打印 PASS/FAIL 不输出 MaxAE/MaxRE/MeanRE
