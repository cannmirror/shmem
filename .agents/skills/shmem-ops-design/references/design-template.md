# [算子名称] 设计文档

## 1. 算子接口与 Assumptions

### 1.1 用户确认项

| 确认项 | 内容 | 确认状态 |
| --- | --- | --- |
| 算子名称 `op_name` | [用户确认的算子名称] | confirmed |
| 算子数据类型 `dtype/dtypes` | [逐项列出输入、输出、累加/中间 dtype] | confirmed |

### 1.2 功能描述

[一句话描述算子功能，明确 local compute、communication、final result。]

### 1.3 Assumptions 与 Open Questions

| 类型 | 内容 | 是否阻塞代码生成 |
| --- | --- | --- |
| assumption | [可接受假设] | no |
| open question | [必须用户确认的问题] | yes/no |

## 2. Canonical DSL

DSL 是本设计文档的唯一结构化信息源。后续代码生成、走读和性能采集均按 DSL 字段执行。必须保留完整 fenced `yaml` block，供 `shmem-ops-code-gen` 直接消费。

```yaml
dsl_version: "0.3"

meta:
  op_name: "replace_me"
  op_kind: "fused_compute_comm"   # transport | collective | compute | fused_compute_comm
  target_soc: ["ascend910b"]
  scope: "intra_node"             # intra_node | inter_node | both

source:
  kind: "natural_language"        # natural_language | user_pseudocode | heterogeneous_reference | mixed
  summary: "replace_me"
  user_confirmations:
    op_name: "confirmed_by_user"
    dtypes: []
  references: []

interface:
  inputs: []
  # - name: "input"
  #   dtype: "fp16"
  #   shape: "[n_pes, shard_elems]"
  #   role: "local_input"
  outputs: []
  # - name: "output"
  #   dtype: "fp16"
  #   shape: "[total_elems]"
  #   visibility: "replicated"    # replicated | sharded | owner_only
  attrs: []
  # - name: "block_dim"
  #   type: "int"
  #   default: "2 * n_pes"
  shape_relations: []

semantics:
  local_compute: "replace_me"
  communication: "replace_me"
  finalize: "replace_me"
  result: "replace_me"

topology:
  group: "replace_me"
  peer_model: "replace_me"
  addressing: "symmetric_memory"

memory:
  buffers: []
  # - name: "staging"
  #   type: "symmetric"
  #   size: "shard_elems * sizeof(dtype)"
  symmetric_layout: "replace_me"
  output_ownership: "replace_me"

schedule:
  # 设计顺序：先分核 → 再 tiling → 最后 phases
  core_partition:
    mode: "replace_me"              # 引用 core-allocation.md 模式：A（全员协作）/ B（按PE分组）/ C1（生产者消费者）/ C2（维度分组）/ CoC（通算融合）
    groups:
      []
      # - name: "send"
      #   aiv_range: "[0, aiv_num/2)"
      #   role: "本地 input 搬运到 GVA"
      #   data_interface: "aclshmemx_mte_put_nbi"
      # - name: "recv"
      #   aiv_range: "[aiv_num/2, aiv_num)"
      #   role: "从远端 GVA 拉取数据到 output"
      #   data_interface: "aclshmemx_mte_get_nbi"
    overlap: "replace_me"           # 组间如何重叠：并行 / 2-stage pipeline / 无
    sync_between_groups: "replace_me"  # 组间同步方式：cross-core flag / barrier / signal-wait / quiet
    rationale: "replace_me"         # 分核依据，引用 core-allocation.md 或 shmem/examples 路径
    cross_pe_dataplane: "replace_me"  # 跨 PE 数据面接口，必须是 aclshmem_*/aclshmemx_*
  tiling:
    tile: "replace_me"
    chunk: "replace_me"
    tail: "replace_me"
  phases:
    []
    # - name: "staging"
    #   action: "所有 AIV 将本地 input 拷贝到 symmetric memory"
    # - name: "remote_fetch"
    #   action: "Recv 组从远端 GVA 拉取数据到 output"
    # - name: "finalize"
    #   action: "回写最终结果到输出 buffer"

correctness:
  oracle: "replace_me"
  tolerance:
    rtol: "replace_me"
    atol: "replace_me"
  invariants:
    []
    # - invariant: "symmetric allocation 顺序和大小一致"
    #   test_method: "rank pattern"
    #   case: "all cases"
  case_matrix: []
  # 功能 case 和 stress/边界 case 统一放在此处：
  # - case: "boundary_tail"
  #   shape: "不整除 chunk 的 shape"
  #   category: "stress"
  # - case: "medium_8pe"
  #   shape: "[8, 32M]"
  #   category: "functional"

performance:
  metric: ["e2e_latency_us", "kernel_latency_us", "algo_bandwidth_GBps", "bus_bandwidth_GBps"]
  baseline: "replace_me"
  baseline_search: "replace_me"       # 已检查哪些 HCCL/CANN/拼接方案，为何不适用（无 baseline 时必填）
  baseline_target: "replace_me"       # 有 baseline: "current >= 80% baseline"；无 baseline: metric_only 指标目标
  target_cases: []
  min_scale: "replace_me"             # 集合通信 >= 256MB；计算/通算融合 hidden size >= 1000
  profiling_method: "replace_me"
  max_opt_rounds: 5
```

## 3. Capability Mapping & Gap Analysis

| Requirement | SHMEM capability/source | Classification | Adaptation / Gap | Risk |
| --- | --- | --- | --- | --- |
| [需求点] | [API/example/template/source path] | [可复用/需适配/需新实现] | [可复用：无；需适配：适配点；需新实现：gap 原因 + 设计方案 + 验证方式] | [正确性/性能风险] |

`Classification` 只能是 `可复用`、`需适配`、`需新实现`。

映射至少覆盖六类：lifecycle、memory、transport、sync、compute、scheduler。

如果没有"需新实现"项，在表后写明 `No known gap`。

## 4. 实现边界

| 位置 | 职责 |
| --- | --- |
| Device kernel | [通信、计算、搬运、同步等算子核心语义] |
| `main.cpp` | [单 PE Host 编排：参数、初始化、资源、读写、launch、cleanup] |
| Host helper `.cpp/.h` | [可选：运行时 shape/layout/tiling/route/payload/packing 计划] |
| Python scripts | [输入/golden/checker、测试用 route/payload、误差报告] |

约束：`main.cpp` 必须单进程单 PE，不 fork/spawn 多 PE；复杂 Host 逻辑必须拆到独立 `.cpp/.h`；golden 和 checker 不进入 Host C++。

## 5. Design Review Before Handoff

| Check | Result | Fix or rationale |
| --- | --- | --- |
| 并发/分核思路是否在首版 correctness 交付中落地 | [pass/fail] | [说明] |
| 是否存在不必要的 Host/global barrier | [pass/fail] | [说明] |
| tile/chunk/tail 是否支持并发且不靠单核兜底 | [pass/fail] | [说明] |
| route/offset/prefix 计算是否避免明显 O(N²) 扫描 | [pass/fail] | [说明] |
| 是否避免多余 GM scratch 写读，或说明无法避免原因 | [pass/fail] | [说明] |
| 高性能传输路径、并发分核或 overlap 是否属于本轮实现范围 | [pass/fail] | [说明] |

## 6. Compile/Test Contract

### 6.1 Compile Contract

| 字段 | 内容 |
| --- | --- |
| build_mode | `independent_project` 或 `in_tree_example` |
| cann_env | [用户指定的 CANN set_env.sh 路径；不得硬编码默认路径] |
| shmem_repo | [路径或发现规则] |
| op_dir | [路径] |
| target | [binary/shared library] |
| command | [完整命令或发现规则] |
| required_flags | [-examples/-debug/-enable_rdma/-enable_ascendc_dump/-soc_type ...] |
| expected_outputs | [build/bin 或 build/lib 产物] |

### 6.2 Functional/Correctness Contract

| 字段 | 内容 |
| --- | --- |
| command | [run.sh/checker 命令] |
| python_env | [python_cmd 或 conda/venv 激活命令；用于 gen_data.py、golden、check_result.py、baseline/profiler] |
| python_deps | [numpy/torch/torch_npu 或脚本实际 import 的依赖] |
| determinism | [seed、repeats、输出路径隔离] |

Oracle、tolerance、case matrix（含 stress/边界 case）已在 DSL `correctness` 中定义，此处不再重复。

## 7. Handoff Checklist for shmem-ops-code-gen

- [ ] Canonical DSL 完整，且无阻塞 open question。
- [ ] 用户已确认 `op_name` 和 dtype/dtypes，且记录在 `source.user_confirmations`。
- [ ] Capability Mapping 覆盖 lifecycle、memory、transport、sync、compute、scheduler；"需新实现"项有设计方案和验证方式。
- [ ] DSL `schedule` 的 core_partition、tiling、phases 可实现。
- [ ] DSL `correctness.invariants` 每项有 test_method 和 case；`case_matrix` 含 stress/边界 case。
- [ ] DSL `performance` 的 baseline_search、baseline_target、min_scale 已填写。
- [ ] Design review（Section 5）已完成，未留下明显单核交付、O(N²) offset、多余 scratch 流量或无理由全局 barrier。
- [ ] 实现边界清楚：`main.cpp` 只做单 PE 编排，复杂 Host 逻辑已归入独立 `.cpp/.h`，golden/checker 归入 Python。
- [ ] compile/test contract 可执行，test contract 写明 Python 环境或依赖。
