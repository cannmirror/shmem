# SHMEM 性能 Baseline 选择策略

性能采集需要明确的 baseline 作为对比基准。硬规则：

- 有 HCCL/aclnn/拼接 baseline 时，**MUST** 接入 baseline
- 有 baseline 时，默认达标线为 current ≥ baseline 的 80%
- 有 baseline 但未达 80% 时，进入性能优化（最多 5 轮）
- 无直接 baseline 时，**MUST** 记录搜索过程，并使用 metric_only 指标验收；通信算子能计算带宽利用率时 **NEVER** 低于 20%

按以下优先级选择 baseline。

## 1. 优先级 1：HCCL / aclnn 算子库

**适用场景**：算子功能与 HCCL 集合通信算子或 aclnn 融合算子库中的某个算子完全或部分对应。

> **重要**：很多通算融合算子在 aclnn 算子库中有对应实现（如 `aclnnMatmulAllReduce`），**MUST** 优先查找。声称"无 baseline"前 **MUST** 逐一排查以下两张清单。

### 1.1 HCCL 集合通信算子清单

纯通信算子优先在此清单中查找 baseline：

| SHMEM 算子 | HCCL Baseline | HCCL API | 说明 |
| --- | --- | --- | --- |
| AllGather | HcclAllGather | `HcclAllGather(sendBuf, recvBuf, count, dataType, comm, stream)` | 每 PE 数据收集到全部 PE |
| AllReduce | HcclAllReduce | `HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream)` | 全局归约 |
| ReduceScatter | HcclReduceScatter | `HcclReduceScatter(sendBuf, recvBuf, count, dataType, op, comm, stream)` | 归约后分片 |
| AllToAll | HcclAlltoAll | `HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream)` | 全交换 |
| Broadcast | HcclBroadcast | `HcclBroadcast(buf, count, dataType, root, comm, stream)` | 广播 |
| Reduce | HcclReduce | `HcclReduce(sendBuf, recvBuf, count, dataType, op, root, comm, stream)` | 归约到 root |
| Scatter | HcclScatter | `HcclScatter(sendBuf, recvBuf, count, dataType, root, comm, stream)` | root 分发 |
| Gather | HcclGather | `HcclGather(sendBuf, recvBuf, count, dataType, root, comm, stream)` | 收集到 root |

### 1.2 aclnn 融合算子清单

通算融合算子和部分复杂通信算子优先在 aclnn 算子库中查找 baseline：

| SHMEM 算子 | aclnn Baseline | aclnn API | 说明 |
| --- | --- | --- | --- |
| MatMul AllReduce | aclnnMatmulAllReduce | `aclnnMatmulAllReduce(...)` | matmul + allreduce 融合 |
| MatMul ReduceScatter | aclnnMatmulReduceScatter | `aclnnMatmulReduceScatter(...)` | matmul + reduce_scatter 融合 |
| AllGather MatMul | aclnnAllGatherMatmul | `aclnnAllGatherMatmul(...)` | allgather + matmul 融合 |
| MatMul AllToAll | aclnnMatmulAllToAll | `aclnnMatmulAllToAll(...)` | matmul + alltoall 融合 |
| AllToAll（V 版本） | aclnnAlltoAllV | `aclnnAlltoAllV(...)` | 变长全交换 |

> **查找方法**：在 CANN 安装目录下搜索 `aclnn` 开头的头文件：`find $ASCEND_HOME_PATH -name "aclnn*.h" | grep -iE "key1|key2|..."`。关键词必须列出所有常见变体，如 matmul_alltoall 对应 `matmul_alltoall|mat_mul_allto_all|alltoall_matmul` 等多组组合，**NEVER** 只用单一关键词。也可查阅 CANN 在线文档的"aclnn 算子接口"章节。

### 1.3 Baseline 接入方式（C++ 可执行文件）

HCCL 和 aclnn baseline **MUST** 以 C++ 可执行文件方式接入（**NEVER** 使用 Python 脚本），在算子目录的 `baseline/` 子目录下编写独立的 baseline 测试程序和 `CMakeLists.txt`，**NEVER** 将 baseline 源码或编译 target 放在算子 `src/` 或根目录 `CMakeLists.txt` 中。

**HCCL baseline 示例**（以 AllToAll 为例）：

```cpp
#include <hccl/hccl.h>
#include <acl/acl.h>

bool LaunchHcclBaseline(void *sendBuf, void *recvBuf,
                         int64_t count, HcclComm comm, aclrtStream stream) {
    HcclResult ret = HcclAlltoAll(
        sendBuf, static_cast<uint64_t>(count), HCCL_DATA_TYPE_FP16,
        recvBuf, static_cast<uint64_t>(count), HCCL_DATA_TYPE_FP16,
        comm, stream);
    return (ret == HCCL_SUCCESS);
}

aclrtEvent startEvent, endEvent;
aclrtCreateEvent(&startEvent);
aclrtCreateEvent(&endEvent);

for (int i = 0; i < warmup; i++) {
    LaunchHcclBaseline(sendBuf, recvBuf, count, comm, stream);
}
aclrtSynchronizeStream(stream);

aclrtRecordEvent(startEvent, stream);
for (int i = 0; i < iters; i++) {
    LaunchHcclBaseline(sendBuf, recvBuf, count, comm, stream);
}
aclrtRecordEvent(endEvent, stream);
aclrtSynchronizeStream(stream);

float elapsed_ms;
aclrtEventElapsedTime(&elapsed_ms, startEvent, endEvent);
float avg_us = elapsed_ms * 1000.0f / iters;
printf("HCCL_PERF pe=%d avg_us=%.2f algo_GBps=%.2f\n", my_pe, avg_us, ...);
```

**CMake 集成**（baseline 的 CMakeLists.txt 和源码 **MUST** 放在 `baseline/` 目录下，**NEVER** 混入算子 `src/`）：

```cmake
add_executable(${OP_NAME}_hccl_baseline baseline/src/${OP_NAME}_hccl.cpp)
target_link_libraries(${OP_NAME}_hccl_baseline PRIVATE hccl ascendcl)
```

**aclnn baseline 示例**（以 MatmulAllToAll 为例）：

```cpp
#include <aclnn/aclnn_matmul_alltoall.h>
#include <acl/acl.h>

uint64_t workspaceSize = 0;
aclnnMatmulAllToAllGetWorkspaceSize(inputDesc, weightDesc, outputDesc,
                                     hcomm, &workspaceSize, &executor);

void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclnnMatmulAllToAll(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

**记录格式**：

```yaml
baseline:
  type: "cann_operator"
  source: "HCCL"
  api: "HcclAlltoAll"
  latency_us: 125.3
  command: "./alltoall_hccl --n_pes 8 --count 1048576 --dtype fp16 --iters 100"
  environment:
    cann_version: "8.0.RC1"
    device: "Atlas 800T A2"
    n_pes: 8
```

## 2. 优先级 2：小算子拼接（C++ 实现）

**适用场景**：HCCL/aclnn 算子库中没有完全对应的融合算子，但可以用多个小算子串行拼接实现相同功能。

> **核心原则**：拼接 baseline **MUST** 用 C++ 实现（HCCL API + CATLASS/AscendC 计算），**NEVER** 使用 Python 脚本。拼接的目的是测量"不融合时的串行执行时间"，作为融合算子的性能上界参考。

### 2.1 常见拼接方案

| 融合算子 | 拼接方案 | 组件 1 | 组件 2 | 组件 3 |
| --- | --- | --- | --- | --- |
| AllToAll MatMul | HCCL AllToAll + CATLASS MatMul | `HcclAlltoAll` | `catlass_matmul` | — |
| MatMul AllReduce | CATLASS MatMul + HCCL AllReduce | `catlass_matmul` | `HcclAllReduce` | — |
| AllGather MatMul | HCCL AllGather + CATLASS MatMul | `HcclAllGather` | `catlass_matmul` | — |
| MatMul ReduceScatter | CATLASS MatMul + HCCL ReduceScatter | `catlass_matmul` | `HcclReduceScatter` | — |
| Dispatch GMM Combine | HCCL AllToAll + GMM + HCCL AllToAll | `HcclAlltoAll` | `catlass_grouped_matmul` | `HcclAlltoAll` |
| KV Shuffle | Put + Get + Barrier | `aclshmemx_mte_put_nbi` | `aclshmemx_mte_get_nbi` | `aclshmem_barrier_all` |

### 2.2 拼接 baseline C++ 示例（AllToAll + MatMul）

```cpp
struct StitchedResult {
    float alltoall_us;
    float matmul_us;
    float total_us;
};

StitchedResult RunStitchedBaseline(
    void *sendBuf, void *recvBuf, int64_t alltoallCount,
    void *matA, void *matB, void *matC, int M, int N, int K,
    HcclComm comm, aclrtStream stream, int warmup, int iters)
{
    aclrtEvent e0, e1, e2;
    aclrtCreateEvent(&e0);
    aclrtCreateEvent(&e1);
    aclrtCreateEvent(&e2);

    for (int i = 0; i < warmup; i++) {
        HcclAlltoAll(sendBuf, alltoallCount, HCCL_DATA_TYPE_FP16,
                     recvBuf, alltoallCount, HCCL_DATA_TYPE_FP16,
                     comm, stream);
    }
    aclrtSynchronizeStream(stream);

    float alltoall_ms = 0, matmul_ms = 0;

    aclrtRecordEvent(e0, stream);
    for (int i = 0; i < iters; i++) {
        HcclAlltoAll(sendBuf, alltoallCount, HCCL_DATA_TYPE_FP16,
                     recvBuf, alltoallCount, HCCL_DATA_TYPE_FP16,
                     comm, stream);
    }
    aclrtRecordEvent(e1, stream);
    aclrtSynchronizeStream(stream);
    aclrtEventElapsedTime(&alltoall_ms, e0, e1);

    aclrtRecordEvent(e1, stream);
    for (int i = 0; i < iters; i++) {
        // catlass_matmul(matA, matB, matC, M, N, K, stream);
    }
    aclrtRecordEvent(e2, stream);
    aclrtSynchronizeStream(stream);
    aclrtEventElapsedTime(&matmul_ms, e1, e2);

    StitchedResult r;
    r.alltoall_us = alltoall_ms * 1000.0f / iters;
    r.matmul_us   = matmul_ms * 1000.0f / iters;
    r.total_us    = r.alltoall_us + r.matmul_us;
    return r;
}
```

**记录格式**：

```yaml
baseline:
  type: "stitched"
  description: "HCCL AllToAll + CATLASS MatMul 串行拼接"
  components:
    - name: "AllToAll"
      api: "HcclAlltoAll"
      latency_us: 125.3
    - name: "MatMul"
      api: "catlass_matmul"
      latency_us: 85.2
  measurement:
    total_latency_us: 210.5
    algo_bandwidth_GBps: 19.0
  command: "./alltoall_matmul_stitched --n_pes 8 --count 1048576"
  notes: "串行拼接，未考虑 overlap"
```

## 3. 优先级 3：指标测试（无直接 baseline）

**适用场景**：既找不到对应的 CANN 算子，也无法用小算子拼接，或者是全新的算子设计。

使用 metric_only 前 **MUST** 说明已逐一检查过以下来源且均无对应实现：HCCL 清单（§1.1）、aclnn 清单（§1.2）、已有 SHMEM example、用户参考实现、可拼接方案（§2）。metric_only 不是跳过性能采集或指标验收的理由。

**测试策略**：

1. **时延测试**：测量端到端时延，分析是否存在明显的性能瓶颈，与理论时延对比（如果可计算）
2. **带宽测试**：计算算法带宽和总线带宽，与硬件峰值带宽对比；对通信算子，能计算带宽利用率时 **MUST** 输出 utilization，且 **NEVER** 低于 20%
3. **计算能力使用率**：对于通算融合算子，测量计算单元使用率，计算理论 FLOPS 与实际 FLOPS 的比值

**判断标准**：

- 时延明显过高（简单搬运操作耗时超过设计或理论阈值）
- 通信算子带宽利用率低于 20%
- 通算融合算子的通信阶段带宽利用率低于 20%，或 compute/wait/sync 占比显示明显瓶颈
- 存在明显的空闲等待时间

**示例**：

```python
logical_payload_bytes = n_pes * tokens_per_pe * hidden * 4
latency_us = measure_latency()
latency_s = latency_us / 1e6
algo_bandwidth = logical_payload_bytes / latency_s / 1e9

bus_factor = (n_pes - 1) / n_pes
bus_bandwidth = algo_bandwidth * bus_factor

peak_bandwidth = 28
utilization = bus_bandwidth / peak_bandwidth * 100
```

**记录格式**：

```yaml
baseline:
  type: "metric_only"
  current_latency_us: 245.8
  logical_payload_bytes: 4.0 MB
  algo_bandwidth_GBps: 16.3
  bus_bandwidth_GBps: 8.15
  peak_bandwidth_GBps: 300
  bandwidth_utilization_percent: 2.7
  bottleneck_analysis: 带宽利用率极低
```

## 4. Baseline 选择决策树

```
Step 1: 查找 HCCL 集合通信算子清单（§1.1）
        ├─ 有完全对应 → 使用 HCCL 算子作为 baseline
        └─ 无 → Step 2

Step 2: 查找 aclnn 融合算子清单（§1.2）
        ├─ 有对应融合算子 → 使用 aclnn 算子作为 baseline
        └─ 无 → Step 3

Step 3: 能否用 HCCL + CATLASS/AscendC 拼接？（§2）
        ├─ 能拼接 → 使用拼接方案作为 baseline
        └─ 无法拼接 → Step 4

Step 4: 使用指标测试（§3，metric_only）
        ├─ 测量时延和带宽利用率
        └─ 带宽利用率 ≥ 20%
```

**关键提醒**：

- 声称"无 baseline"前 **MUST** 走完 Step 1-3，并在报告中记录每步的搜索结果
- 很多通算融合算子在 aclnn 中有对应实现，不要遗漏 Step 2
- HCCL 库包含 AllToAll 等算子，不要误认为 HCCL 只有 AllGather/AllReduce/ReduceScatter

## 5. Baseline 记录要求

无论使用哪种 baseline，都 **MUST** 记录：

```yaml
baseline:
  type: "hccl" | "aclnn" | "stitched" | "metric_only"
  source: "HCCL" | "aclnn" | "HCCL+CATLASS" | "metric_only"
  api: "HcclAlltoAll"
  components:
    - name: "MatMul"
      latency: 85.2
      source: "CATLASS"
    - name: "AllReduce"
      latency: 125.3
      source: "HCCL"
  measurement:
    latency_us: 210.5
    algo_bandwidth_GBps: 19.0
    bus_bandwidth_GBps: 28.5
    peak_bandwidth_GBps: 28.0
    bandwidth_utilization_percent: 9.5
    effective_flops: 123000000000000.0
    compute_utilization_percent: 42.0
  command: "./baseline_test --n_pes 8 --count 1048576"
  environment:
    cann_version: "8.0.RC1"
    device: "Atlas 800T A2"
    soc: "Ascend910B3"
    n_pes: 8
  search_process: "已检查 HCCL 清单、aclnn 清单、拼接方案，选定 ..."
```