# rdma_perftest

## 示例概述

`rdma_perftest` 是用于**测试 shmem RDMA 低阶接口性能**的参数化测试示例，平行于同目录下的 `mte_perftest`（针对 MTE 引擎）和 `udma_perftest`（针对 UDMA 引擎）。该示例通过 [SHMEMI_PROF_START/END](../../../src/device/utils/prof/shmemi_prof.h) 宏采集性能数据，覆盖 `aclshmemx_roce_put_nbi` / `aclshmemx_roce_get_nbi` 两个 RDMA 低阶接口在不同数据量下的传输带宽。**该脚本测试结果仅做参考，性能以实际场景为准**。

## 测试目的

针对以下 RDMA 数据传输操作的性能：

1. **单向 Put** (`put`)：仅 `SHMEM_CYCLE_PROF_PE` 指定的 PE 调用 RDMA put 接口，将数据传输到对端 PE。
2. **双向 Put** (`bi_put`)：两个 PE 同时调用 put，互相传输数据。
3. **单向 Get** (`get`)：仅 prof PE 调用 RDMA get 接口，从对端 PE 拉取数据。
4. **双向 Get** (`bi_get`)：两个 PE 同时调用 get，互相拉取数据。

## 与 `mte_perftest`、`udma_perftest` 的差异

| 维度 | `mte_perftest` (MTE) | `udma_perftest` (UDMA) | `rdma_perftest` (RDMA) |
|------|----------------------|----------------------|----------------------|
| 引擎 | 默认 MTE | 显式 `ACLSHMEM_DATA_OP_UDMA` | RDMA 引擎 |
| 多核并发 | 同 peer 多核 (默认 32 核切分数据) | **强制单核** (`block_dim=1`)：UDMA 不允许同 peer 多核并发 | **强制单核** (`block_dim=1`)：RDMA 不允许同 peer 多核并发 |
| `-b/--block-size`、`--block-range` | 控制核数 | 入参兼容，但**强制 1**，输入其他值会打印 WARN 后忽略 | 入参兼容，但**强制 1**，输入其他值会打印 WARN 后忽略 |
| UB 缓冲 | MTE 必需，影响传输 | UDMA 内部不消耗 UB，仅形式上保留 `--ub-size` 入参 | RDMA 必须，大小至少为 64B，默认为 64B |
| 测试模式 | put / bi_put / get / bi_get | put / bi_put / get / bi_get / **put_signal** | put / bi_put / get / bi_get |
| SOC 限制 | 通用 | **仅 Ascend950**：非 950 上 device kernel 内置 abort | **Ascend950（需配合云脉网卡）或 Ascend910B/C** |
| CSV 文件名 | `<test>_<dtype>_<pe>.csv` | `udma_<test>_<dtype>_<pe>.csv` | `rdma_<test>_<dtype>_<pe>.csv` |

## 环境要求

同[rdma_demo](../../rdma_demo/README.md)中的环境要求。

## 编译说明

RDMA 功能需要在编译时启用 `-enable_rdma` 参数，并根据 SOC 类型进行配置：

**Ascend910B/C 平台**（默认配置）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -enable_rdma
```

**Ascend950 平台**（需指定云脉网卡后端）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -enable_rdma -soc_type Ascend950 -rdma_backend XSCALE
```

## 使用方法

### 基本用法

```bash
cd examples/shmem_perftest/rdma_perftest/
./run.sh [选项]
```

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | 测试类型 (put / bi_put / get / bi_get / all) | `put` |
| `--datatype <type>` | `-d <type>` | 数据类型 (float / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64 / char / all) | `float` |
| `--block-size <size>` | `-b <size>` | RDMA 目前强制为 1，其他值打 WARN 后忽略 | 1 |
| `--block-range <min> <max>` | - | 同上 | 1 1 |
| `--exponent <exponent>` | `-e <exponent>` | 数据量幂数 (2^exponent 字节) | - |
| `--exponent-range <min> <max>` | - | 数据量幂数范围 | 3 17 |
| `--loop-count <count>` | - | 循环次数 | 1000 |
| `--ub-size <size>` | - | UB size (KB)；RDMA 需要至少 64B 的 UB 空间 | 64 |
| `--batch <count>` | - | 单 QP 上每次调用 quiet 之前连续提交的 NBI 个数 (0 表示全异步) | 0 |
| `--sync-id <id>` | - | 显式传给 Put、Get、Quiet 的同步 ID | 0 |
| `-q/--qp <num>` | - | QP 的个数，当前版本仅支持单 QP | 1 |
| `-pes <size>` | - | PE 数量 (目前强制为 2) | 2 |
| `-ipport <ip:port>` | - | 通信地址 | tcp://127.0.0.1:8768 |
| `-gnpus <num>` | - | NPU 数量 | 2 |
| `-fnpu <id>` | - | 首个 NPU ID | 0 |
| `-fpe <id>` | - | 首个 PE ID | 0 |
| `-a/--analyse <mode>` | - | 分析模式 (none / plot / md) | none |

### DRAM 内存约束

本示例仅测试 HBM (DEVICE_SIDE) 内存路径，**不支持 D2H / `HOST_SIDE` (DRAM)**。

默认 1 GB 本地内存；当数据量较大时，程序会自动上调 `local_mem_size`（最多 40 GB）。

### 使用示例

```bash
# 单向 PUT，float，幂数 8-20
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000

# 双向 GET，int32
./run.sh -t bi_get -d int32 --exponent-range 8 20 --loop-count 1000

# 四种模式 × float
./run.sh -t all -d float --exponent-range 8 20 --loop-count 1000

# 单一模式 × 全部数据类型
./run.sh -t put -d all --exponent-range 8 20

# 单向 PUT, 所有 NBI 执行过后确认一次
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 0

# 单向 PUT, 每 128 个 NBI 确认一次
./run.sh -t put -d float --exponent-range 8 20 --loop-count 1000 --batch 128
```

## CSV 输出

CSV 列与 MTE 版完全一致，便于复用 `examples/utils/perf_data_process.py` 出图：

```
DataSize/B, Npus, Blocks, UBsize/KB, Bandwidth/GB/s, CoreMaxTime/us, SingleCoreTime/us
```

`Blocks` 列恒为 1。文件名前缀为 `rdma_`：`output/rdma_<test_type>_<dtype>_<pe>.csv`。

## 输出示例

```
[INFO] rdma_perftest start, pe=0, t=put, d=float, exp=10-10, loop=100, ub=16KB
pe: 0 size: 1024 frame_id: 0
[Verification] put: checking...
[Verification] SUCCESS
[SUCCESS] rdma_perftest done in pe 0
```

## 已知约束

1. RDMA 头文件注明：concurrent RMA/AMO operations to the same PE are not supported。本 perftest 通过 `block_dim=1` 规避，多核场景留作后续扩展。
2. RDMA 功能需要在编译时启用 `-enable_rdma` 参数，否则编译期会报错；Ascend950 平台还需额外指定 `-soc_type Ascend950 -rdma_backend XSCALE` 参数。
3. **不支持 D2H / `HOST_SIDE` (DRAM)**: RDMA 引擎当前未对 Host 侧 DRAM 提供 RMA 路径，仅测 HBM。
4. 原子操作不在本 perftest 范围。
