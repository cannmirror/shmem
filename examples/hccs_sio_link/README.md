# HCCS SIO Link

HCCS/SIO 链路测试工具，用于验证 NPU 之间 SIO / HCCS 链路的正确性。

> **注意**：当前 CANN 版本暂未支持，预计 CANN 9.1.0 版本支持该功能。


## 链路说明

对于 A3 芯片，一个 NPU 包含两个 Die，两 Die 之间通过 SIO 链路互连。本方案在 SIO 链路基础上新增 HCCS 链路，实现 SIO/HCCS 双路并行传输，从而提高 Die 间数据传输速度。

- **SIO**：Die 间原有互连链路。shmem 初始化后默认获得的 VA 访问对端 Die 时即走 SIO 链路。

- **HCCS**：新增的 Die 间互连链路。通过 [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/03_api_ref/11-04_虚拟内存管理.md#aclrtmemmapselectedlink) 将新 VA 映射到与原 SIO VA 相同的 PA，并使用 `ACL_RT_MEM_LINK_IDX_1` 选中 HCCS 链路，此时通过新 VA 访问即走 HCCS 链路。

- **SIO + HCCS 双路并行**：同时使用 SIO 和 HCCS 两条链路传输数据，充分利用双链路带宽提升传输性能。

## 编译

本示例需要通过 `-examples` 编译选项启用，CMake 会自动检测当前 CANN 版本是否支持 `aclrtMemMapSelectedLink` 函数，若支持则自动编译本示例：

```bash
bash scripts/build.sh -examples
```

编译产物：`build/bin/hccs_sio_link`

## 前置条件

- 已按上述方式编译 shmem 项目
- 已设置 `ASCEND_HOME_PATH` 环境变量

> **多实例说明**：`aclshmemx_get_heap_base` 返回当前活跃实例的堆基址。在多实例场景下，需先通过 `aclshmemx_instance_ctx_set_impl` 切换到目标实例，再调用 `aclshmemx_get_heap_base`。

## 用法

本工具通过 `run.sh` 脚本启动，脚本会为每个 PE 启动一个后台进程，各 PE 之间通过 shmem 初始化建立通信。

### 运行命令

```bash
bash run.sh [选项]
```

### 典型用例

```bash
# 默认配置：2 个 PE，SIO+HCCS 全链路测试，4KB 数据，int 类型
bash run.sh

# 指定 4 个 PE，仅测 HCCS 链路
bash run.sh -pes 4 -mode hccs

# 指定 8 个 PE，8MB 数据，fp32 类型，仅测 SIO 链路
bash run.sh -pes 8 -size 8 -type fp32 -mode sio

# SIO + HCCS 混合测试（3/5 数据走 SIO，2/5 数据走 HCCS）
bash run.sh -mode mixed
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-ipport` | `tcp://127.0.0.1:8766` | 通信初始化地址 |
| `-pes` | `2` | 参与测试的 PE 总数（与 NPU 数量一致） |
| `-fpe` | `0` | 首个 PE 编号 |
| `-fnpu` | `0` | 首个 NPU 编号 |
| `-type` | `int` | 测试数据类型：`int` / `int64` / `fp32` |
| `-mode` | `all` | 测试模式（见下表） |
| `-size` | `4` | 每个 PE 的数据大小（KB） |

### 测试模式

| 模式 | 说明 |
|------|------|
| `sio` | SIO 链路正确性测试 |
| `hccs` | HCCS 链路正确性测试 |
| `all` | SIO + HCCS 全链路正确性测试 |
| `mixed` | SIO + HCCS 混合正确性测试（3/5 数据走 SIO，2/5 数据走 HCCS） |

## 输出示例

正确性测试：

```
PE 0: [SIO] path verification PASSED for PE 1
PE 1: [SIO] path verification PASSED for PE 0
PE 0: [HCCS] path verification PASSED for PE 1
PE 1: [HCCS] path verification PASSED for PE 0
```

混合测试：

```
PE 0: [MIXED-SIO] path verification PASSED for PE 1
PE 0: [MIXED-HCCS] path verification PASSED for PE 1
PE 1: [MIXED-SIO] path verification PASSED for PE 0
PE 1: [MIXED-HCCS] path verification PASSED for PE 0
```
