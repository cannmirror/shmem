# 算子实现边界

设计阶段必须明确算子的实现边界，避免下游实现阶段把复杂逻辑放到错误的位置。

## 各层职责

| 位置 | 职责 |
| --- | --- |
| Device kernel | 算子语义中的计算和通信（RMA、collective）、Device 侧数据搬运和格式转换、Phase 间同步 |
| `main.cpp` | ACL/SHMEM 初始化和清理、读写输入输出文件、Host-Device 数据拷贝、Kernel 调用、资源释放；单进程单 PE |
| Host helper `.cpp/.h` | 运行时 shape/layout 推导、tiling/launch 参数构造、输入 packing/输出 unpacking、route/payload 计划 |
| Python scripts | 输入数据生成、测试用 Route/Payload 计算、Golden 计算、精度验证 |

## 禁止的实现

- Host RMA 作为算子主要通信路径（即使是 correctness-first）
- Device kernel 中用 `DataCopy` 直接访问远端 PE 地址完成跨 PE 搬运
- `main.cpp` 中包含复杂逻辑（route、编码、解码、tiling、packing、业务预处理等）
- `main.cpp` 中包含 golden 生成或精度验证
- `main.cpp` fork/spawn 多个子进程分别负责多个 PE
- 以"先实现 Host 版本"为由跳过 Device kernel

## 判断标准

- 只服务测试数据或 oracle 的逻辑 -> Python
- 运行时 Host 计划逻辑 -> 独立 Host `.cpp/.h`
- 算子通信或计算语义 -> Device kernel
- `main.cpp` 应该是胶水代码，只串联阶段
