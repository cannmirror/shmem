# SHMEM 算子 README 生成规范

本文总结官方 SHMEM examples 的 README 写法，用于约束 `shmem-ops-code-gen` 生成算子时同步产出用户可运行的 `README.md`。注意：这是生成算子目录下的交付文档，不是 skill 自身的 README。

## 1. 生成要求

每个生成算子目录必须包含 `README.md`，并且 README 中的命令要能和实际 `CMakeLists.txt`、`scripts/run.sh`、`scripts/gen_data.py`、`scripts/check_result.py` 对上。禁止只写“按需修改参数”而不给可执行示例。

README 不写未测量的性能结论。性能数据只有在真实采集后才能写入；没有采集时写明采集入口和当前状态。

## 2. 推荐结构

生成 README 默认使用以下结构，若目标 example 已有更强的本地约定，可以在不丢字段的前提下调整顺序：

```markdown
# {op_name} 使用说明

## 算子介绍
说明算子的输入输出语义、通信模式、是否包含 local compute/finalize，以及输出 visibility。

## 环境要求
列出 CANN/SHMEM/Python 依赖、硬件范围、支持 PE 数、dtype、shape、必要环境变量、`PYTHON_CMD` 或 Python 环境激活命令，以及 `numpy`、`torch`、`torch_npu` 等脚本实际需要的包。

## 目录结构
列出 CMake、`src/`、`include/`、`scripts/`、`baseline/`（如有）、`input/golden/output/log/result` 等目录用途。

## 参数说明
用表格列出 `PE_SIZE`、`FIRST_NPU`、`IP_PORT`、shape、dtype、engine、repeats、perf 等参数。

## 编译项目
给出从工程根目录或 SHMEM 根目录执行的完整编译命令。

## 生成数据
给出 `scripts/gen_data.py` 的完整命令，说明 input、golden 和 config 文件位置。命令应使用 README 中声明的 `PYTHON_CMD`，不要隐式假设系统 `python` 可用。

## 运行算子
给出 `scripts/run.sh` 的完整命令。必须说明 run.sh 启动多个独立进程，每个进程只对应一个 PE；`main.cpp` 本身不 fork/spawn 多 PE。

## 验证结果
给出 `scripts/check_result.py` 的完整命令，说明 tolerance、输出文件命名和失败日志。命令应与 `scripts/run.sh` 使用同一个 Python 环境。

## 性能采集
给出 perf/repeats/profiler 入口、结果文件路径和 baseline 说明；没有实测时不要填写数值。

## 注意事项
列出 symmetric allocation 顺序、NBI 完成路径、signal/epoch、输出目录清理、多 PE 输出隔离等约束。
```

## 3. 官方 example 模式

官方 SHMEM examples 中常见三类 README：

- 简短运行型：如 allgather，只说明编译、运行脚本、PE 数和 dtype。
- 参数说明型：如 dispatch_gmm_combine，列出 config 参数、数据生成、运行脚本和单独校验命令。
- 接口说明型：如 sdma、kv_shuffle，除运行方式外，还说明 C++/Torch/API 参数、数据布局、注意事项和数据流转。

生成算子优先采用“参数说明型 + 正确性/性能入口”的结构；如果算子对外暴露 C++/Torch 接口，再补充接口说明型内容。

## 4. README 审查清单

- [ ] 存在 `README.md`，且文件名大小写固定。
- [ ] 算子语义、输入输出、dtype/shape/PE 限制写清楚。
- [ ] CANN、SHMEM 和 Python 环境要求写清楚，包含 `PYTHON_CMD` 或环境激活方式。
- [ ] 编译、生成数据、运行、校验、性能采集命令完整可执行。
- [ ] 参数默认值、取值范围、示例值和输出路径明确。
- [ ] `run.sh` 多 PE 启动方式说明为多个独立进程。
- [ ] 明确 `main.cpp` 单进程单 PE，不负责启动多个 PE。
- [ ] 正确性阈值、golden 来源、输出文件命名规则明确。
- [ ] 性能结果没有编造；未采集时只保留采集入口和待采集说明。
