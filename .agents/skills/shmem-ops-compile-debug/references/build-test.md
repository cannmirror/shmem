# SHMEM 构建、运行与测试

本文说明 SHMEM 算子两类构建方式：独立工程依赖已编译 SHMEM 库，以及算子作为 `shmem/examples` 内置 example 编译。`shmem-ops-code-gen` 只负责生成代码和测试入口；实际构建、运行和失败定位由 `shmem-ops-compile-debug` 执行。

## 1. 通用环境

构建和运行前，必须确认以下环境已就绪：

**CANN 环境**：`ASCEND_HOME_PATH` 已设置，`bisheng` 编译器可用。缺失时询问用户提供 `set_env.sh` 路径，不直接停止。

**SHMEM 核心库**：`install/shmem/lib/libshmem.so` 和 bootstrap 插件存在。不存在时先执行 `bash scripts/build.sh` 构建核心库，再 `source install/set_env.sh`。

**Python 环境**：`PYTHON_CMD` 或 `python3` 可用，且能导入测试脚本需要的依赖（`numpy`，按脚本再检查 `torch` 等）。缺失时询问用户提供可执行文件或 conda/venv 激活命令。

**构建产物位置**：`build/lib`、`build/bin`（编译产物），`install/shmem/include` 和 `install/shmem/lib`（安装产物）。

常用构建参数：

| 参数 | 用途 |
| --- | --- |
| `-examples` | 编译 `examples/` 下的样例 |
| `-debug` | Debug 构建，定义 `DEBUG_MODE` |
| `-enable_rdma` | 开启 RDMA 相关 target 和编译宏 |
| `-enable_simt` | 开启 SIMT example |
| `-enable_ascendc_dump` | 开启 AscendC Dump/printf 调测宏 |
| `-mssanitizer` | example 编译 sanitizer 支持 |
| `-soc_type Ascend950` | Ascend950 类 SOC 使用 |
| `-uttests` | 编译单元测试 |
| `-python_example` | 编译部分 Python/Torch 扩展示例 |

注意：`scripts/build.sh` 每次会清理并重建 `build/` 和 `install/`。调试单 target 时优先使用已有 `build/` 做增量构建（`cmake --build build --target <target> -j`）。

## 2. 模式 A：独立工程依赖已编译 SHMEM 库

### 2.1 适用场景

使用该模式时，算子工程位于 shmem 仓库外，不修改 `shmem/examples`。工程只依赖：

- `${SHMEM_HOME_PATH}/shmem/include`
- `${SHMEM_HOME_PATH}/shmem/lib/libshmem.so`
- `${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so`
- bootstrap 插件：`aclshmem_bootstrap_config_store.so`、`aclshmem_bootstrap_uid.so`，MPI 场景还需要 `aclshmem_bootstrap_mpi.so`

bootstrap 插件应与 `libshmem.so` 位于同一目录；不要只拷贝 `libshmem.so`。

本地仓库中的 `examples/init/CMakeLists.txt` 虽位于 examples 目录，但它展示了“链接 `install/shmem/lib` 中已编译 SHMEM 库”的写法，可作为独立工程 CMake 参考。

### 2.2 构建前置

```bash
export CANN_ENV=/path/to/user/specified/set_env.sh
source "${CANN_ENV}"
cd /path/to/shmem
bash scripts/build.sh
source install/set_env.sh
```

确认：

```bash
test -f "$SHMEM_HOME_PATH/shmem/include/shmem.h"
test -f "$SHMEM_HOME_PATH/shmem/lib/libshmem.so"
test -f "$SHMEM_HOME_PATH/shmem/lib/aclshmem_bootstrap_config_store.so"
```

### 2.3 独立工程 CMake 要点

独立工程需要自己提供 CANN、AscendC、SHMEM include 和 link 配置。最小骨架：

```cmake
cmake_minimum_required(VERSION 3.18)
project(shmem_op LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER bisheng)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set; ask the user for CANN set_env.sh and source it before build")
endif()
if(NOT DEFINED ENV{SHMEM_HOME_PATH})
  message(FATAL_ERROR "SHMEM_HOME_PATH is not set; source shmem/install/set_env.sh first")
endif()

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})

include_directories(
  ${ASCEND_HOME_PATH}/compiler/tikcpp
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/experiment/runtime
  ${ASCEND_HOME_PATH}/include/experiment/msprof
  ${SHMEM_HOME_PATH}/shmem/include
)

link_directories(
  ${ASCEND_HOME_PATH}/lib64
  ${SHMEM_HOME_PATH}/shmem/lib
)

link_libraries(runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread)

add_executable(my_shmem_op
  src/main.cpp
  src/my_shmem_op_kernel.cpp
  src/op_host_plan.cpp        # 可选：复杂 Host 计划逻辑
  src/op_io.cpp               # 可选：文件读写 helper
  src/op_runtime.cpp          # 可选：资源和错误处理 helper
)
target_include_directories(my_shmem_op PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(my_shmem_op PRIVATE -O2 -std=c++17 -xc++)
target_link_options(my_shmem_op PRIVATE --cce-fatobj-link)
target_link_libraries(my_shmem_op PRIVATE shmem)
```

若算子采用简单官方 example 根目录布局，也可以把 `main.cpp` 和 kernel 文件放在根目录；但生成算子若包含复杂 Host 计划逻辑，优先使用 `src/` 目录并在 target 中显式列出 helper 源文件，头文件也放在 `src/` 中。

如果包含 Device kernel，必须补齐与目标 SOC 匹配的 CCE/AscendC 编译选项。可从本地 `shmem/CMakeLists.txt` 的 `CMAKE_CCE_COMPILE_OPTIONS` 和相近 example 复制，不要凭空写 arch。

### 2.4 独立工程构建命令

```bash
export CANN_ENV=/path/to/user/specified/set_env.sh
source "${CANN_ENV}"
export PYTHON_CMD=${PYTHON_CMD:-python3}
${PYTHON_CMD} --version
source /path/to/shmem/install/set_env.sh
cd /path/to/my_shmem_op
cmake -S . -B build
cmake --build build --target my_shmem_op -j
```

运行前确认：

```bash
ldd build/bin/my_shmem_op | grep -E "libshmem.so|not found"
```

若部署到自定义目录，把 `libshmem.so`、`libshmem_utils.so` 和 bootstrap 插件一起放入同一 `lib/`，并让 `LD_LIBRARY_PATH` 优先指向该目录。

## 3. 模式 B：算子作为 shmem/examples 内置 example

### 3.1 适用场景

使用该模式时，算子目录位于：

```text
shmem/examples/<op_name>/
```

适合复用 `examples/utils`、`examples/templates/include`、catcoc 模板、现有 example 分核/CMake 函数，以及需要随 SHMEM 源码一起编译的算子。

### 3.2 目录结构

纯通信或 collective 常见结构：

```text
examples/<op_name>/
  CMakeLists.txt
  README.md
  docs/
    design.md
  src/
    main.cpp
    <op_name>_kernel.cpp
    <op_name>_kernel.h
    op_host_plan.cpp
    op_host_plan.h
  scripts/
    run.sh
    gen_data.py
    check_result.py
```

通算融合或单文件 demo 可按相近 example 放置：

```text
examples/<op_name>/
  CMakeLists.txt
  README.md
  src/
    main.cpp
    op_host_plan.cpp
    op_host_plan.h
  scripts/
    run.sh
    test_shapes.csv
```

### 3.3 CMake 接入

在 `examples/<op_name>/CMakeLists.txt` 中优先复用仓库函数：

```cmake
aclshmem_add_collective_example(<op_name>)
```

该函数假定存在官方约定的 `main.cpp` 和 `<op_name>_kernel.cpp`，会生成：

- `build/bin/<op_name>`
- `build/lib/lib<op_name>_kernel.so`

通算融合或单文件 demo 使用：

```cmake
aclshmem_add_fusion_example(<op_name> main.cpp)
```

复杂工程可参考 `examples/dynamic_tiling/CMakeLists.txt` 自定义 shared library、tiling library 和依赖关系。

如果生成算子采用 `src/` 目录和多个 Host helper `.cpp/.h` 的模块化布局，不要强行套用只识别根目录文件的 helper 函数；应在 example-local `CMakeLists.txt` 中显式列出 `src/main.cpp`、kernel 源文件和 Host helper 源文件，并用 `target_include_directories(... PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)` 补齐头文件搜索路径。若必须复用官方 helper，则提供薄包装文件或调整目录使其符合 helper 的实际假设。

然后把新目录加入 `examples/CMakeLists.txt` 对应列表：

- 普通 example：加入主 `foreach(EXAMPLE ...)`。
- RDMA example：加入 `if(ACLSHMEM_RDMA_SUPPORT)` 内列表，并使用 `bash scripts/build.sh -examples -enable_rdma`。
- SIMT example：加入 `if(ACLSHMEM_SIMT_SUPPORT)` 内列表。

### 3.4 编译命令

首次完整配置并编译 examples：

```bash
export CANN_ENV=/path/to/user/specified/set_env.sh
source "${CANN_ENV}"
cd /path/to/shmem
bash scripts/build.sh -examples
source install/set_env.sh
```

常用变体：

```bash
bash scripts/build.sh -examples -debug
bash scripts/build.sh -examples -enable_ascendc_dump
bash scripts/build.sh -examples -enable_rdma
bash scripts/build.sh -examples -soc_type Ascend950
```

首次配置完成后，单 target 调试可用：

```bash
cmake --build build --target <op_name> -j
cmake --build build --target <op_name>_kernel -j
```

注意：再次执行 `scripts/build.sh` 会清空并重建 `build/`，增量调试时不要反复用它。

### 3.5 运行命令

优先使用 example 自带脚本：

```bash
cd /path/to/shmem/examples/<op_name>
bash scripts/run.sh 0,1
```

或按纯通信 demo 风格：

```bash
cd /path/to/shmem/examples/<op_name>
bash run.sh -pes 2 -fnpu 0 -gnpus 2 -ipport tcp://127.0.0.1:8766
```

run 脚本应做：

- `source ${PROJECT_ROOT}/install/set_env.sh`
- 设置 `SHMEM_UID_SESSION_ID=<host>:<port>`。
- 每个 PE 启动一个进程，参数包含 `n_pes`、rank、`ip_port`、device id、shape/dtype。
- 输出目录按 PE/case 隔离。
- 等待所有 PE 退出，并聚合返回码。
- 运行 checker，返回非 0 表示失败。

## 4. 正确性测试入口

测试脚本至少提供：

| 入口 | 目的 |
| --- | --- |
| smoke | 2 PE、小 shape、单 dtype，验证 init/malloc/launch/finalize |
| correctness | 按 `design.md` case matrix 验证 golden |
| stress | tail、chunk 边界、不均衡 PE、repeats |
| perf | correctness 通过后才可运行，带 warmup/repeats |

结果记录必须包含 command、workdir、case、oracle、tolerance、pass/fail 和日志路径。

## 5. 常见产物位置

| 模式 | 可执行文件 | kernel/shared library | 安装库 |
| --- | --- | --- | --- |
| 独立工程 | `<op>/build/bin/<target>` | `<op>/build/lib/*.so` | `$SHMEM_HOME_PATH/shmem/lib` |
| in-tree example | `shmem/build/bin/<op_name>` | `shmem/build/lib/lib<op_name>_kernel.so` | `shmem/install/shmem/lib` |
