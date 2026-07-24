# shmem-config 命令参考

`shmem-config` 是 SHMEM Python 包自带的配置查询工具，安装 `cann-shmem` 后即可使用。用于查询安装路径、后端类型、编译链接参数、环境检测和诊断信息。

## 命令概览

| 命令 | 用途 | 输出示例 |
|---|---|---|
| `shmem-config --version` | 查询 SHMEM 版本号 | `1.0.0` |
| `shmem-config --backend` | 当前 SoC 后端选择结果 | `910` 或 `950` |
| `shmem-config --include` | C/C++ 头文件 include 路径 | `/usr/local/lib/python3.x/.../shmem/include` |
| `shmem-config --lib` | 后端 .so 库目录路径 | `/usr/local/lib/python3.x/.../shmem/backends/910` |
| `shmem-config --ldflags` | 推荐链接参数（含 `-I`、`-L`、`-l`） | `-I.../include -L.../backends/910 -lshmem` |
| `shmem-config --rpath` | 推荐 rpath 链接参数 | `-Wl,-rpath,.../backends/910` |
| `shmem-config --root` | SHMEM 包根目录路径 | `/usr/local/lib/python3.x/.../shmem` |
| `shmem-config --runtime-root` | 运行时加载的 libshmem.so 根路径 | `/usr/local/lib/python3.x/.../shmem` |
| `shmem-config --diagnose` | 结构化诊断信息（JSON） | 见下方示例 |
| `shmem-config --check` | 运行安装前环境检测脚本 | 终端输出，含芯片/卡数/MTE/SDMA/UDMA/RDMA 状态 |

## 各命令详解

### --version

输出 SHMEM 包的版本号，数据来源为 `shmem/version.info` 的第一行。

```bash
$ shmem-config --version
1.0.0
```

### --backend

通过 `libascendcl.so` 的 `aclrtGetSocName()` 接口自动检测当前 SoC 型号，返回后端标识。

- `Ascend910` / `DAV-220*` 系列 → `910`
- `Ascend950` / `DAV-3510` 系列 → `950`
- 检测失败时兜底返回 `910`

```bash
$ shmem-config --backend
910
```

### --include / --lib / --root

查询 SHMEM 安装目录下的关键路径，常用于在 CMake 或 Makefile 中引用。

```bash
$ shmem-config --include
/usr/local/lib/python3.11/site-packages/shmem/include

$ shmem-config --lib
/usr/local/lib/python3.11/site-packages/shmem/backends/910

$ shmem-config --root
/usr/local/lib/python3.11/site-packages/shmem
```

### --ldflags / --rpath

输出编译链接时推荐的 flags，直接用于 `gcc` / `g++` 命令行或 CMake 变量。

```bash
$ shmem-config --ldflags
-I/usr/local/lib/python3.11/site-packages/shmem/include \
-L/usr/local/lib/python3.11/site-packages/shmem/backends/910 -lshmem

$ shmem-config --rpath
-Wl,-rpath,/usr/local/lib/python3.11/site-packages/shmem/backends/910
```

### --runtime-root

查询当前进程实际加载的 `libshmem.so` 所在包根路径。先读取 `/proc/self/maps`，再从 `LD_LIBRARY_PATH` 查找。

```bash
$ shmem-config --runtime-root
/usr/local/lib/python3.11/site-packages/shmem
```

### --diagnose

以 JSON 格式输出结构化诊断信息，适用于 CI 集成或自动化巡检。输出字段包括：

| 字段 | 说明 |
|---|---|
| `version` | SHMEM 包版本 |
| `backend.selected` | 当前选择的后端 |
| `backend.auto_detected_soc` | 自动检测的 SoC 型号 |
| `release_build` | 是否为 Release 构建 |
| `multi_so_conflict` | 是否存在多版本 libshmem.so 冲突 |
| `multi_so_conflict.loaded_paths` | 已加载的 libshmem.so 路径列表 |
| `backend_artifacts` | 后端 .so 文件完整性检查结果 |
| `runtime_root` | 运行时加载路径与包路径是否一致 |
| `degraded` | 是否存在降级运行 |
| `next_steps` | 诊断结论和建议操作 |

示例输出：

```json
{
  "version": "9.0.0.beta.2",
  "backend": {
    "selected": "910",
    "auto_detected_soc": "Ascend910"
  },
  "release_build": true,
  "multi_so_conflict": {
    "detected": false,
    "loaded_paths": []
  },
  "backend_artifacts": {
    "backend": "910",
    "complete": true,
    "missing": []
  },
  "runtime_root": {
    "path": "/usr/local/lib/python3.11/site-packages/shmem",
    "matches_package_root": true,
    "package_root": "/usr/local/lib/python3.11/site-packages/shmem"
  },
  "degraded": false,
  "next_steps": [
    "No issues detected. SHMEM is ready."
  ]
}
```

### --check [--package <路径>]

运行安装前环境检测脚本 `preinstall_check.sh`，检测内容包括：

1. 芯片平台识别
2. CANN / HDK 版本基线
3. 拓扑链路与 MTE 支持
4. SDMA 支持（910B/C 平台）
5. UDMA 支持（Ascend950 平台）
6. RDMA 网卡与网络健康状态
7. 包内容完整性（含 `--package` 时）

```bash
# 检测当前环境
shmem-config --check
```

## 使用场景

- **编译外部工程**：用 `--ldflags` 和 `--rpath` 拼装编译命令或 CMake 变量
- **CI 巡检**：用 `--diagnose` 输出 JSON 做自动化健康检查
- **问题排查**：用 `--runtime-root` 确认实际加载的 SHMEM 版本，用 `--diagnose` 检查 .so 冲突和完整性
- **部署前验证**：用 `--check` 一次性检测环境是否满足运行条件
