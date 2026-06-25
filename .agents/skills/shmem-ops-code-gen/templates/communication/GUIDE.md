# Communication 算子模板

## 适用条件

- `design.md` 的 `op_kind` 为 `transport`、`collective` 或无本地重计算的通信模式
- 主要逻辑是 put/get/exchange/reduce/scatter/gather
- compute 只限于简单 pack/unpack、copy、local add 或格式辅助

## 使用方法

Agent 读取本 GUIDE.md，按以下标题格式匹配各模板代码块，提取代码块内容并写入目标路径：

```text
## <relative_path> → <op_name>/<target_path>
```

具体步骤：

1. 在 GUIDE.md 中找到上述标题格式
2. 提取标题下方 fenced code block 中的完整代码内容
3. 将内容写入 `<op_name>/<target_path>`（如 `<op_name>/CMakeLists.txt`、`<op_name>/src/main.cpp` 等）
4. 替换所有占位符：`<op_name>` → 算子名（snake_case），`<OpName>` → CamelCase，`<OP_NAME>` → UPPER_CASE

本目录不提供独立模板文件；所有模板代码均内嵌在本 GUIDE.md 的 fenced code block 中，Agent 只需提取代码块内容并写入目标路径。

## 模板文件 → 目标路径

| 代码块标题（relative_path） | 目标路径 | 语言 |
| --- | --- | --- |
| `CMakeLists.txt` | `<op_name>/CMakeLists.txt` | cmake |
| `src/main.cpp` | `<op_name>/src/main.cpp` | cpp |
| `src/<op_name>_kernel.h` | `<op_name>/src/<op_name>_kernel.h` | cpp |
| `src/<op_name>_kernel.cpp` | `<op_name>/src/<op_name>_kernel.cpp` | cpp |
| `scripts/gen_data.py` | `<op_name>/scripts/gen_data.py` | python |
| `scripts/check_result.py` | `<op_name>/scripts/check_result.py` | python |
| `scripts/run.sh` | `<op_name>/scripts/run.sh` | bash |

baseline 模板（有 baseline 时 Agent 需额外生成，无内嵌代码块）：

| 生成目标路径 | 用途 |
| --- | --- |
| `<op_name>/baseline/CMakeLists.txt` | 独立 baseline 编译 target |
| `<op_name>/baseline/src/<op_name>_baseline.cpp` | HCCL/aclnn/拼接 baseline 源码 |
| `<op_name>/baseline/scripts/run_baseline.sh` | baseline 运行脚本 |

## 定制要点

1. 从 GUIDE.md 提取代码块：按标题匹配，写入目标目录（`src/`、`scripts/` 等）
2. 替换占位符：`<op_name>` → 算子名（snake_case），`<OpName>` → CamelCase，`<OP_NAME>` → UPPER_CASE
3. `src/main.cpp`：按 design.md 的 interface 调整输入输出参数和 symmetric buffer 大小
4. `src/<op_name>_kernel.cpp`：按 design.md 的 schedule.phases 实现具体 phase 逻辑（put/get 目标、chunk 大小、signal slot）
5. `scripts/gen_data.py`：按 design.md 的 correctness.oracle 实现 golden 计算
6. `scripts/run.sh`：按 compile/test contract 调整路径和参数

## 约束

- `src/main.cpp` 只做单 PE 编排，不含 golden/checker/route/tiling 复杂逻辑
- kernel 必须使用 `aclshmem_*`/`aclshmemx_*` 跨 PE 通信，禁止 `DataCopy` 写远端
- 详细约束和性能指标要求见 [SKILL.md](../../SKILL.md) 和 [references/](../../references/)

---

## CMakeLists.txt → `<op_name>/CMakeLists.txt`

```cmake
# ==== 定制：替换 <op_name> 为算子名 ====
# 独立工程构建模式：显式引用 src/ 下源码，include src/ 和 examples/utils/
cmake_minimum_required(VERSION 3.16)
project(<op_name> LANGUAGES CXX)

set(CANN_ENV "$ENV{ASCEND_HOME_PATH}")
if(NOT CANN_ENV)
    message(FATAL_ERROR "ASCEND_HOME_PATH not set; source set_env.sh first")
endif()

set(SHMEM_ROOT "${CANN_ENV}/shmem")
set(EXAMPLE_UTILS "${SHMEM_ROOT}/examples/utils")

include_directories(
    ${CANN_ENV}/include
    ${SHMEM_ROOT}/include
    ${SHMEM_ROOT}/examples/utils
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(<op_name>
    src/main.cpp
    src/<op_name>_kernel.cpp
)
target_link_libraries(<op_name> aclshmem ascendcl)
```

---

## src/main.cpp → `<op_name>/src/main.cpp`

```cpp
// ============================================================
// SHMEM Communication 算子 Host 模板
// 来源: shmem/examples/allgather/main.cpp 简化
// 使用: 写入 <op_name>/src/main.cpp，替换 <op_name>/<OpName> 占位符
// ============================================================

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cstdint>

#include "acl/acl.h"
#include "shmem.h"

#include "<op_name>_kernel.h"

#define CHECK_ACL(call)                                                     \
    do {                                                                    \
        auto _ret = (call);                                                 \
        if (_ret != 0) {                                                    \
            std::cerr << #call << " failed: " << _ret << std::endl;         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

#define CHECK_SHMEM(call)                                                   \
    do {                                                                    \
        int _ret = (call);                                                  \
        if (_ret != 0) {                                                    \
            std::cerr << #call << " failed: " << _ret << std::endl;         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// ---- SHMEM init helper（内联，不依赖 examples/utils）----
static void init_set_attr(int pe_id, int n_pes, uint64_t local_mem_size,
                          const char *ip_port,
                          aclshmemx_uniqueid_t &uid,
                          aclshmemx_init_attr_t &attr)
{
    attr.my_pe      = pe_id;
    attr.n_pes      = n_pes;
    attr.local_mem_size = local_mem_size;
    strncpy(attr.ip_port, ip_port, sizeof(attr.ip_port) - 1);
    attr.ip_port[sizeof(attr.ip_port) - 1] = '\0';

    if (pe_id == 0) {
        aclshmemx_get_uniqueid(&uid);
    }
    attr.uniqueid = uid;
}

// ---- FFTS config helper ----
static uint64_t get_ffts_config()
{
    uint64_t fftsAddr = 0;
    CHECK_ACL(aclrtGetMemInfo(ACL_HBM_MEM, &fftsAddr, nullptr));
    return fftsAddr;
}

// ---- 文件读写工具 ----
template <typename T>
void ReadBinaryFile(const std::string &path, T *dst, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    f.read(reinterpret_cast<char *>(dst), count * sizeof(T));
}

template <typename T>
void WriteBinaryFile(const std::string &path, const T *src, size_t count)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot create " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    f.write(reinterpret_cast<const char *>(src), count * sizeof(T));
}

// ---- 参数 ----
struct Options {
    int n_pes = 2;
    int pe_id = 0;
    std::string ip_port = "tcp://127.0.0.1:8998";
    std::string data_dir = "./data";
    int perf_times = 0;  // 0 = correctness only
    int warmup = 5;
    // ==== 定制：添加算子特有参数（shape、dtype 等）====
    int elements = 1024;

    int Parse(int argc, char **argv)
    {
        if (argc < 5) {
            std::cerr << "Usage: <op_name> <n_pes> <pe_id> <ip_port> <data_dir>"
                      << " [elements] [perf_times]" << std::endl;
            return -1;
        }
        n_pes     = std::atoi(argv[1]);
        pe_id     = std::atoi(argv[2]);
        ip_port   = argv[3];
        data_dir  = argv[4];
        if (argc > 5) elements   = std::atoi(argv[5]);
        if (argc > 6) perf_times = std::atoi(argv[6]);
        return 0;
    }
};

int main(int argc, char *argv[])
{
    Options opts;
    if (opts.Parse(argc, argv) != 0) return 1;

    int pe_id = opts.pe_id;
    int n_pes = opts.n_pes;
    int32_t device_id = pe_id;  // ==== 定制：多卡映射 ====

    // ---- ACL & SHMEM init ----
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));

    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    aclshmemx_uniqueid_t uid;
    aclshmemx_init_attr_t attributes;
    init_set_attr(pe_id, n_pes, local_mem_size, opts.ip_port.c_str(), uid, attributes);
    CHECK_ACL(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint64_t fftsAddr = get_ffts_config();

    // ---- 读取输入 ----
    // ==== 定制：按 design.md 的 interface.inputs 调整 ====
    size_t input_bytes = opts.elements * sizeof(int32_t);
    void *input_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&input_dev, input_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<int32_t> input_host(opts.elements);
    std::string input_path = opts.data_dir + "/input_pe" + std::to_string(pe_id) + ".bin";
    ReadBinaryFile(input_path, input_host.data(), opts.elements);
    CHECK_ACL(aclrtMemcpy(input_dev, input_bytes, input_host.data(), input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 分配输出 ----
    // ==== 定制：按 design.md 的 interface.outputs 和 visibility 调整 ====
    size_t output_elems = opts.elements * n_pes;
    size_t output_bytes = output_elems * sizeof(int32_t);
    void *output_dev = nullptr;
    CHECK_ACL(aclrtMalloc(&output_dev, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // ---- 分配 symmetric buffer ----
    // ==== 定制：按 design.md 的 memory.buffers 调整大小 ====
    size_t symm_size = opts.elements * sizeof(int32_t) + 16 * 16 * sizeof(int32_t);
    if (symm_size == 0) {
        std::cerr << "symmetric buffer size is zero" << std::endl;
        return 1;
    }
    void *symm_ptr = aclshmem_malloc(symm_size);
    if (symm_ptr == nullptr) {
        std::cerr << "aclshmem_malloc failed, size=" << symm_size << std::endl;
        return 1;
    }

    // ---- 执行 ----
    uint32_t block_num = 2 * n_pes;  // ==== 定制：按 schedule.core_partition ====
    int magic = 1;

    // ==== 定制：staging 拷贝函数 —— 将 input_dev 拷贝到 symmetric buffer ====
    // staging 是 e2e 性能循环的必要组成，按 design.md 的 memory.staging 选择路径
    auto staging_copy = [&](void *src, void *dst, size_t bytes) {
        CHECK_ACL(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
    };

    if (opts.perf_times > 0) {
        // warmup（含 staging + barrier + kernel）
        for (int i = 0; i < opts.warmup; i++) {
            magic++;
            staging_copy(input_dev, symm_ptr, input_bytes);
            aclshmem_barrier_all();
            <op_name>_kernel<int32_t>(block_num, stream, fftsAddr,
                (uint8_t *)input_dev, (uint8_t *)output_dev, (uint8_t *)symm_ptr,
                opts.elements, magic);
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }

        // ---- perf timing：单循环同时采集 e2e 和 kernel-us ----
        double kernel_us_acc = 0;
        auto t0_e2e = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < opts.perf_times; i++) {
            magic++;
            staging_copy(input_dev, symm_ptr, input_bytes);
            aclshmem_barrier_all();
            auto t0_kern = std::chrono::high_resolution_clock::now();
            <op_name>_kernel<int32_t>(block_num, stream, fftsAddr,
                (uint8_t *)input_dev, (uint8_t *)output_dev, (uint8_t *)symm_ptr,
                opts.elements, magic);
            CHECK_ACL(aclrtSynchronizeStream(stream));
            auto t1_kern = std::chrono::high_resolution_clock::now();
            kernel_us_acc += std::chrono::duration<double, std::micro>(t1_kern - t0_kern).count();
        }
        auto t1_e2e = std::chrono::high_resolution_clock::now();
        double e2e_us = std::chrono::duration<double, std::micro>(t1_e2e - t0_e2e).count()
                        / opts.perf_times;
        double kernel_us = kernel_us_acc / opts.perf_times;

        // ==== 定制：按算子类型设置 bus_factor ====
        // AllReduce: 2*(n-1)/n, ReduceScatter/AllGather: (n-1)/n, AllToAll/Shuffle: (n-1)/n, P2P: 1
        double bus_factor = (double)(n_pes - 1) / n_pes;
        double payload_bytes = (double)input_bytes;
        double algo_bw = payload_bytes / (e2e_us * 1e-6) / 1e9;
        double bus_bw  = algo_bw * bus_factor;
        double peak_bw = 196.0;  // ==== 定制：按通信模式和拓扑调整（P2P: 28, 集合 8卡 full-mesh: 196）====
        double util_pct = bus_bw / peak_bw * 100.0;

        if (pe_id == 0) {
            std::cout << "[PERF] e2e_us=" << e2e_us
                      << " kernel_us=" << kernel_us
                      << " algo_bandwidth_GBps=" << algo_bw
                      << " bus_bandwidth_GBps=" << bus_bw
                      << " bandwidth_utilization_pct=" << util_pct
                      << " payload_bytes=" << payload_bytes
                      << " bus_factor=" << bus_factor
                      << " peak_bandwidth_GBps=" << peak_bw
                      << std::endl;
        }

        aclshmemx_show_prof();
    } else {
        staging_copy(input_dev, symm_ptr, input_bytes);
        aclshmem_barrier_all();
        magic++;
        <op_name>_kernel<int32_t>(block_num, stream, fftsAddr,
            (uint8_t *)input_dev, (uint8_t *)output_dev, (uint8_t *)symm_ptr,
            opts.elements, magic);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }

    // ---- 拷回输出并写文件 ----
    std::vector<int32_t> output_host(output_elems);
    CHECK_ACL(aclrtMemcpy(output_host.data(), output_bytes, output_dev, output_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    std::string output_path = opts.data_dir + "/output_pe" + std::to_string(pe_id) + ".bin";
    WriteBinaryFile(output_path, output_host.data(), output_elems);

    // ---- 清理 ----
    aclshmem_free(symm_ptr);
    CHECK_ACL(aclrtFree(input_dev));
    CHECK_ACL(aclrtFree(output_dev));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_SHMEM(aclshmem_finalize());
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());

    std::cout << "[SUCCESS] <op_name> completed on PE " << pe_id << std::endl;
    return 0;
}
```

---

## src/<op_name>_kernel.h → `<op_name>/src/<op_name>_kernel.h`

```cpp
// ============================================================
// SHMEM Communication 算子 kernel 声明模板
// 使用: 写入 <op_name>/src/<op_name>_kernel.h
//       替换 <op_name>/<OpName> 占位符
// ============================================================

#ifndef <OP_NAME>_KERNEL_H
#define <OP_NAME>_KERNEL_H

#include <cstdint>

template <class T>
void <op_name>_kernel(uint32_t block_dim, void *stream, uint64_t fftsAddr,
                      uint8_t *input, uint8_t *output, uint8_t *symmetric,
                      int elements, int magic);

#endif // <OP_NAME>_KERNEL_H
```

---

## src/<op_name>_kernel.cpp → `<op_name>/src/<op_name>_kernel.cpp`

```cpp
// ============================================================
// SHMEM Communication 算子 Device kernel 模板
// 来源: shmem/examples/allgather/allgather_kernel.cpp 简化
// 使用: 复制到 <op_name>/src/<op_name>_kernel.cpp
//       替换 <op_name>/<OpName> 占位符，实现具体 phase 逻辑
// ============================================================

#include "<op_name>_kernel.h"
#include "kernel_operator.h"
#include "acl/acl.h"
#include "shmem.h"
#include "utils/prof/shmemi_prof.h"

#undef inline
#include "opdev/fp16_t.h"
#include "opdev/bfloat16.h"
#define inline inline attribute((always_inline))

using namespace AscendC;
using fp16_t = op::fp16_t;
using bf16_t = op::bfloat16;

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;

// ==== 定制：按 design.md 的 schedule.phases 实现具体通信逻辑 ====
template <typename T>
ACLSHMEM_DEVICE void <op_name>_impl(uint64_t fftsAddr,
                                     __gm__ T *input,
                                     __gm__ T *output,
                                     __gm__ T *symmetric,
                                     int elements,
                                     int magic)
{
    util_set_ffts_config(fftsAddr);

    const int64_t aivNum = GetBlockNum();
    const int64_t aivIndex = GetBlockIdx();
    const int64_t flag_offset = aivIndex * SYNC_FLAG_INTERVAL;

    int64_t my_rank = aclshmem_my_pe();
    int64_t pe_size = aclshmem_n_pes();

    // ---- 分核 ----
    // ==== 定制：按 schedule.core_partition 划分 sender / receiver / sync 核 ====
    // 示例：前半核为 sender（本地 → symmetric），后半核为 receiver（远端 symmetric → 本地）
    int core_group_num = aivNum / 2;
    bool is_sender = (aivIndex < core_group_num);

    // Symmetric buffer 布局
    __gm__ int32_t *sync_flags = (__gm__ int32_t *)symmetric;
    __gm__ T *data_buf = (__gm__ T *)(sync_flags + aivNum * SYNC_FLAG_INTERVAL);

    // UB 临时缓冲（ping-pong）
    __ubuf__ T *ping_buf = reinterpret_cast<__ubuf__ T *>(uint64_t(1024 + 32));
    __ubuf__ T *pong_buf = reinterpret_cast<__ubuf__ T *>(uint64_t(96 * 1024 + 32));

    uint32_t ub_size = UB_DMA_MAX_SIZE;
    uint32_t ub_elems = ub_size / sizeof(T);

    if (is_sender) {
        // ==== Phase: 本地 GM → symmetric (sender 核) ====
        // ==== 定制：按 design.md 的 phase 逻辑调整 put 目标和 chunk 策略 ====
        int64_t per_core = elements / core_group_num;
        int64_t my_offset = aivIndex * per_core;
        if (aivIndex == core_group_num - 1) {
            per_core = elements - my_offset;
        }

        int64_t remaining = per_core * sizeof(T);
        int64_t done_elems = 0;
        int64_t chunk_count = 0;

        while (remaining > 0) {
            uint32_t copy_elems = (remaining > ub_size) ? ub_elems : (remaining / sizeof(T));

            SHMEMI_PROF_START(0);  // copy_in phase
            aclshmemx_mte_put_nbi(data_buf + my_offset + done_elems,
                                  input + my_offset + done_elems,
                                  ping_buf, ub_size, copy_elems,
                                  my_rank, EVENT_ID0);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            SHMEMI_PROF_END(0);

            chunk_count++;
            int64_t flag_val = chunk_count + magic;
            aclshmemx_signal_op(sync_flags + flag_offset, flag_val,
                                ACLSHMEM_SIGNAL_SET, my_rank);

            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            done_elems += copy_elems;
            remaining -= copy_elems * sizeof(T);
        }
    } else {
        // ==== Phase: 远端 symmetric → 本地 output (receiver 核) ====
        // ==== 定制：按 design.md 的 peer_model 和 addressing 调整 get 来源 ====
        int receiver_idx = aivIndex - core_group_num;
        // 示例：每个 receiver 核从一个远端 PE 拉取数据
        int64_t src_pe = receiver_idx;  // ==== 定制：映射逻辑 ====

        // 等待远端 sender 完成后 get
        // ==== 定制：按 schedule.phases 的 dependency 和 sync 调整等待逻辑 ====
        aclshmem_signal_wait_until(
            (__gm__ int32_t *)aclshmem_ptr(sync_flags, src_pe) + flag_offset,
            ACLSHMEM_CMP_GE, magic + 1);

        int64_t per_core = elements;
        int64_t dst_offset = src_pe * elements;
        int64_t remaining = per_core * sizeof(T);
        int64_t done_elems = 0;
        int pingpong = 0;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        while (remaining > 0) {
            uint32_t copy_elems = (remaining > ub_size / 2) ? (ub_size / 2 / sizeof(T))
                                                            : (remaining / sizeof(T));
            TEventID eid = (pingpong == 0) ? EVENT_ID0 : EVENT_ID1;
            __ubuf__ T *buf = (pingpong == 0) ? ping_buf : pong_buf;

            WaitFlag<HardEvent::MTE3_MTE2>(eid);

            SHMEMI_PROF_START(1);  // remote_get phase
            aclshmemx_mte_get_nbi(output + dst_offset + done_elems,
                                  data_buf + done_elems,
                                  buf, ub_size / 2, copy_elems,
                                  src_pe, eid);
            SHMEMI_PROF_END(1);

            SetFlag<HardEvent::MTE3_MTE2>(eid);

            done_elems += copy_elems;
            remaining -= copy_elems * sizeof(T);
            pingpong = 1 - pingpong;
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

// ---- Kernel 入口（按 dtype 展开）----
// ==== 定制：按 design.md 的 dtypes 调整支持的类型 ====
#define <OP_NAME>_FUNC_DEF(type)                                                    \
    extern "C" __global__ __aicore__ void Shmem<OpName>_##type(                     \
        uint64_t fftsAddr, GM_ADDR input, GM_ADDR output,                           \
        GM_ADDR symmetric, int elements, int magic)                                 \
    {                                                                               \
        <op_name>_impl<type>(fftsAddr, (__gm__ type *)input,                        \
                             (__gm__ type *)output, (__gm__ type *)symmetric,        \
                             elements, magic);                                      \
    }

<OP_NAME>_FUNC_DEF(int32_t);
<OP_NAME>_FUNC_DEF(float16_t);
<OP_NAME>_FUNC_DEF(bfloat16_t);

// ---- Host-callable wrapper ----
template <class T>
void <op_name>_kernel(uint32_t block_dim, void *stream, uint64_t fftsAddr,
                      uint8_t *input, uint8_t *output, uint8_t *symmetric,
                      int elements, int magic)
{
    if (std::is_same<T, int32_t>::value) {
        Shmem<OpName>_int32_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, input, output, symmetric, elements, magic);
    } else if (std::is_same<T, fp16_t>::value) {
        Shmem<OpName>_float16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, input, output, symmetric, elements, magic);
    } else if (std::is_same<T, bf16_t>::value) {
        Shmem<OpName>_bfloat16_t<<<block_dim, nullptr, stream>>>(
            fftsAddr, input, output, symmetric, elements, magic);
    }
}

// ==== 定制：按实际使用的 dtype 显式实例化 ====
template void <op_name>_kernel<int32_t>(uint32_t, void *, uint64_t,
    uint8_t *, uint8_t *, uint8_t *, int, int);
template void <op_name>_kernel<fp16_t>(uint32_t, void *, uint64_t,
    uint8_t *, uint8_t *, uint8_t *, int, int);
template void <op_name>_kernel<bf16_t>(uint32_t, void *, uint64_t,
    uint8_t *, uint8_t *, uint8_t *, int, int);
```

---

## scripts/gen_data.py → `<op_name>/scripts/gen_data.py`

```python
#!/usr/bin/env python3
# ============================================================
# SHMEM Communication 算子数据生成 + golden 计算模板
# 来源: shmem/examples/allgather/scripts/data_gen.py
# 使用: 复制到 <op_name>/scripts/gen_data.py，
#       按 design.md 的 correctness.oracle 实现 golden 逻辑
# ============================================================

import argparse
import json
import os
import numpy as np

np.random.seed(42)

DTYPE_MAP = {
    "int32":    np.int32,
    "float16":  np.float16,
    "float32":  np.float32,
    "bfloat16": "bfloat16",
}


def get_dtype(name: str):
    if name == "bfloat16":
        from ml_dtypes import bfloat16
        return bfloat16
    return DTYPE_MAP[name]


def gen_random_data(shape, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 1000, size=shape, dtype=dtype)
    return np.random.uniform(0.0, 10.0, size=shape).astype(dtype)


# ==== 定制：按 design.md 的 correctness.oracle 实现 golden 逻辑 ====
def compute_golden(inputs, n_pes, elements, dtype):
    """示例: allgather golden — 拼接所有 PE 的输入"""
    golden = np.concatenate(inputs, axis=0)
    return golden


def main():
    parser = argparse.ArgumentParser(description="生成输入数据和 golden")
    parser.add_argument("--n_pes",    type=int, required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--dtype",    type=str, default="int32")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    os.makedirs(args.data_dir, exist_ok=True)

    inputs = []
    for pe in range(args.n_pes):
        data = gen_random_data((args.elements,), dtype)
        data.tofile(os.path.join(args.data_dir, f"input_pe{pe}.bin"))
        inputs.append(data)

    golden = compute_golden(inputs, args.n_pes, args.elements, dtype)
    golden.tofile(os.path.join(args.data_dir, "golden.bin"))

    config = {
        "n_pes":    args.n_pes,
        "elements": args.elements,
        "dtype":    args.dtype,
        "rtol":     0,
        "atol":     0,
    }
    with open(os.path.join(args.data_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated data for {args.n_pes} PEs, {args.elements} elements, dtype={args.dtype}")


if __name__ == "__main__":
    main()
```

---

## scripts/check_result.py → `<op_name>/scripts/check_result.py`

```python
#!/usr/bin/env python3
# ============================================================
# SHMEM 算子精度验证模板
# 来源: shmem-ops-testcase-gen/references/correctness.md 标准 checker
# 使用: 复制到 <op_name>/scripts/check_result.py，
#       按 design.md 的 correctness.tolerance 调整阈值
# ============================================================

import argparse
import json
import sys
import numpy as np


def get_dtype(name: str):
    dtype_map = {
        "int32":    np.int32,
        "float16":  np.float16,
        "float32":  np.float32,
    }
    if name == "bfloat16":
        from ml_dtypes import bfloat16
        return bfloat16
    return dtype_map[name]


def check_precision(output, golden, rtol, atol, label=""):
    if output.shape != golden.shape:
        print(f"\033[31m✗ SHAPE MISMATCH {label}: output={output.shape} golden={golden.shape}\033[0m")
        return False

    if np.any(np.isnan(output)):
        nan_count = np.sum(np.isnan(output))
        print(f"\033[31m✗ NaN DETECTED {label}: {nan_count} NaN values\033[0m")
        return False

    if np.any(np.isinf(output)):
        inf_count = np.sum(np.isinf(output))
        print(f"\033[31m✗ Inf DETECTED {label}: {inf_count} Inf values\033[0m")
        return False

    diff = np.abs(output.astype(np.float64) - golden.astype(np.float64))
    golden_abs = np.abs(golden.astype(np.float64))
    rel_error = diff / (golden_abs + 1e-8)

    if rtol == 0 and atol == 0:
        mask = (output != golden)
    else:
        mask = (rel_error >= rtol) & (diff >= atol)

    error_indices = np.where(mask.flatten())[0]

    if len(error_indices) == 0:
        print(f"\033[32m✓ PASS {label}: all {output.size} elements within tolerance "
              f"(rtol={rtol}, atol={atol})\033[0m")
        return True

    print(f"\033[31m✗ FAIL {label}: {len(error_indices)}/{output.size} elements exceed tolerance\033[0m")
    print(f"  Max absolute error: {np.max(diff):.6e}")
    print(f"  Max relative error: {np.max(rel_error):.6e}")
    print(f"  Mean absolute error: {np.mean(diff):.6e}")

    flat_out = output.flatten()
    flat_gol = golden.flatten()
    for idx in error_indices[:10]:
        print(f"  Index {idx}: output={flat_out[idx]}, golden={flat_gol[idx]}, "
              f"diff={diff.flatten()[idx]:.6e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="验证算子输出精度")
    parser.add_argument("--data_dir", type=str, required=True)
    # ==== 定制：如果每个 PE 的 golden 不同，调整 golden 读取逻辑 ====
    args = parser.parse_args()

    with open(f"{args.data_dir}/config.json") as f:
        config = json.load(f)

    dtype = get_dtype(config["dtype"])
    rtol = config.get("rtol", 0)
    atol = config.get("atol", 0)
    n_pes = config["n_pes"]

    golden = np.fromfile(f"{args.data_dir}/golden.bin", dtype=dtype)

    all_pass = True
    for pe in range(n_pes):
        output_path = f"{args.data_dir}/output_pe{pe}.bin"
        output = np.fromfile(output_path, dtype=dtype)
        if not check_precision(output, golden, rtol, atol, label=f"PE {pe}"):
            all_pass = False

    if all_pass:
        print(f"\033[32m✓ ALL {n_pes} PEs PASSED\033[0m")
        return 0
    else:
        print(f"\033[31m✗ SOME PEs FAILED\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## scripts/run.sh → `<op_name>/scripts/run.sh`

```bash
#!/bin/bash
# ============================================================
# SHMEM 算子多 PE 启动脚本模板
# 来源: shmem/examples/matmul_reduce_scatter/scripts/run.sh
# 使用: 复制到 <op_name>/scripts/run.sh
#       替换 <op_name> 占位符，按 compile/test contract 调整路径
# ============================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OP_DIR=$(dirname "$SCRIPT_DIR")
# ==== 定制：按实际仓库结构调整 ====
PROJECT_ROOT=$(dirname $(dirname "$OP_DIR"))
EXEC_BIN="${PROJECT_ROOT}/build/bin/<op_name>"
DATA_DIR="${OP_DIR}/data"

# ---- 参数 ----
DEVICE_LIST="${1:-0,1}"
IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_LIST"
PE_SIZE=${#DEVICE_ID_LIST[@]}
ELEMENTS="${2:-1024}"
PERF_TIMES="${3:-0}"  # 0 = correctness only
IPPORT="tcp://127.0.0.1:8998"
DTYPE="${4:-int32}"

echo "=== <op_name> === PE_SIZE=${PE_SIZE} ELEMENTS=${ELEMENTS} DTYPE=${DTYPE}"

# ---- 生成数据和 golden ----
mkdir -p "${DATA_DIR}"
python3 "${SCRIPT_DIR}/gen_data.py" \
    --n_pes "${PE_SIZE}" \
    --elements "${ELEMENTS}" \
    --dtype "${DTYPE}" \
    --data_dir "${DATA_DIR}"

# ---- SHMEM 环境 ----
export SHMEM_UID_SESSION_ID="${IPPORT}"

# ---- 启动多 PE 进程 ----
PIDS=()
for (( idx = 0; idx < PE_SIZE; idx++ )); do
    ${EXEC_BIN} "${PE_SIZE}" "${idx}" "${IPPORT}" "${DATA_DIR}" \
        "${ELEMENTS}" "${PERF_TIMES}" &
    PIDS+=($!)
done

# ---- 等待所有进程 ----
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || EXIT_CODE=1
done

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Some PE processes failed"
    exit 1
fi

# ---- 验证结果 ----
if [ "${PERF_TIMES}" -eq 0 ]; then
    python3 "${SCRIPT_DIR}/check_result.py" --data_dir "${DATA_DIR}"
    exit $?
fi
```

---
