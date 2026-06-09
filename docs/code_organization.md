# 代码组织结构
## SHMEM组织结构
```
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```

## include
```
include目录下的头文件是按照如下文件层级进行组织的
include/
├── shmem.h                                 // shmem所有对外API汇总
├── device/                                 // device侧头文件
│   ├── shmem_def.h                         // device侧定义的公共标准接口
│   ├── ub2gm/                              // device侧aicore驱动ub2gm数据面低阶接口，扩展接口+x
│   │   ├── shmem_device_rma.h              // aicore ub2gm 远端内存访问 RMA 接口
│   │   └── engine/                         // aicore 直驱ub2gm 低阶接口，扩展接口+x
│   │       └── shmem_device_mte.h          // aicore 直驱 mte接口
│   ├── gm2gm/                              // device侧aicore驱动gm2gm数据面高阶和低阶接口，扩展接口+x
│   │   ├── shmem_device_rma.h              // aicore high-level RMA 接口
│   │   ├── shmem_device_amo.h              // aicore high-level 原子内存操作接口
│   │   ├── shmem_device_so.h               // aicore high-level 信号操作接口
│   │   ├── shmem_device_cc.h               // aicore high-level 集合通信接口
│   │   ├── shmem_device_p2p_sync.h         // aicore high-level p2p同步接口
│   │   ├── shmem_device_mo.h               // aicore high-level 内存保序接口
│   │   └── engine/                         // aicore 直驱gm2gm 低阶接口，扩展接口+x
│   │       ├── shmem_device_rdma.h         // aicore 直驱 rdma 接口
│   │       ├── shmem_device_sdma.h         // aicore 直驱 sdma 接口
│   │       └── shmem_device_mte.h          // aicore 直驱 mte接口
│   └── team/                               // device侧通信域管理接口
│       └── shmem_device_team.h             // device侧通信域管理接口
├── host_device/                            // device_host共用目录
│       └── shmem_common_types.h            // 公共数据结构等
└── host/                                   // host侧头文件
    ├── shmem_host_def.h                    // host侧定义的接口
    ├── init/                               // host侧初始化接口
    │   └── shmem_host_init.h               // host初始化接口
    ├── team/                               // host侧通信域管理接口
    │   └── shmem_host_team.h               // host通信域管理接口
    ├── mem/                                // host侧内存管理接口
    │   └── shmem_host_heap.h               // host内存管理接口
    ├── data_plane/                         // host侧CPU驱动数据面接口
    │   ├── shmem_host_rma.h                // host high-level RMA 接口
    │   ├── shmem_device_so.h               // host high-level 信号操作接口
    │   ├── shmem_host_cc.h                 // host high-level 集合通信接口
    │   └── shmem_host_p2p_sync.h           // host high-level p2p同步接口
    └── utils/                              // host侧dfx接口
        └── shmem_log.h                     // host dfx-log接口
```
## src
```
└── src
    ├── device             // device侧接口实现
    └── host
        ├─bootstrap       // bootstrap接口实现
        ├─hybm            // hybrid memory
        ├─init            // host侧初始化接口实现
        ├─mem             // host侧内存管理接口实现
        ├─python_wrapper  // Py接口封装
        ├─sync            // 同步接口实现
        ├─team            // host侧通信域管理接口实现
        └─transport       // 建链相关内容
```
## examples
```
└── examples
    ├── allgather                              // allgather通信算子样例
    ├── allgather_matmul                       // allgather+matmul融合算子样例
    ├── allgather_matmul_padding               // 含padding的allgather+matmul融合算子样例
    ├── allgather_matmul_with_gather_result    // 融合allgather并保留gather的matmul样例
    ├── cmo                                    // cmo缓存性能样例
    ├── combine                                // moe combine样例
    ├── dispatch                               // moe dispatch样例
    ├── dispatch_gmm_combine                   // gmm分派与结果合并样例
    ├── dynamic_tiling                         // 动态分块实现样例
    ├── hccs_sio_link                          // hccs/sio链路测试样例
    ├── init                                   // shmem初始化流程样例
    ├── kv_shuffle                             // kv cache shuffle样例
    ├── matmul_allreduce                       // matmul+allreduce融合算子样例
    ├── matmul_reduce_scatter                  // matmul+reduce_scatter融合算子样例
    ├── matmul_reduce_scatter_padding          // 含padding的matmul+reduce_scatter融合算子样例
    ├── multi_instance                         // 多实例样例
    ├── notifywait                             // sdma notify/wait同步样例
    ├── python_extension                       // Python扩展与torch调用样例
    ├── rdma_aclgraph_demo                     // aclgraph 中调用RDMA allgather样例
    ├── rdma_demo                              // rdma实现样例
    ├── rdma_handlewait_test                   // rdma handle wait测试样例
    ├── rdma_perftest                          // rdma性能测试样例
    ├── rma_d2h_demo                           // device访问host内存rma样例
    ├── sdma                                   // sdma实现样例
    ├── simt_rma                               // simt 连续内存访问样例
    ├── simt_rma_scalar                        // simt 标量访问样例
    ├── torch_binding                          // pytorch binding样例
    ├── udma_atomic_add                        // udma atomic add样例
    ├── udma_demo                              // udma实现样例
    └── utils                                  // 样例公共工具
```
## tests
```
└── tests
    ├── examples
    └── unittest
        ├── init  // 初始化接口单元测试
        ├── mem   // 内存管理接口单元测试
        ├── sync  // 同步管理接口单元测试
        └── team  // 通信域管理接口单元测试
```
## docs
```
└── docs
    ├── api                          // shmem api
    ├── debug                        // shmem debug调测
    ├── deployment                   // SO文件部署与使用指导
    ├── doxygen
    ├── example                      // 使用样例
    ├── images                       //图片
    ├── SECURITY.md                  //安全声明
    ├── code_organization.md         // 工程组织架构（本文件）
    ├── compilation_build_guide.md   // 相关脚本介绍
    ├── conf.py                      // conf文件
    ├── index.rst                    // 目录树
    ├── multi_instance.md            // 多实例支持
    ├── principles.md                // SHMEM原理概述
    ├── quickstart.md                // 快速开始
```

## scripts
存放相关脚本。
[脚本具体功能和使用](compilation_build_guide.md)