# 参考资料索引

本目录包含 `shmem-ops-performance-optim` 所需的参考资料。

| 文件 | 用途 | 何时读取 |
| --- | --- | --- |
| [optimization-patterns.md](optimization-patterns.md) | 通信、流水线、同步、内存、分核优化手段 | OptimStep 1 定位瓶颈后选择对应优化 |
| [compute-optimization.md](compute-optimization.md) | 通算融合中 Matmul/Compute 调优（TileShape、Swizzle、DispatchPolicy、SplitK、存储层次） | 瓶颈在 compute 部分时 |
