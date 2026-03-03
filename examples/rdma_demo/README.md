使用方式:
1.在shmem/目录编译:
```bash
bash scripts/build.sh -enable_rdma -examples
```
2.在shmem/目录运行:
```bash
export PROJECT_ROOT=<shmem-root-directory>
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:$LD_LIBRARY_PATH
export SHMEM_UID_SESSION_ID=127.0.0.1:8899
./build/bin/rdma_demo 2 0 tcp://127.0.0.1:8765 2 0 0 & # pe 0
./build/bin/rdma_demo 2 1 tcp://127.0.0.1:8765 2 1 1 & # pe 1
```

3.命令行参数说明
    ./rdma_demo <n_pes> <relative_pe_id> <g_npus> <f_pe> <f_npu>

- n_pes: 全局Pe数量。
- ipport: SHMEM初始化需要的IP及端口号，格式为tcp://<IP>:<端口号>。如果执行跨机测试，需要讲IP设为pe0所在Host的IP。
- g_npus: 当前卡上启动的NPU数量。
- f_pe: 当前卡上使用的第一个Pe号。
- f_npu: 当前卡上使用的第一个NPU卡号。