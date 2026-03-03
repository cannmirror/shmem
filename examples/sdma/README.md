使用方式：
1.在`shmem/`目录编译软件包并安装：
```bash
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
```

2.在`shmem/`目录下编译examples：
```bash
bash scripts/build.sh -examples
```

3.在`shmem/examples/sdma`目录执行demo:
```bash
bash run.sh -pes ${PES} -type ${TYPES}
````
    - **参数说明**：
        - PES：指定用于运行的设备（NPU）数量，限定单台机器内。
        - TYPES：指定传输数据类型，当前支持：int，uint8，int64，fp32。
