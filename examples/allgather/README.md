使用方式:

1. 在shmem/目录编译:
```
bash scripts/build.sh -examples
```
2. 在shmem/examples/allgather目录执行demo:
```
# 完成PEs卡下的allgather同时验证精度，性能数据会输出在result.csv中。
# PEs : [2, 4, 8]
# TYPEs : [int, int32_t, float16_t, bfloat16_t]
bash run.sh -pes ${PEs} -type ${TYPEs}
```

跨机使用方式:

1. 在shmem/目录编译: 
```
bash scripts/build.sh -examples
```

2. 在两台机器上shmem/examples/allgather目录中分别生成golden数据:
```
rm -rf ./golden
mkdir -p golden
python3 ./scripts/data_gen.py 8 "int"
```

3. 在其中一台机器上shmem/examples/allgather执行(ip_host1和ip_host2为各自机器的ip地址, PROJECT_ROOT为shmem/目录)

```
mpirun -host ip_host1:4,ip_host2:4 -np 8 ${PROJECT_ROOT}/build/bin/allgather
```