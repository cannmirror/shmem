# shmem算子接入torch样例
该目录提供了shmem部分算子接入torch后的使用效果，仅做展示使用，不建议生产环境使用！
## 编译运行
在shmem根目录执行
```sh
# 编译example算子用例及其torch扩展
bash scripts/build.sh -python_example
source install/set_env.sh
cd examples/python_extension/torch_test
python xxx.py # 默认拉起八卡用例
python xxx.py --pes 2 # --pes可以指定卡数，拉起两卡用例
```