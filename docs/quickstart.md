# 快速开始

## 1 介绍
本系统主要面向昇腾平台上的模型和算子开发者，提供便捷易用的多机多卡内存访问方式，方便用户开发在卡间同步数据，加速通信或通算融合类算子开发。  

## 2 软件架构
共享内存库接口主要分为 host 和 device 接口部分：
- host 侧接口提供初始化、内存管理、通信域管理以及同步功能。
- device 侧接口提供内存访问、同步以及通信域管理功能。

## 3 目录结构说明
详细介绍见[code_organization](code_organization.md)
``` 
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```

## 4 环境准备

### 4.1 硬件要求
- Atlas 系列：800I A2/A3、800T A2/A3
- 架构兼容：aarch64、x86

### 4.2 工具链依赖
   - gcc >= 7.3.0 / g++ >= 7.3.0 （注意：gcc与g++版本一致）
   - cmake ≥ 3.19
   - GLIBC ≥ 2.28
   - Python >= 3.9.0

### 4.3 CANN
#### 4.3.1 CANN 版本说明
| 驱动固件 | CANN 版本 | D2D | D2H/H2D | D2rH/rH2D | 其他依赖 |
| --- | --- | --- | --- | --- | --- |
| Ascend HDK 25.0.RC1.1 | 9.0.0-beta.2 及以上<br>[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA<br>SDMA | MTE | MTE | 使能 SDMA 需下载社区版 [ops-legacy 包](https://www.hiascend.com/developer/download/community/result?module=cann) |
| Ascend HDK 25.0.RC1.1 | 9.0.0 及以上<br>尝鲜版 toolkit 包：[x86_64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260305000326487/x86_64/Ascend-cann-toolkit_9.0.0_linux-x86_64.run) / [aarch64](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260305000326487/aarch64/Ascend-cann-toolkit_9.0.0_linux-aarch64.run) | MTE<br>RDMA<br>SDMA | MTE | MTE | 使能 SDMA 需下载 ops-legacy 包（按硬件平台）：[A2 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-x86_64.run) / [A2 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-910b-ops-legacy_9.0.0_linux-aarch64.run) / [A3 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-x86_64.run) / [A3 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260305_newest/cann-A3-ops-legacy_9.0.0_linux-aarch64.run) |
| Ascend HDK 25.0.RC1.1 | 8.5.0 及以上<br>[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA | MTE | MTE | 使能 A3 D2rH/rH2D：LingQu Computing Network [1.5.0 版本](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841/software)<br>升级指导：[安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/lingqu-computing-network-pid-258003841) |
| Ascend HDK 25.0.RC1.1 | 8.3.RC1 及以上<br>[社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann) | MTE<br>RDMA |  |  | |

#### 4.3.2 CANN 包安装
参考[快速安装 CANN](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=local)

商发版本只需要安装`toolkit`的`.run`包即可。

社区版本需要额外安装一个`{soc_name}-ops`的`.run`安装包，需要选择cann版本，选择对应的soc、操作系统、架构的版本。（如果没有`{soc_name}-ops`的算子包，可能会提示缺少`libhccl.so`库文件）。

社区资源参考：[CANN 9.0.0 社区版资源](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)。

soc的信息如下，可以参考安装。

|soc version|备注|
|-----------|---|
|Ascend 910B|A2|
|Ascend 910C|A3|
|Ascend 950	|无|

安装 toolkit 包 
```sh
chmod +x Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run
# 默认安装路径
./Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run --install
# 自定义安装路径
./Ascend-cann-toolkit_${cann_version}_linux-$(uname -m).run --install --install-path=${install_path}
```

安装 ops 包
```sh
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run
# 默认安装路径
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run --install
# 自定义安装路径
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-$(uname -m).run --install --install-path=${install_path}
```

配置 CANN 环境变量（默认安装路径）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

配置 CANN 环境变量（自定义安装路径）：
```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

### 4.4 安装`Pytorch`框架和`torch_npu`插件
编译运行 torch 输入输出 tensor 的算子时必须安装本包。

以`CANN 9.0.0`为例，参考 [Ascend Extension for PyTorch 26.0.0-昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) 文档。

进入该页面，根据系统架构、CANN版本、Python版本，选择对应的版本。然后参考页面下方提示的安装命令进行安装。

以`Python 3.11`，`PyTorch 2.7.1`，`torch_npu 26.0.0`，`aarch64`版本为例。
+ 安装`PyTorch`框架：
```sh
# 下载PyTorch框架
wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl
# 安装
pip3 install torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
```
+ 安装`torch_npu`插件
```sh
# 下载torch_npu插件
wget https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.7.1/torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_aarch64.whl
# 安装命令
pip3 install torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_aarch64.whl
```


### 4.5 Python 依赖安装
```bash
# 基础依赖（构建期 + 运行期硬依赖）
python3 -m pip install -r requirements.txt

# 运行 Python 示例所需的额外依赖（按需安装）
python3 -m pip install -r requirements-examples.txt
```

### 4.6 可选依赖
- MPI：OpenMPI 4.0+（分布式通信场景）
- PyTorch：1.12+（Python 示例运行）


## 5 快速上手

### 5.1 安装方式
#### 5.1.1 方式一：源码编译

```bash
# 克隆代码仓
git clone https://gitcode.com/cann/shmem.git
cd shmem

# 编译核心库（默认不包含xDMA能力，也不包含示例和测试）
bash scripts/build.sh

# 配置环境变量
source install/set_env.sh
```
备注：build.sh 的参数可以参考 [compilation_build_guide.md](./compilation_build_guide.md)

#### 5.1.2 方式二：二进制包安装
获取方式：`bash scripts/build.sh -package`

软件包格式：`SHMEM_{version}_linux-{arch}.run`

其中，`{version}`表示软件版本号，`{arch}`表示`CPU`架构。<br>
```sh
chmod +x 软件包名.run # 增加对软件包的可执行权限
./软件包名.run --check # 校验软件包安装文件的一致性和完整性
./软件包名.run --install # 安装软件，可使用--help查询相关安装选项
```
出现提示`xxx install success!`则安装成功

配置环境变量：
```sh
# 配置环境变量
# （默认路径：/usr/local/Ascend/shmem）
source /usr/local/Ascend/shmem/latest/set_env.sh
# （自定义路径：${install_path}/shmem）
source ${install_path}/shmem/latest/set_env.sh
```

shmem 默认开启tls通信加密。如果需要关闭，需要调用接口主动关闭：
```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```
具体细节详见安全声明章节

### 5.2 样例执行

以 `matmul_allreduce` 为例，验证核心功能： 

1.在shmem/目录编译:

```sh
bash scripts/build.sh -examples
```

2.在shmem/examples/matmul_allreduce目录执行demo:

```sh
bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
```
注：example及其他样例代码仅供参考，在生产环境中请谨慎使用。

`run.sh`脚本提供`-ranks`、`-ipport`、`-test_filter`等参数自定义执行用例的卡数、ip端口、gtest_filter等  

例

```sh
# 8卡，ip:port 127.0.0.1:8666，运行所有*Init*用例
bash scripts/run.sh -ranks 8 -ipport tcp://127.0.0.1:8666 -test_filter Init
```

### 5.3 debug 模式使用
在源码 shmem/ 目录编译:

   ```sh
   bash scripts/build.sh -examples -debug
   ```

注意：此处 `-examples` 参数非必选项，仅提供[使用参考](./debug/Troubleshooting_FAQs.md#shmem-常见问题)。

## 6 Python 侧 test 用例     [Python 接口 API 列表](api/pythonAPI.md)
1. 在 scripts 目录下编译的时候，带上 build python 的选项

```sh
bash build.sh -python_extension
```

2. 在 install 目录下，source 环境变量

```sh
source set_env.sh
```

3. 在 src/python 目录下，进行 setup ，获取到 wheel 安装包

```sh
python3 setup.py bdist_wheel
```

4. 在src/python/dist目录下，安装wheel包

```sh
pip3 install shmem-xxx.whl --force-reinstall
```

5. 设置是否开启 TLS 认证，默认开启，若关闭 TLS 认证，请使用如下接口

```python
import shmem as shm
shm.set_conf_store_tls(False, "")   # 关闭tls认证
```

```python
import shmem as shm
tls_info = "xxx"
shm.set_conf_store_tls(True, tls_info)      # 开启TLS认证
```

6. 使用torchrun运行测试demo

```sh
torchrun --nproc-per-node=k test.py # k为想运行的ranksize
```
看到日志中打印出“test.py running success!”即为demo运行成功

## 7 `unique id` 初始化方式

注：使用unique id的接口初始化，可以配置环境变量SHMEM_UID_SESSION_ID或者SHMEM_UID_SOCK_IFNAME，同时配置时只读SHMEM_UID_SESSION_ID，都不配置时会自动搜索可用网口。SHMEM_UID_SESSION_ID支持IPv4地址、IPv6地址和主机名。SHMEM_UID_SESSION_ID配置示例：
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
SHMEM_UID_SESSION_ID=localhost:8888
SHMEM_UID_SOCK_IFNAME配置示例：
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  取ipv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  取ipv6
未配置时可自动搜索可用网口（IPv4/IPv6均可，优先非虚拟网口）。

- python初始化例子
```python
import shmem as ash

# xxx

uid = ash.aclshmem_get_unique_id()
ret = ash.aclshmem_init_using_unique_id(rank, world_size, mem_size, uid)

# xxx
```

准备以上启动代码`init.py`后，使用`torchrun --nproc-per-node 8 init.py`，其中进程数 8 根据实际需要改动，更详细的例子，请参考`unique_id_test.py`文件。

- c++ 初始化例子
```cpp
aclshmemx_uniqueid_t uid;
aclshmemx_init_attr_t *attr;
int ret = aclshmemx_get_uniqueid(&uid);
ret = aclshmemx_set_attr_uniqueid_args(my_pe, n_pes, mem_size, &uid, attr);
```
## 8 SHMEM 方式
注：使用unique id的接口初始化，可以手动配置环境变量SHMEM_UID_SESSION_ID或者SHMEM_UID_SOCK_IFNAME，同时配置时只读SHMEM_UID_SESSION_ID，都不配置会自动搜索可用网口。SHMEM_UID_SESSION_ID支持IPv4地址、IPv6地址和主机名。
SHMEM_UID_SESSION_ID配置示例：
SHMEM_UID_SESSION_ID=127.0.0.1:1234
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666]:886
SHMEM_UID_SESSION_ID=[6666:6666:6666:6666:6666:6666:6666:6666%eth]:886
SHMEM_UID_SESSION_ID=localhost:8888
SHMEM_UID_SOCK_IFNAME配置示例：
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet4  取ipv4
SHMEM_UID_SOCK_IFNAME=enpxxxx:inet6  取ipv6
不配置默认自动搜索可用网口(IPv4/IPv6均可)，搜索优先级：非虚拟网口(排除lo/docker/veth/br-/virbr/tun/tap) >> 虚拟网口。


- c++ 初始化例子
```cpp
aclshmemx_uniqueid_t uid;
aclshmemx_init_attr_t *attr;
int ret = aclshmemx_get_uniqueid(&uid);
aclshmemx_set_attr_uniqueid_args(rank, rank_size, local_mem_size, &uid, &attributes);
status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, attributes);
```


## 9 测试框架
- **单元测试：** 覆盖核心接口（初始化、内存操作、同步等），位于 `tests/unittest/`

- **算子泛化性测试：** 针对 `matmul_allreduce` 等样例，支持动态生成测试数据与精度校验

### 9.1 运行单元测试

```bash
# 编译并运行单元测试
bash scripts/build.sh -uttests
bash scripts/run.sh
```
run.sh 脚本提供 `-ranks`、`-test_filter` 等参数自定义执行用例的卡数、`gtest_filter` 等，例如：
```bash
# 8卡，运行所有*Init*用例
bash scripts/run.sh -ranks 8 -test_filter Init
```
具体参数请见[相关脚本-run.sh](docs/compilation_build_guide.md#SHMEM关键文件介绍)

### 9.2 样例编译与执行

```bash
bash scripts/build.sh -examples
bash scripts/run_examples.sh
```

### 9.3 样例单独执行
可查看 examples 目录下对应的样例目录下的 README.md

### 9.4 运行 Python 测试用例
```bash
# 编译 Python 扩展
bash scripts/build.sh -python_extension
# 安装构建脚本生成的 wheel 包
pip3 install dist/shmem-xxx.whl --force-reinstall
```

也可以在仓库根目录手动构建并安装 wheel 包：

```bash
python3 setup.py bdist_wheel
pip3 install dist/shmem-xxx.whl --force-reinstall
```

```bash
# 2卡运行 Python 测试
torchrun --nproc-per-node=2 examples/python_extension/test/init_test.py
```

### 9.5 自定义测试
基于 `tests/` 目录的 gtest 框架，新增测试用例需遵循：
- 测试文件命名：`{module}_test.cc`
- 测试用例命名：`{FunctionName}_{Scenario}_Test`

## 10 配置与调优（可选，进阶使用）
**1. 关闭 TLS 加密（提升通信性能）**

默认开启 TLS 加密，若为内网可信环境，可关闭以提升通信速度：
```c
int32_t ret = aclshmemx_set_conf_store_tls(false, NULL, 0);
```
```python
import shmem as shm
shm.set_conf_store_tls(False, "")
```

**2. 调整共享内存大小**

初始化时指定共享内存池大小（默认 16GB），适配大内存场景：
```c
aclshmemx_init_attr_t attr;
attr.local_mem_size = 32 * 1024 * 1024 * 1024; // 32GB
aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

**3. 性能调优建议**
- 优先使用 Device 侧接口（减少 Host-Device 交互）；
- 通信域分组时，按物理机节点划分，减少跨节点通信；
- 大批次数据传输时，开启 RDMA 协议（编译时加 `-DSHMEM_RDMA=ON`）。