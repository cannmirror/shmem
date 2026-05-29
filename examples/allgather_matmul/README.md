### 使用方式

**前置条件**：
- 需要安装 torch_npu 环境，数据生成脚本依赖 NPU 设备
- 确保已正确配置 CANN 和 torch_npu 环境

1. **编译项目**  
   在 `shmem/` 根目录下执行编译脚本：
   ```bash
   bash scripts/build.sh -examples
   ```

2. **运行AllGather-MatMul示例程序**  
   进入示例目录并执行运行脚本：
   ```bash
   cd examples/allgather_matmul
   bash scripts/run.sh [device_list]
   ```

   - **参数说明**：
     - `device_list`：指定用于运行的设备（NPU）编号列表，以逗号分隔。
     - 示例：使用第6和第7个NPU设备运行2卡AllGather-MatMul示例：
       ```bash
       bash scripts/run.sh 6,7
       ```

   - **配置计算规模**：  
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。  
     修改该文件以定义测试用例的输入维度。
