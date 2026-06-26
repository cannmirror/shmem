#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH

EXEC_BIN=${PROJECT_ROOT}/build/bin/rdma_perftest

# 默认测试类型为put
TEST_TYPE="put"
# 默认数据类型为float
DATA_TYPE="float"
# RDMA强制单核, block-size 仅作为入参兼容
BLOCK_SIZE_INPUT="1"
# 默认幂数范围
MIN_EXPONENT="3"
MAX_EXPONENT="17"
# 默认循环次数
LOOP_COUNT="1000"
# 默认UB size(KB) - RDMA需要至少64B的UB空间
UB_SIZE="64"
# RDMA特有参数
BATCH="0"
SYNC_ID="0"
QP_NUM="1"
# 默认RANK配置 - RDMA目前强制PE数量为2
PE_SIZE="2"
IPPORT="tcp://127.0.0.1:8768"
GNPU_NUM="2"
FIRST_NPU="0"
# 分析模式: none/plot/md
ANALYSE_MODE="none"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-type)
            if [ -n "$2" ]; then TEST_TYPE="$2"; shift 2; else echo "Error: -t requires a value."; exit 1; fi
            ;;
        -d|--datatype)
            if [ -n "$2" ]; then DATA_TYPE="$2"; shift 2; else echo "Error: -d requires a value."; exit 1; fi
            ;;
        --loop-count)
            if [ -n "$2" ]; then LOOP_COUNT="$2"; shift 2; else echo "Error: --loop-count requires a value."; exit 1; fi
            ;;
        --ub-size)
            if [ -n "$2" ]; then UB_SIZE="$2"; shift 2; else echo "Error: --ub-size requires a value."; exit 1; fi
            ;;
        --batch)
            if [ -n "$2" ]; then BATCH="$2"; shift 2; else echo "Error: --batch requires a value."; exit 1; fi
            ;;
        --sync-id)
            if [ -n "$2" ]; then SYNC_ID="$2"; shift 2; else echo "Error: --sync-id requires a value."; exit 1; fi
            ;;
        -q|--qp)
            if [ -n "$2" ]; then QP_NUM="$2"; shift 2; else echo "Error: -q requires a value."; exit 1; fi
            ;;
        -b|--block-size)
            if [ -n "$2" ]; then
                BLOCK_SIZE_INPUT="$2"
                if [[ "$BLOCK_SIZE_INPUT" != "1" ]]; then
                    echo "WARN: RDMA perftest forces block_size=1, ignoring -b $BLOCK_SIZE_INPUT"
                fi
                shift 2
            else echo "Error: -b requires a value."; exit 1; fi
            ;;
        --block-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                if [[ "$2" != "1" || "$3" != "1" ]]; then
                    echo "WARN: RDMA perftest forces block_size=1, ignoring --block-range $2 $3"
                fi
                shift 3
            else echo "Error: --block-range requires two values."; exit 1; fi
            ;;
        -e|--exponent)
            if [ -n "$2" ]; then MIN_EXPONENT="$2"; MAX_EXPONENT="$2"; shift 2;
            else echo "Error: -e requires a value."; exit 1; fi
            ;;
        --exponent-range)
            if [ -n "$2" ] && [ -n "$3" ]; then
                MIN_EXPONENT="$2"; MAX_EXPONENT="$3"; shift 3
            else echo "Error: --exponent-range requires two values."; exit 1; fi
            ;;
        -pes)
            if [ -n "$2" ]; then
                PE_SIZE="$2"
                if [[ "$GNPU_NUM" -gt "$PE_SIZE" ]]; then
                    GNPU_NUM="$PE_SIZE"
                fi
                shift 2
            else echo "Error: -pes requires a value."; exit 1; fi
            ;;
        -ipport)
            if [ -n "$2" ]; then IPPORT="$2"; shift 2; else echo "Error: -ipport requires a value."; exit 1; fi
            ;;
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                if [[ "$GNPU_NUM" -gt "$PE_SIZE" ]]; then GNPU_NUM="$PE_SIZE"; fi
                shift 2
            else echo "Error: -gnpus requires a value."; exit 1; fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then FIRST_NPU="$2"; shift 2; else echo "Error: -fnpu requires a value."; exit 1; fi
            ;;
        -a|--analyse)
            if [ -n "$2" ]; then ANALYSE_MODE="$2"; shift 2; else echo "Error: -a requires a value."; exit 1; fi
            ;;
        *)
            echo "Error: Unknown option $1."
            echo "使用方法: $0 [选项]"
            echo "  -t|--test-type <type>           put|bi_put|get|bi_get|all"
            echo "  -d|--datatype <type>            float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char|all"
            echo "  -b|--block-size <size>          RDMA 强制为 1，输入其他值会打印 WARN 后忽略"
            echo "  --block-range <min> <max>       RDMA 强制为 1，输入其他值会打印 WARN 后忽略"
            echo "  -e|--exponent <exponent>        数据量幂数"
            echo "  --exponent-range <min> <max>    数据量幂数范围"
            echo "  --loop-count <count>            循环次数 (默认 1000)"
            echo "  --ub-size <size>                UB size(KB), RDMA 需要至少 64B 的 UB 空间 (默认 64)"
            echo "  --batch <count>                 单 QP 上每次调用 quiet 之前连续提交的 NBI 个数 (默认 1)"
            echo "  --sync-id <id>                  显式传给 Put、Get、Quiet 的同步 ID (默认 0)"
            echo "  -q|--qp <num>                   QP 的个数，当前版本仅支持单 QP (默认 1)"
            echo "  -pes <size>                     PE 数量 (目前强制为 2)"
            echo "  -ipport <ip:port>               通信地址"
            echo "  -gnpus <num>                    NPU 数量"
            echo "  -fnpu <id>                      首个 NPU ID"
            echo "  -a|--analyse <none|plot|md>     分析模式"
            exit 1
            ;;
    esac
done

# 辅助函数：验证是否为非负整数
validate_non_negative_int() {
    local param_name="$1"
    local param_value="$2"
    if ! [[ "$param_value" =~ ^[0-9]+$ ]]; then
        echo "错误: $param_name 必须是非负整数 (got '$param_value')"
        exit 1
    fi
}

# 辅助函数：验证是否为正整数
validate_positive_int() {
    local param_name="$1"
    local param_value="$2"
    if ! [[ "$param_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误: $param_name 必须是正整数 (got '$param_value')"
        exit 1
    fi
}

# 辅助函数：验证 IPPORT 格式
validate_ipport() {
    local param_value="$1"
    if ! [[ "$param_value" =~ ^tcp://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$ ]]; then
        echo "错误: -ipport 格式无效，应为 'tcp://IP:PORT' 格式 (got '$param_value')"
        exit 1
    fi
}

# 验证测试类型
VALID_TEST_TYPES="put bi_put get bi_get all"
if [[ ! " $VALID_TEST_TYPES " =~ " $TEST_TYPE " ]]; then
    echo "错误: 测试类型必须是 'put' / 'bi_put' / 'get' / 'bi_get' 或 'all'"
    exit 1
fi

# 验证数据类型
VALID_DATATYPES="float int8 int16 int32 int64 uint8 uint16 uint32 uint64 char all"
if [[ ! " $VALID_DATATYPES " =~ " $DATA_TYPE " ]]; then
    echo "错误: 数据类型不在支持列表中"
    exit 1
fi

# 验证分析模式
VALID_ANALYSE_MODES="none plot md"
if [[ ! " $VALID_ANALYSE_MODES " =~ " $ANALYSE_MODE " ]]; then
    echo "错误: 分析模式必须是 'none' / 'plot' / 'md' (got '$ANALYSE_MODE')"
    exit 1
fi

# 验证数值参数
validate_non_negative_int "MIN_EXPONENT" "$MIN_EXPONENT"
validate_non_negative_int "MAX_EXPONENT" "$MAX_EXPONENT"
validate_positive_int "LOOP_COUNT" "$LOOP_COUNT"
validate_positive_int "UB_SIZE" "$UB_SIZE"
validate_non_negative_int "BATCH" "$BATCH"
validate_non_negative_int "SYNC_ID" "$SYNC_ID"
validate_positive_int "QP_NUM" "$QP_NUM"
validate_positive_int "PE_SIZE" "$PE_SIZE"
validate_positive_int "GNPU_NUM" "$GNPU_NUM"
validate_non_negative_int "FIRST_NPU" "$FIRST_NPU"

# 验证 IPPORT 格式
validate_ipport "$IPPORT"

# 验证幂数范围
if [[ "$MIN_EXPONENT" -gt "$MAX_EXPONENT" ]]; then
    echo "错误: MIN_EXPONENT ($MIN_EXPONENT) 不能大于 MAX_EXPONENT ($MAX_EXPONENT)"
    exit 1
fi

# 验证 QP_NUM
if [[ "$QP_NUM" != "1" ]]; then
    echo "错误: 当前版本仅支持单 QP，QP 数量必须为 1"
    exit 1
fi

# 验证并强制 PE_SIZE
if [[ "$PE_SIZE" != "2" ]]; then
    echo "WARN: RDMA perftest 目前强制 PE 数量为 2，忽略 -pes $PE_SIZE"
    PE_SIZE="2"
fi

if [[ "$UB_SIZE" -lt "1" ]]; then
    echo "错误: UB size 必须至少为 1KB (64B)"
    exit 1
fi

echo "测试类型: $TEST_TYPE"
echo "数据类型: $DATA_TYPE"
echo "幂数范围: $MIN_EXPONENT-$MAX_EXPONENT"
echo "循环次数: $LOOP_COUNT"
echo "UB size(KB): $UB_SIZE"
echo "Batch size: $BATCH"
echo "Sync ID: $SYNC_ID"
echo "QP num: $QP_NUM"
echo "PE_SIZE: $PE_SIZE, GNPU_NUM: $GNPU_NUM"
echo "FIRST_NPU: $FIRST_NPU"

export SHMEM_CYCLE_PROF_PE=${SHMEM_CYCLE_PROF_PE:-0}

ALL_TEST_TYPES=("put" "bi_put" "get" "bi_get")
ALL_DATATYPES=("float" "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64" "char")

run_test() {
    local test_type="$1"
    local data_type="$2"
    for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
        ${EXEC_BIN} --pes "$PE_SIZE" --pe-id "$idx" --ipport "$IPPORT" --gnpus "$GNPU_NUM" \
            --fnpu "$FIRST_NPU" -t "$test_type" -d "$data_type" \
            --exponent-range "$MIN_EXPONENT" "$MAX_EXPONENT" --loop-count "$LOOP_COUNT" \
            --ub-size "$UB_SIZE" --batch "$BATCH" --sync-id "$SYNC_ID" \
            --qp "$QP_NUM" &
    done
    wait
}

if [[ "$TEST_TYPE" == "all" && "$DATA_TYPE" == "all" ]]; then
    for type in "${ALL_TEST_TYPES[@]}"; do
        for dtype in "${ALL_DATATYPES[@]}"; do
            echo -e "\n=== 运行测试类型: $type, 数据类型: $dtype ==="
            run_test "$type" "$dtype"
        done
    done
elif [[ "$TEST_TYPE" == "all" ]]; then
    for type in "${ALL_TEST_TYPES[@]}"; do
        echo -e "\n=== 运行测试类型: $type, 数据类型: $DATA_TYPE ==="
        run_test "$type" "$DATA_TYPE"
    done
elif [[ "$DATA_TYPE" == "all" ]]; then
    for dtype in "${ALL_DATATYPES[@]}"; do
        echo -e "\n=== 运行测试类型: $TEST_TYPE, 数据类型: $dtype ==="
        run_test "$TEST_TYPE" "$dtype"
    done
else
    run_test "$TEST_TYPE" "$DATA_TYPE"
fi

cd ${CURRENT_DIR}

PERF_SCRIPT="${SCRIPT_DIR}/../../utils/perf_data_process.py"
if [ -f "${PERF_SCRIPT}" ]; then
    if [ "$ANALYSE_MODE" = "plot" ] || [ "$ANALYSE_MODE" = "md" ]; then
        echo -e "\n========== Generating performance charts =========="
        cmd_args=("-d" "${SCRIPT_DIR}/output")
        if [ "$ANALYSE_MODE" = "plot" ]; then
            cmd_args+=("--no-markdown")
        fi
        python3 "${PERF_SCRIPT}" "${cmd_args[@]}"
    fi
fi
