# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
import numpy as np
import torch


class CommType(IntEnum):
    MATMUL_ALLREDUCE = 0
    ALLGATHER_MATMUL = 1
    MATMUL_REDUCE_SCATTER = 2
    MATMUL_REDUCE_SCATTER_PADDING = 3
    ALLGATHER_MATMUL_WITH_GATHER_RESULT = 4
    ALLGATHER_MATMUL_PADDING = 5

    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))


class DataType(IntEnum):
    FLOAT = 0
    FLOAT16 = 1
    BF16 = 27

    @property
    def torch_type(self):
        return {
            DataType.FLOAT: torch.float,
            DataType.FLOAT16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }[self]
    
    @classmethod
    def from_str(cls, arg: str):
        return cls(int(arg))


def tensor_to_file(tensor: torch.Tensor, file_name: str) -> None:
    """
    Save tensor to binary file.
    
    Note: For bfloat16 tensors, we convert to float32 before saving to disk.
    This results in 4 bytes per element (vs 2 bytes for raw bf16), but ensures
    compatibility with numpy's tofile() which doesn't support bf16 natively.
    File size will be 2x larger for bf16 data compared to fp16.
    """
    if tensor.dtype == torch.bfloat16:
        tensor.to(torch.float32).numpy().tofile(file_name)
    else:
        tensor.numpy().tofile(file_name)


def tensor_from_file(file_name: str, dtype: torch.dtype, is_fp32_format: bool = False) -> torch.Tensor:
    """
    Load tensor from binary file.
    
    Args:
        file_name: Path to the binary file
        dtype: Target torch dtype
        is_fp32_format: For bfloat16 tensors, indicates the file format:
            - False (default): Raw BF16 format (2 bytes/element), used for SHMEM/C++ output
            - True: Float32 format (4 bytes/element), used for torch_npu reference or CPU golden
    
    Note: For bfloat16 tensors, the file format depends on the source:
        - SHMEM/C++ output: Raw BF16 (2 bytes/element), read as uint16 and view as bf16
        - torch_npu/CPU golden: Float32 format (4 bytes/element), read as float32 and convert to bf16
    """
    if dtype == torch.bfloat16:
        if is_fp32_format:
            # Float32 format: used for torch_npu reference or CPU golden
            return torch.from_numpy(np.fromfile(file_name, dtype=np.float32)).to(torch.bfloat16)
        else:
            # Raw BF16 format: used for SHMEM/C++ output
            return torch.from_numpy(np.fromfile(file_name, dtype=np.uint16)).view(torch.bfloat16)
    else:
        numpy_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        return torch.from_numpy(np.fromfile(file_name, numpy_dtype))


def get_rtol(dtype: torch.dtype, compute_times: int) -> float:
    if dtype == torch.float16:
        return 2 ** (-8) if compute_times < 2048 else 2 ** (-7)
    elif dtype == torch.bfloat16:
        return 2 ** (-7) if compute_times < 2048 else 2 ** (-6)
    elif dtype == torch.float32:
        return 2 ** (-11) if compute_times < 2048 else 2 ** (-10)
    else:
        raise ValueError(f"Invalid dtype: {dtype}.")