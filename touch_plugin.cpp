/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

at::Tensor linearize_cache_indices_from_row_idx_npu(
    const at::Tensor& cache_hash_size_cumsum,
    const at::Tensor& update_table_indices,
    const at::Tensor& update_row_indices)
{
    const at::OptionalDeviceGuard guard(device_of(update_row_indices));

    // 检查所有输入张量均在 NPU 设备上
    std::vector<at::Tensor> tensors = {
        cache_hash_size_cumsum,
        update_table_indices,
        update_row_indices
    };
    std::vector<std::string> names = {
        "cache_hash_size_cumsum",
        "update_table_indices",
        "update_row_indices"
    };
    check_tensor_npu_device(tensors, names);

    // 输入校验：update_row_indices 必须是 1D
    int64_t total_length = update_row_indices.size(0);
    TORCH_CHECK(update_row_indices.dim() == 1,
                "update_row_indices must be 1-D, but got dim=",
                update_row_indices.dim(), "\n");
    TORCH_CHECK(update_table_indices.dim() == 1,
                "update_table_indices must be 1-D, but got dim=",
                update_table_indices.dim(), "\n");
    TORCH_CHECK(update_table_indices.size(0) == total_length,
                "update_table_indices and update_row_indices must have the same length, ",
                "got ", update_table_indices.size(0), " vs ", total_length, "\n");

    // 空输入快速返回
    if (total_length == 0) {
        return at::empty_like(update_row_indices);
    }

    // 保证内存连续（DataCopy 要求连续内存）
    auto cache_hash_size_cumsum_contin = cache_hash_size_cumsum.contiguous();
    auto update_table_indices_contin   = update_table_indices.contiguous();
    auto update_row_indices_contin     = update_row_indices.contiguous();

    // 输出张量：shape [K]，类型与 update_row_indices 一致
    auto output = at::empty({total_length}, update_row_indices.options());

    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnLinearizeCacheIndicesFromRowIdx,
                 cache_hash_size_cumsum_contin,
                 update_table_indices_contin,
                 update_row_indices_contin,
                 output);

    return output;
}
//算子注册
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("linearize_cache_indices_from_row_idx("
          "Tensor cache_hash_size_cumsum, "
          "Tensor update_table_indices, "
          "Tensor update_row_indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("linearize_cache_indices_from_row_idx",
           &linearize_cache_indices_from_row_idx_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("linearize_cache_indices_from_row_idx",
           &linearize_cache_indices_from_row_idx_npu);
}
