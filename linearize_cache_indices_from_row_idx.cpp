/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "linearize_cache_indices_from_row_idx_kernel.h"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 入口函数（Scalar 版本）
//
//   对齐 CUDA LinearizeCacheIndicesScalarKernel 的参数列表与线程模型：
//     · cache_hash_size_cumsum [T+1]  各 table 在 cache 中的偏移前缀和
//     · update_table_indices   [K]    每条记录所属的 table id
//     · update_row_indices     [K]    每条记录在 table 内的本地行号
//     · linear_cache_indices   [K]    输出：全局 cache 地址
//     · workspace                     框架分配的系统 workspace
//     · tiling                        tiling 参数（含 blockSize / numBlocks / totalLength）
//
//   线程映射：
//     CUDA blockIdx.x  → Ascend C GetBlockIdx()       （AI Core 编号）
//     CUDA blockDim.x  → tiling.blockSize = BLOCK_DIM （每核 SIMT 线程数）
//     CUDA threadIdx.x → Ascend C GetSubBlockIdx()    （核内线程编号）
//     全局索引 idx = GetBlockIdx() * blockSize + GetSubBlockIdx()
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__ __aicore__ void linearize_cache_indices_from_row_idx(
    GM_ADDR cache_hash_size_cumsum,
    GM_ADDR update_table_indices,
    GM_ADDR update_row_indices,
    GM_ADDR linear_cache_indices,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    LinearizeArgs args{
        cache_hash_size_cumsum,
        update_table_indices,
        update_row_indices,
        linear_cache_indices,
        workspace,
        tiling
    };

    // DTYPE_UPDATE_ROW_INDICES 由编译框架根据算子注册的数据类型自动注入
    LinearizeCacheIndicesFromRowIdx::LinearizeCacheIndicesScalarKernel<DTYPE_UPDATE_ROW_INDICES> kernel(args);
    kernel.Compute();
}
