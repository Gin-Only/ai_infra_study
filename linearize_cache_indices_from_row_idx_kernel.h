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

#ifndef LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H
#define LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H

#include "kernel_operator.h"
#include "linearize_cache_indices_from_row_idx_tiling.h"

using namespace AscendC;

// ─────────────────────────────────────────────────────────────────────────────
// Scalar 版本：对齐 CUDA LinearizeCacheIndicesScalarKernel 的线程模型
//
//   CUDA 对应逻辑：
//     idx = blockIdx.x * blockDim.x + threadIdx.x
//     if idx >= total_length: return
//     table_id  = update_table_indices[idx]
//     row_id    = update_row_indices[idx]
//     offset    = cache_hash_size_cumsum[table_id]
//     linear_cache_indices[idx] = offset + row_id
//
//   Ascend C 映射：
//     blockIdx.x  → GetBlockIdx()          （AI Core 编号）
//     blockDim.x  → blockSize（tiling 参数，固定 BLOCK_DIM）
//     threadIdx.x → GetSubBlockIdx()       （Core 内 SIMT 线程编号）
//
//   UB 用途：
//     cumsumBuf   — 将 cache_hash_size_cumsum[T+1] 整体缓存到 UB，
//                   避免 scalar 循环中对 GM 做随机散点读，
//                   对应 CUDA 的 __ldg() L1 只读广播优化。
// ─────────────────────────────────────────────────────────────────────────────

// 每个 AI Core 内启动的 SIMT 线程数，对应 CUDA blockDim.x
constexpr int32_t BLOCK_DIM = 32;

// UB 中 cumsum 缓冲的最大容量（元素个数）
// T（embedding table 数量）在实际场景远小于此值
constexpr int32_t MAX_CUMSUM_UB_ELEMS = 512;

struct LinearizeArgs {
    GM_ADDR cache_hash_size_cumsum;  // [T+1]
    GM_ADDR update_table_indices;    // [K]
    GM_ADDR update_row_indices;      // [K]
    GM_ADDR linear_cache_indices;    // [K] 输出
    GM_ADDR workspace;
    GM_ADDR tiling;
};

namespace LinearizeCacheIndicesFromRowIdx {

template <typename T>
class LinearizeCacheIndicesScalarKernel {
public:
    __aicore__ inline LinearizeCacheIndicesScalarKernel(LinearizeArgs& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        InitTilingParams(tilingData);
        InitGmParams(args);
        InitUbParams();
    }

    // ─── Compute：对齐 CUDA kernel 的 idx = blockIdx * blockDim + threadIdx ──
    __aicore__ inline void Compute()
    {
        // 将 cache_hash_size_cumsum 整体预取到 UB
        // 等价于 CUDA 中通过 __ldg() 利用 L1 只读缓存广播给 warp 内所有线程
        LocalTensor<T> cumsumLt = cumsumBuf.Get<T>();
        uint32_t copyLen = static_cast<uint32_t>(cumsumLength <= MAX_CUMSUM_UB_ELEMS
                                                     ? cumsumLength
                                                     : MAX_CUMSUM_UB_ELEMS);
        DataCopy(cumsumLt, cumsumGT[0], copyLen);

        // blockIdx.x → GetBlockIdx()，threadIdx.x → GetSubBlockIdx()
        int64_t blockIdx  = static_cast<int64_t>(GetBlockIdx());
        int64_t threadIdx = static_cast<int64_t>(GetSubBlockIdx());

        // 全局线程索引：idx = blockIdx * blockDim + threadIdx
        int64_t idx = blockIdx * static_cast<int64_t>(blockSize) + threadIdx;

        // 边界判断（对应 CUDA：if (idx >= total_length) return;）
        if (idx >= totalLength) {
            return;
        }

        // ── 标量读取：table_id 和 row_id ──────────────────────────────────────
        // 从 GM 单点读取当前线程负责的元素
        // （每个线程只处理 1 个元素，与 CUDA scalar kernel 一一对应）
        LocalTensor<T> scalarBuf = scalarTmpBuf.Get<T>();

        DataCopy(scalarBuf, tableIdxGT[idx], 1);
        T tableId = scalarBuf.GetValue(0);

        DataCopy(scalarBuf, rowIdxGT[idx], 1);
        T rowId = scalarBuf.GetValue(0);

        // ── 查表：从 UB 中读取 cache_hash_size_cumsum[table_id] ──────────────
        // UB 命中时直接 GetValue，无需再访问 GM（对应 __ldg() 的 L1 命中路径）
        T offset;
        if (tableId < static_cast<T>(MAX_CUMSUM_UB_ELEMS)) {
            offset = cumsumLt.GetValue(static_cast<uint32_t>(tableId));
        } else {
            // UB 未覆盖的极端情况：回退到 GM 读取
            DataCopy(scalarBuf, cumsumGT[tableId], 1);
            offset = scalarBuf.GetValue(0);
        }

        // ── 计算结果并写回 GM ─────────────────────────────────────────────────
        // 对应 CUDA：linear_cache_indices[idx] = offset + row_id
        T result = offset + rowId;

        scalarBuf.SetValue(0, result);
        DataCopy(outputGT[idx], scalarBuf, 1);
    }

private:
    __aicore__ inline void InitTilingParams(
        const LinearizeCacheIndicesFromRowIdxTilingData& tilingData)
    {
        totalLength  = tilingData.totalLength;
        cumsumLength = tilingData.cumsumLength;
        blockSize    = tilingData.blockSize;
        numBlocks    = tilingData.numBlocks;
    }

    __aicore__ inline void InitGmParams(LinearizeArgs& args)
    {
        cumsumGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.cache_hash_size_cumsum), cumsumLength);
        tableIdxGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.update_table_indices), totalLength);
        rowIdxGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.update_row_indices), totalLength);
        outputGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.linear_cache_indices), totalLength);
    }

    __aicore__ inline void InitUbParams()
    {
        // cumsumBuf：容纳最多 MAX_CUMSUM_UB_ELEMS 个 T 类型元素
        pipe.InitBuffer(cumsumBuf,    MAX_CUMSUM_UB_ELEMS * sizeof(T));
        // scalarTmpBuf：单元素临时读写缓冲（每次 DataCopy 1 个元素）
        // 大小取 32B 以满足 Ascend C DataCopy 的最小对齐要求
        pipe.InitBuffer(scalarTmpBuf, 32);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> cumsumBuf;    // UB：缓存整个 cumsum 数组
    TBuf<TPosition::VECCALC> scalarTmpBuf; // UB：单元素标量读写中转

    GlobalTensor<T> cumsumGT;
    GlobalTensor<T> tableIdxGT;
    GlobalTensor<T> rowIdxGT;
    GlobalTensor<T> outputGT;

    int64_t totalLength;
    int64_t cumsumLength;
    int32_t blockSize;
    int32_t numBlocks;
};

}  // namespace LinearizeCacheIndicesFromRowIdx

#endif  // LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H
