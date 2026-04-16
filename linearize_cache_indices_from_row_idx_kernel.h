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

#include "../op_host/linearize_cache_indices_from_row_idx_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

// 每个 AI Core 一次处理的元素数（240KB / 4数组 / 8B = 7680）
constexpr int32_t BLOCK_SIZE = 7680;

// UB 中 cumsum 缓冲的最大容量（元素个数，对应最多 1024 张 embedding table）
constexpr int32_t MAX_CUMSUM_UB_ELEMS = 1024;

struct LinearizeArgs {
    GM_ADDR cache_hash_size_cumsum; // [T+1]
    GM_ADDR update_table_indices; // [K]
    GM_ADDR update_row_indices; // [K]
    GM_ADDR linear_cache_indices; // [K] 输出
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

    
    __aicore__ inline void Compute()
    {
        
        LocalTensor<T> cumsumLt = cumsumBuf.Get<T>();
        uint32_t cumsumActual = static_cast<uint32_t>(cumsumLength <= MAX_CUMSUM_UB_ELEMS
                                                          ? cumsumLength
                                                          : MAX_CUMSUM_UB_ELEMS);
        uint32_t cumsumCopyLen = AlignTo32(cumsumActual * sizeof(T)) / sizeof(T);
        for (int64_t i = 0; i < blockLen; i++) {
             T tableId = tableGM.GetValue(static_cast<uint32_t>(i));
             T rowId = rowGM.GetValue(static_cast<uint32_t>(i));
             offset = cumsumGM.GetValue(static_cast<uint32_t>(tableId));
             outputGT.setvalue(i, offset+rowId);
        }
        // DataCopy(cumsumLt, cumsumGT[0], cumsumCopyLen);

        
        // LocalTensor<T> tableLt = tableBuf.Get<T>();
        // LocalTensor<T> rowLt = rowBuf.Get<T>();
        // LocalTensor<T> outLt = outBuf.Get<T>();

        
        // uint32_t alignedLen = AlignTo32(static_cast<uint32_t>(blockLen) * sizeof(T)) / sizeof(T);

        // DataCopy(tableLt, tableIdxGT[0], alignedLen);
        // DataCopy(rowLt, rowIdxGT[0], alignedLen);

        // for (int64_t i = 0; i < blockLen; i++) {
        //     T tableId = tableLt.GetValue(static_cast<uint32_t>(i));
        //     T rowId = rowLt.GetValue(static_cast<uint32_t>(i));

        //     T offset;
        //     if (tableId >= 0 && tableId < static_cast<T>(MAX_CUMSUM_UB_ELEMS)) {
        //         // UB 命中：直接查表
        //         offset = cumsumLt.GetValue(static_cast<uint32_t>(tableId));
        //     } else {
        //         // UB 未覆盖（极端情况）：回退到 GM
        //         // 注意：此处单元素 DataCopy 需借助 scalarTmpBuf 满足 32B 对齐
        //         LocalTensor<T> tmpLt = scalarTmpBuf.Get<T>();
        //         uint32_t tmpLen = AlignTo32(sizeof(T)) / sizeof(T);
        //         DataCopy(tmpLt, cumsumGT[tableId], tmpLen);
        //         offset = tmpLt.GetValue(0);
        //     }

        //     outLt.SetValue(static_cast<uint32_t>(i), offset + rowId);
        // }

        // DataCopy(outputGT[0], outLt, alignedLen);
    }

private:
    __aicore__ inline void InitTilingParams(
        const LinearizeCacheIndicesFromRowIdxTilingData& tilingData)
    {
        totalLength = tilingData.totalLength;
        cumsumLength = tilingData.cumsumLength;
        blockSize = tilingData.blockSize; // = 7680
        numBlocks = tilingData.numBlocks;
    }

    __aicore__ inline void InitGmParams(LinearizeArgs& args)
    {
        
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t start = blockIdx * static_cast<int64_t>(blockSize);
        blockLen = (start + blockSize <= totalLength)
                               ? blockSize
                               : (totalLength - start);


        cumsumGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.cache_hash_size_cumsum),
            cumsumLength);

        
        tableIdxGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.update_table_indices) + start,
            blockLen);
        rowIdxGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.update_row_indices) + start,
            blockLen);
        outputGT.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(args.linear_cache_indices) + start,
            blockLen);
    }

    __aicore__ inline void InitUbParams()
    {
        
        pipe.InitBuffer(cumsumBuf, MAX_CUMSUM_UB_ELEMS * sizeof(T));
        pipe.InitBuffer(tableBuf, BLOCK_SIZE * sizeof(T));
        pipe.InitBuffer(rowBuf, BLOCK_SIZE * sizeof(T));
        pipe.InitBuffer(outBuf, BLOCK_SIZE * sizeof(T));
        // scalarTmpBuf：32B，用于 UB 未覆盖时单元素 GM 回退读取
        pipe.InitBuffer(scalarTmpBuf, 32);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> cumsumBuf; 
    TBuf<TPosition::VECCALC> tableBuf; 
    TBuf<TPosition::VECCALC> rowBuf; 
    TBuf<TPosition::VECCALC> outBuf; 
    TBuf<TPosition::VECCALC> scalarTmpBuf; 

    GlobalTensor<T> cumsumGT;
    GlobalTensor<T> tableIdxGT;
    GlobalTensor<T> rowIdxGT;
    GlobalTensor<T> outputGT;

    int64_t totalLength;
    int64_t cumsumLength;
    int32_t blockSize;
    int32_t numBlocks;
    int64_t blockLen; // 当前核实际处理的元素数（末尾核可能 < blockSize）
};

} // namespace LinearizeCacheIndicesFromRowIdx

#endif // LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H

