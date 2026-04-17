#ifndef LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H
#define LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H

#include "../op_host/linearize_cache_indices_from_row_idx_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 7680;
constexpr int32_t MAX_CUMSUM_UB_ELEMS = 1024;

struct LinearizeArgs {
    GM_ADDR cache_hash_size_cumsum;
    GM_ADDR update_table_indices;
    GM_ADDR update_row_indices;
    GM_ADDR linear_cache_indices;
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
    }

    __aicore__ inline void Compute()
    {
        T sentinel = cumsumGT.GetValue(static_cast<uint32_t>(cumsumLength - 1));

        for (int64_t i = 0; i < blockLen; ++i) {

            T tableId = tableIdxGT.GetValue(static_cast<uint32_t>(i));
            T rowId = rowIdxGT.GetValue(static_cast<uint32_t>(i));
            T offset = cumsumGT.GetValue(static_cast<uint32_t>(tableId));

            T result;
            if (offset >= 0 && rowId >= 0) {
                result = offset + rowId;
            } else {
                result = sentinel;
            }

            outputGT.SetValue(static_cast<uint32_t>(i), result);
        }
    }

private:
    __aicore__ inline void InitTilingParams(
        const LinearizeCacheIndicesFromRowIdxTilingData& tilingData)
    {
        totalLength = tilingData.totalLength;
        cumsumLength = tilingData.cumsumLength;
        blockSize = tilingData.blockSize;
        numBlocks = tilingData.numBlocks;
    }

    __aicore__ inline void InitGmParams(LinearizeArgs& args)
    {
    //     if (GetBlockIdx() < tailBlockNum) {
    //     totalBlockNum=basicBlockNum+1;
    //     }
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t start = blockIdx * static_cast<int64_t>(blockSize);
        blockLen = (start + blockSize <= totalLength) ? blockSize : (totalLength - start);
        printf("blockLen %d,  start %d \n", blockLen, start);
        cumsumGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.cache_hash_size_cumsum), cumsumLength);
        tableIdxGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.update_table_indices) + start, blockLen);
        rowIdxGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.update_row_indices) + start, blockLen);
        outputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args.linear_cache_indices) + start, blockLen);
    }

private:
    GlobalTensor<T> cumsumGT;
    GlobalTensor<T> tableIdxGT;
    GlobalTensor<T> rowIdxGT;
    GlobalTensor<T> outputGT;

    int64_t totalLength;
    int64_t cumsumLength;
    int32_t blockSize;
    int32_t numBlocks;
    int64_t blockLen;
};

} // namespace LinearizeCacheIndicesFromRowIdx

#endif


