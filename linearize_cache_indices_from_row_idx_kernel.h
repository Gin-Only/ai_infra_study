#ifndef LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H
#define LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H

#include "../op_host/linearize_cache_indices_from_row_idx_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 7680;

// 不再使用 UB 缓存 cumsum，因此该常量仅保留兼容 tiling
constexpr int32_t MAX_CUMSUM_UB_ELEMS = 1024;

struct LinearizeArgs {
    GM_ADDR cache_hash_size_cumsum; // [T+1]
    GM_ADDR update_table_indices;   // [K]
    GM_ADDR update_row_indices;     // [K]
    GM_ADDR linear_cache_indices;   // [K] 输出
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
        // 完全在 GM 上标量读取计算，不使用任何 UB 缓存
        for (int64_t i = 0; i < blockLen; ++i) {
            T tableId = tableIdxGT.GetValue(static_cast<uint32_t>(i));
            T rowId   = rowIdxGT.GetValue(static_cast<uint32_t>(i));

            // 直接从 GM 读取 cumsum 偏移
            T offset  = cumsumGT.GetValue(static_cast<uint32_t>(tableId));

            // 计算线性索引并写回 GM
            outputGT.SetValue(static_cast<uint32_t>(i), offset + rowId);
        }
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
        int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        int64_t start    = blockIdx * static_cast<int64_t>(blockSize);
        blockLen         = (start + blockSize <= totalLength) ? blockSize : (totalLength - start);

        // 全局内存张量初始化
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

private:
    // 已删除所有 TBuf<UB>、pipe、cumsumBuf 等成员

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

#endif // LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_KERNEL_H
