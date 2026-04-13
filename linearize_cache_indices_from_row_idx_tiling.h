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

#ifndef LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_TILING_H
#define LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(LinearizeCacheIndicesFromRowIdxTilingData)
    // 更新记录的总数量 K（对应 CUDA total_length）
    TILING_DATA_FIELD_DEF(int64_t, totalLength);
    // cache_hash_size_cumsum 的长度（T+1，最后一项为 sentinel max_offset）
    TILING_DATA_FIELD_DEF(int64_t, cumsumLength);
    // 每个 Block（AI Core）内的线程数（对应 CUDA blockDim.x）
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    // Block 总数（对应 CUDA gridDim.x = ceil(K / blockSize)）
    TILING_DATA_FIELD_DEF(int32_t, numBlocks);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(LinearizeCacheIndicesFromRowIdx, LinearizeCacheIndicesFromRowIdxTilingData)
}
#endif // LINEARIZE_CACHE_INDICES_FROM_ROW_IDX_TILING_H
