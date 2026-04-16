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

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "linearize_cache_indices_from_row_idx_tiling.h"

namespace {
    constexpr int32_t MAX_THREADS_PER_BLOCK = 1024;
    constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
    constexpr int32_t MAX_WARPS = MAX_THREADS_PER_BLOCK / 32;
    constexpr int DCACHE_SIZE = 128 * 1024;
    constexpr int32_t MULTIPLIER = 2;
    constexpr int32_t DIVISOR = 4;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("cache_hash_size_cumsumShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("update_table_indicesShape", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("update_row_indicesTensor", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    
    //输:0：cache_hash_size_cumsum,形状【T+1】 
    //输入1：update_table_indices,形状【N】 
    //输入2：update_row_indices，形状【N】
    int64_t totalLength = context->GetInputShape(2)->GetOriginShape().GetShapeSize();
    int64_t cumsumLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    uint32_t dimNumRow = context->GetInputShape(2)->GetOriginShape().GetDimNum();
    OPS_LOG_E_IF(dimNumRow != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]LinearizeCacheIndicesFromRowIdx: update_row_indices must be 1-D");

    ge::DataType inputDataType = context->GetInputTensor(2)->GetDataType();
    OPS_CHECK(inputDataType != ge::DT_INT32 && inputDataType != ge::DT_INT64,
              OPS_LOG_E("[ERROR]Invalid data type",
                        "LinearizeCacheIndicesFromRowIdx only supports int64 and int32."),
              return ge::GRAPH_FAILED);

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    workspaceSize[0] = ascendPlatform.GetLibApiWorkSpaceSize();

    
    
    int32_t numBlocks = static_cast<int32_t>((totalLength + coresize - 1) / coresize);
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    LinearizeCacheIndicesFromRowIdxTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_cumsumLength(cumsumLength);
    tiling.set_blockSize(coresize);
    tiling.set_numBlocks(numBlocks);

    context->SetBlockDim(static_cast<uint32_t>(numBlocks));
    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* rowIdxShape = context->GetInputShape(2);
    OPS_LOG_E_IF_NULL("update_row_indices shape", rowIdxShape, return ge::GRAPH_FAILED);

    gert::Shape* outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("output shape", outShape, return ge::GRAPH_FAILED);

    *outShape = *rowIdxShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(2);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, inputDataType)) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

}  

namespace ops {

class LinearizeCacheIndicesFromRowIdx : public OpDef {
public:
    explicit LinearizeCacheIndicesFromRowIdx(const char* name) : OpDef(name)
    {
        this->Input("cache_hash_size_cumsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("update_table_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("update_row_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("linear_cache_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(LinearizeCacheIndicesFromRowIdx);

} 
