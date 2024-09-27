
#include "global_avg_pool_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    GlobalAvgPoolTilingData tiling;

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小

    //获取输入shape信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    uint32_t batchsize = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        if(i > 1) {
            batchsize *= x1_shape->GetStorageShape().GetDim(i);
        }    
    }    
    tiling.set_inputNum(inputNum);
    tiling.set_batchsize(batchsize);

    uint32_t ubDataNum = 16;
    uint32_t ubBlockNum = ubSize / BLOCK_SIZE / ubDataNum;
    uint32_t tileLength = ubBlockNum * BLOCK_SIZE;
    tiling.set_tileLength(tileLength);

    std::cout << "ubSize:" << ubSize << std::endl;
    std::cout << "inputNum:" << inputNum << std::endl;
    std::cout << "batchsize:" << batchsize << std::endl;
    std::cout << "tileLength:" << tileLength << std::endl;

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class GlobalAvgPool : public OpDef {
public:
    explicit GlobalAvgPool(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GlobalAvgPool);
}
