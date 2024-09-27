
#include "three_nn_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>


namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ThreeNNTilingData tiling;

    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); //获取硬件平台存储空间 UB 的内存大小

    //获取输入shape信息
    uint32_t inputNum1 = context->GetInputShape(0)->GetStorageShape().GetShapeSize(); //输入数量
    uint32_t inputBytes = GetSizeByDataType(context->GetInputDesc(0)->GetDataType()); //输入类型
    uint32_t inputLength = inputBytes * inputNum1; //输入长度
    uint32_t inputNum2 = context->GetInputShape(1)->GetStorageShape().GetShapeSize();

    uint32_t b = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    uint32_t n = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    uint32_t m = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    tiling.set_bDim(b);
    tiling.set_inputNum1(inputNum1);
    tiling.set_inputNum2(inputNum2);
    tiling.set_n(n);
    tiling.set_m(m);

    uint32_t ubDataNum = 20;
    uint32_t ubBlockNum = ubSize / BLOCK_SIZE / ubDataNum;
    uint32_t tileLength = ubBlockNum * BLOCK_SIZE / 48 * 48;

    tiling.set_tileLength(tileLength);

    std::cout << "ubSize:" << ubSize << std::endl;
    std::cout << "inputNum1:" << inputNum1 << std::endl;
    std::cout << "inputNum2:" << inputNum2 << std::endl;
    std::cout << "n:" << n << std::endl;
    std::cout << "m:" << m << std::endl;
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
class ThreeNN : public OpDef {
public:
    explicit ThreeNN(const char* name) : OpDef(name)
    {
        this->Input("xyz1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("xyz2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ThreeNN);
}
