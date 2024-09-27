
#include "tril_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  TrilTilingData tiling;
  /*
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  */
    //constexpr int32_t NUM = 8;
    int32_t NUM = 8;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 9;
    }
    else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
    }
    else{
        sizeofdatatype = 4;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = totalLength - aivNum * core_size;

    auto ptr = context->GetAttrs()->GetInt(0);
    tiling.set_diagonal(*ptr);
    std::cout << "diagonal:" << *ptr <<std::endl;
    std::cout << "block_size:" << block_size <<std::endl;
    std::cout << "core_size:" << core_size <<std::endl;

    uint32_t dimnum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    //std::cout << "this is triu flag2-" << dimnum <<std::endl;
    uint32_t dimnuma = context->GetInputShape(0)->GetStorageShape().GetDim(dimnum - 2);
    uint32_t dimnumb = context->GetInputShape(0)->GetStorageShape().GetDim(dimnum - 1);
    std::cout << "dimnuma:" << dimnuma <<std::endl;
    std::cout << "dimnumb:" << dimnumb <<std::endl;
    uint32_t dimnumbalign = dimnumb + (dimnumb % ALIGN_NUM ? ALIGN_NUM - dimnumb % ALIGN_NUM : 0);
    std::cout << "dimnumbalign:" << dimnumbalign <<std::endl;
    tiling.set_dimnuma(dimnuma);
    tiling.set_dimnumb(dimnumb);	
    tiling.set_dimnumbalign(dimnumbalign);
    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    //context->SetBlockDim(aivNum);
    context->SetBlockDim(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
class Tril : public OpDef {
public:
    explicit Tril(const char* name) : OpDef(name)
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
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Tril);
}
