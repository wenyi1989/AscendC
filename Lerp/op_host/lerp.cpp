
#include "lerp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <iostream>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  /*		
  LerpTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
  */
    TilingData tiling;
    int32_t NUM = 10; //申请的内存块的个数（包括buffer和queue）
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size; ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    uint32_t total_length = 0, min_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    for (int i = 0; i < 3; ++i) {
        total_length = std::max<uint32_t>(total_length, context->GetInputShape(i)->GetStorageShape().GetShapeSize());
        min_length = std::min<uint32_t>(min_length, context->GetInputShape(i)->GetStorageShape().GetShapeSize());
    }
    uint32_t start_length = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t end_length = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    uint32_t weight_length = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t sizeofdatatype;
    if (dt == ge::DT_INT8) {
        sizeofdatatype = 1;
        NUM = 12; //不同的数据类型算法不一样，所以会设成不同的值
    }
    else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        sizeofdatatype = 2;
    }
    else {
        sizeofdatatype = 4;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    // if(weight_length != 1) {
    //     if (total_length != min_length) {
    //         block_size = std::min(block_size, min_length);
    //         //while (min_length % block_size || min_length % ALIGN_NUM) {
    //         while (block_size % ALIGN_NUM) {
    //             block_size -= 1;
    //         }
    //         if(block_size == 0) {
    //             block_size = ALIGN_NUM;
    //         }
    //     }
    // }

    aivNum = (aivNum < total_length / block_size) ? aivNum : (total_length / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t core_size = (total_length / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = total_length - aivNum * core_size;

    int64_t numshapes = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    tiling.set_numshapes(numshapes);
    int64_t shape[192];
    for (int k = 0; k < 3; ++k) {
        int64_t *ss = &shape[k * 64];
        const gert::StorageShape* shape = context->GetInputShape(k);
        for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            ss[i] = shape->GetStorageShape().GetDim(i);
        }
    }
    tiling.set_shape(shape);

    int64_t shapefull[64];
    for (int k = 0; k < numshapes; ++k) {
        int64_t *ss = &shape[0];
        int64_t *sf = &shapefull[0];
        int64_t tmp = (ss[k] > ss[k + 64]) ? ss[k] : ss[k + 64];
        const gert::StorageShape* shape = context->GetInputShape(2);
        if(shape->GetStorageShape().GetDimNum() == numshapes) {
            tmp = (tmp > ss[k + 128]) ? tmp : ss[k + 128];
        }    
        sf[k] = tmp;
    }
    tiling.set_shapefull(shapefull);

    std::cout<< "block_size:" << block_size << std::endl;
    std::cout<< "total_length:" << total_length << std::endl;
    std::cout<< "weight_length:" << weight_length << std::endl;
    std::cout<< "start_length:" << start_length << std::endl;
    std::cout<< "end_length:" << end_length << std::endl;
    std::cout<< "min_length:" << min_length << std::endl;
    std::cout<< "numshapes:" << numshapes << std::endl;

    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    tiling.set_total_length(total_length);
    tiling.set_weight_length(weight_length);
    tiling.set_start_length(start_length);
    tiling.set_end_length(end_length);

    context->SetBlockDim(aivNum);

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
class Lerp : public OpDef {
public:
    explicit Lerp(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
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

OP_ADD(Lerp);
}
