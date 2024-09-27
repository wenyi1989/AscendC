
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GlobalAvgPoolTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchsize);
    TILING_DATA_FIELD_DEF(uint32_t, inputNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GlobalAvgPool, GlobalAvgPoolTilingData)
}
