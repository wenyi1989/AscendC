
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ThreeNNTilingData)
      TILING_DATA_FIELD_DEF(uint32_t, bDim);
      TILING_DATA_FIELD_DEF(uint32_t, inputNum1);
      TILING_DATA_FIELD_DEF(uint32_t, inputNum2);
      TILING_DATA_FIELD_DEF(uint32_t, n);
      TILING_DATA_FIELD_DEF(uint32_t, m);
      TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ThreeNN, ThreeNNTilingData)
}
