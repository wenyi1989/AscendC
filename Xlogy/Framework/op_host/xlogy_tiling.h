
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(XlogyTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, CoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, TailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, x1_length);
  TILING_DATA_FIELD_DEF(uint32_t, x2_length);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 128, shape);
  TILING_DATA_FIELD_DEF(int64_t, numshapes);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 64, shapefull);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Xlogy, XlogyTilingData)
}
