/*
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LerpTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lerp, LerpTilingData)
}
*/
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(uint32_t, total_length);
  TILING_DATA_FIELD_DEF(uint32_t, start_length);
  TILING_DATA_FIELD_DEF(uint32_t, end_length);
  TILING_DATA_FIELD_DEF(uint32_t, weight_length);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 192, shape);
  TILING_DATA_FIELD_DEF(int64_t, numshapes);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 64, shapefull);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lerp, TilingData)
}
