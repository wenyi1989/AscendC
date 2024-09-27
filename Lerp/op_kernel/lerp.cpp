/*
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void lerp(GM_ADDR end, GM_ADDR end, GM_ADDR temp, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
*/
#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
template<typename T> struct Map {using type = T;};
template<> struct Map<int8_t> {using type = half;};
template<typename TYPE_START, typename TYPE_END, typename TYPE_WEIGHT, typename TYPE_Y> class KernelLerp {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelLerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_start.SetGlobalBuffer((__gm__ TYPE_START*)start + startPointer, bufferlength);
        Gm_end.SetGlobalBuffer((__gm__ TYPE_END*)end + startPointer, bufferlength);
        Gm_weight.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weight + startPointer, bufferlength);
        //Gm_value.SetGlobalBuffer((__gm__ TYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_start, BUFFER_NUM, this->tileLength * sizeof(TYPE_START));
        pipe.InitBuffer(Q_end, BUFFER_NUM, this->tileLength * sizeof(TYPE_END));
        pipe.InitBuffer(Q_weight, BUFFER_NUM, this->tileLength * sizeof(TYPE_WEIGHT));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileLength * sizeof(float));
        //this->value = Gm_value.GetValue(0);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_START> start = Q_start.AllocTensor<TYPE_START>();
        LocalTensor<TYPE_END> end = Q_end.AllocTensor<TYPE_END>();
        LocalTensor<TYPE_WEIGHT> weight = Q_weight.AllocTensor<TYPE_WEIGHT>();
        DataCopy(start, Gm_start[progress * this->tileLength], length);
        DataCopy(end, Gm_end[progress * this->tileLength], length);
        DataCopy(weight, Gm_weight[progress * this->tileLength], length);
        Q_start.EnQue(start);
        Q_end.EnQue(end);
        Q_weight.EnQue(weight);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_START> start = Q_start.DeQue<TYPE_START>();
        LocalTensor<TYPE_END> end = Q_end.DeQue<TYPE_END>();
        LocalTensor<TYPE_WEIGHT> weight = Q_weight.DeQue<TYPE_WEIGHT>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();        
        /*
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, signed char>) {
            auto p1 = tmp1.Get<half>();
            auto p2 = tmp2.Get<half>();
            Cast(p1, end, RoundMode::CAST_NONE, length);
            Cast(p2, weight, RoundMode::CAST_NONE, length);
            Mul(p1, p1, p2, length);
            //Muls(p1, p1, value, length);
            Cast(p2, start, RoundMode::CAST_NONE, length);
            Add(p1, p1, p2, length);
            Cast(y, p1, RoundMode::CAST_NONE, length);
        }
        else {
            Mul(end, end, weight, length);
            //Muls(end, end, value, length);
            Add(y, end, start, length);
        }
        */
        if constexpr (std::is_same_v<TYPE_START, half>) {
            auto p1 = tmp1.Get<float>();
            auto p2 = tmp2.Get<float>();
            auto p3 = tmp3.Get<float>();
            Cast(p1, start, RoundMode::CAST_NONE, length);
            Cast(p2, end, RoundMode::CAST_NONE, length);
            Cast(p3, weight, RoundMode::CAST_NONE, length);
            Sub(p2, p2, p1, length);
            Mul(p3, p3, p2, length);
            Add(p3, p3, p1, length);
            Cast(y, p3, RoundMode::CAST_NONE, length);
        } else {
            Sub(end, end, start, length);
            Mul(weight, weight, end, length);
            Add(y, start, weight, length);
        }

        Q_start.FreeTensor(start);
        Q_end.FreeTensor(end);
        Q_weight.FreeTensor(weight);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_start, Q_end, Q_weight;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3;
    GlobalTensor<TYPE_START> Gm_start;
    GlobalTensor<TYPE_END> Gm_end;
    GlobalTensor<TYPE_WEIGHT> Gm_weight;
    
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    
};
template<typename TYPE_START, typename TYPE_END, typename TYPE_WEIGHT, typename TYPE_Y> class KernelLerp_Broadcast {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelLerp_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, uint32_t start_length, uint32_t end_length, uint32_t weight_length, uint32_t total_length, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, int64_t ss[], int64_t numshapes, int64_t sf[]) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->startLength = start_length;
        this->endLength = end_length;
        this->weightLength = weight_length;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->startPointer = core_size * GetBlockIdx();
        this->total_length = total_length;

        for (int i = 0; i < 192; ++i) {
            ((int64_t *)this->shape)[i] = ss[i];
        }
        this->numshapes = numshapes;
        for(int i = 0; i < 64; ++i) {
            ((int64_t *)this->shapefull)[i] = sf[i];
        }

        Gm_start.SetGlobalBuffer((__gm__ TYPE_START*)start, total_length);
        Gm_end.SetGlobalBuffer((__gm__ TYPE_END*)end, total_length);
        Gm_weight.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weight, total_length);
        //Gm_value.SetGlobalBuffer((__gm__ TYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, total_length);

        pipe.InitBuffer(Q_start, BUFFER_NUM, this->tileLength * sizeof(TYPE_START));
        pipe.InitBuffer(Q_end, BUFFER_NUM, this->tileLength * sizeof(TYPE_END));
        pipe.InitBuffer(Q_weight, BUFFER_NUM, this->tileLength * sizeof(TYPE_WEIGHT));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileLength * sizeof(float));
        //this->value = Gm_value.GetValue(0);
    }
    __aicore__ inline void Process() {
        // int32_t loopCount = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        // for (int32_t i = 0; i < loopCount-1; i++) {
        //     uint32_t position = startPointer + i * this->tileLength;
        //     CopyIn(position, this->tileLength);
        //     Compute(this->tileLength);
        //     CopyOut(position, this->tileLength);
        // }
        // uint32_t position = startPointer + (loopCount - 1) * this->tileLength;
        // uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        // CopyIn(position, length);
        // Compute(length);
        // CopyOut(position, length);

        int32_t loopCount = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_START> start = Q_start.AllocTensor<TYPE_START>();
        LocalTensor<TYPE_END> end = Q_end.AllocTensor<TYPE_END>();
        LocalTensor<TYPE_WEIGHT> weight = Q_weight.AllocTensor<TYPE_WEIGHT>();
        if(this->startLength < this->total_length) {
            BroadCStart(start, progress * this->tileLength, length);
        } else {            
            DataCopy(start, Gm_start[progress * this->tileLength], length);
        }
        if(this->endLength < this->total_length) {
            BroadCEnd(end, progress * this->tileLength, length);
        } else {
            DataCopy(end, Gm_end[progress * this->tileLength], length);
        }
        if(this->weightLength < this->total_length) {
            BroadCWeight(weight, progress * this->tileLength, length);
        } else {
            DataCopy(weight, Gm_weight[progress * this->tileLength], length);
        }
        // DataCopy(start, Gm_start[position % startLength], length);
        // DataCopy(end, Gm_end[position % endLength], length);
        // DataCopy(weight, Gm_weight[position % weightLength], length);
        Q_start.EnQue(start);
        Q_end.EnQue(end);
        Q_weight.EnQue(weight);
    }
    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<TYPE_START> start = Q_start.DeQue<TYPE_START>();
        LocalTensor<TYPE_END> end = Q_end.DeQue<TYPE_END>();
        LocalTensor<TYPE_WEIGHT> weight = Q_weight.DeQue<TYPE_WEIGHT>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<TYPE_START, half>) {
            auto p1 = tmp1.Get<float>();
            auto p2 = tmp2.Get<float>();
            auto p3 = tmp3.Get<float>();
            Cast(p1, start, RoundMode::CAST_NONE, length);
            Cast(p2, end, RoundMode::CAST_NONE, length);
            Cast(p3, weight, RoundMode::CAST_NONE, length);
            Sub(p2, p2, p1, length);
            Mul(p3, p3, p2, length);
            Add(p3, p3, p1, length);
            Cast(y, p3, RoundMode::CAST_NONE, length);
        } else {
            Sub(end, end, start, length);
            Mul(weight, weight, end, length);
            Add(y, start, weight, length);
        }

        Q_start.FreeTensor(start);
        Q_end.FreeTensor(end);
        Q_weight.FreeTensor(weight);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
    __aicore__ inline void BroadCStart(LocalTensor<TYPE_START> &dst, uint32_t offset, uint32_t length) {
        if(this->startLength == 1) {
            TYPE_START tmp = Gm_start.GetValue(0);
            Duplicate(dst, tmp, length);
            return;
        }
        for(uint32_t i = 0; i < length; i++) {
            int idxtmp = 0;
            int istart = i + offset;
            for(int k = 1; k <= this->numshapes; k++) {
                int kpos = 0;
                int krange = 1;
                if(k < this->numshapes) {
                    for(int m = k + 1; m <= this->numshapes; m++) {
                        krange *= shapefull[m - 1];
                    }
                    kpos = istart / krange;
                    istart = istart % krange;
                } else {
                    krange = shapefull[k - 1];
                    kpos = istart % krange;
                }
                //idxtmp += kpos * this->stride[k - 1];
                int krangeB = 1;
                if(shapefull[k - 1] == shape[0][k - 1]) {
                    if(k < this->numshapes) {
                        for(int m = k + 1; m <= this->numshapes; m++) {
                            krangeB *= shape[0][m - 1];
                        }
                        idxtmp += kpos * krangeB;
                    }  else {
                        idxtmp += kpos;
                    }
                }
            }
            TYPE_START tmp = Gm_start.GetValue(idxtmp);
            dst.SetValue(i, tmp);
        }
    }
    __aicore__ inline void BroadCEnd(LocalTensor<TYPE_START> &dst, uint32_t offset, uint32_t length) {
        if(this->endLength == 1) {
            TYPE_START tmp = Gm_end.GetValue(0);
            Duplicate(dst, tmp, length);
            return;
        }
        for(uint32_t i = 0; i < length; i++) {
            int idxtmp = 0;
            int istart = i + offset;
            for(int k = 1; k <= this->numshapes; k++) {
                int kpos = 0;
                int krange = 1;
                if(k < this->numshapes) {
                    for(int m = k + 1; m <= this->numshapes; m++) {
                        krange *= shapefull[m - 1];
                    }
                    kpos = istart / krange;
                    istart = istart % krange;
                } else {
                    krange = shapefull[k - 1];
                    kpos = istart % krange;
                }
                //idxtmp += kpos * this->stride[k - 1];
                int krangeB = 1;
                if(shapefull[k - 1] == shape[1][k - 1]) {
                    if(k < this->numshapes) {
                        for(int m = k + 1; m <= this->numshapes; m++) {
                            krangeB *= shape[1][m - 1];
                        }
                        idxtmp += kpos * krangeB;
                    }  else {
                        idxtmp += kpos;
                    }
                }
            }
            TYPE_START tmp = Gm_end.GetValue(idxtmp);
            dst.SetValue(i, tmp);
        }
    }
    __aicore__ inline void BroadCWeight(LocalTensor<TYPE_START> &dst, uint32_t offset, uint32_t length) {
        if(this->weightLength == 1) {
            TYPE_START tmp = Gm_weight.GetValue(0);
            Duplicate(dst, tmp, length);
            return;
        }
        for(uint32_t i = 0; i < length; i++) {
            int idxtmp = 0;
            int istart = i + offset;
            for(int k = 1; k <= this->numshapes; k++) {
                int kpos = 0;
                int krange = 1;
                if(k < this->numshapes) {
                    for(int m = k + 1; m <= this->numshapes; m++) {
                        krange *= shapefull[m - 1];
                    }
                    kpos = istart / krange;
                    istart = istart % krange;
                } else {
                    krange = shapefull[k - 1];
                    kpos = istart % krange;
                }
                //idxtmp += kpos * this->stride[k - 1];
                int krangeB = 1;
                if(shapefull[k - 1] == shape[2][k - 1]) {
                    if(k < this->numshapes) {
                        for(int m = k + 1; m <= this->numshapes; m++) {
                            krangeB *= shape[2][m - 1];
                        }
                        idxtmp += kpos * krangeB;
                    }  else {
                        idxtmp += kpos;
                    }
                }
            }
            TYPE_START tmp = Gm_weight.GetValue(idxtmp);
            dst.SetValue(i, tmp);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_start, Q_end, Q_weight;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3;
    GlobalTensor<TYPE_START> Gm_start;
    GlobalTensor<TYPE_END> Gm_end;
    GlobalTensor<TYPE_WEIGHT> Gm_weight;
    
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startPointer;
    uint32_t startLength;
    uint32_t endLength;
    uint32_t weightLength;
    uint32_t total_length;
    int64_t shape[3][64];
    int64_t numshapes;
    int64_t shapefull[64];
};

template<typename TYPE_START, typename TYPE_END, typename TYPE_WEIGHT, typename TYPE_Y> class KernelLerp_Wi {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelLerp_Wi() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_start.SetGlobalBuffer((__gm__ TYPE_START*)start + startPointer, bufferlength);
        Gm_end.SetGlobalBuffer((__gm__ TYPE_END*)end + startPointer, bufferlength);
        Gm_weight.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weight, 1);
        //Gm_value.SetGlobalBuffer((__gm__ TYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_start, BUFFER_NUM, this->tileLength * sizeof(TYPE_START));
        pipe.InitBuffer(Q_end, BUFFER_NUM, this->tileLength * sizeof(TYPE_END));
        //pipe.InitBuffer(Q_weight, BUFFER_NUM, this->tileLength * sizeof(TYPE_WEIGHT));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileLength * sizeof(float));
        this->weight = Gm_weight.GetValue(0);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_START> start = Q_start.AllocTensor<TYPE_START>();
        LocalTensor<TYPE_END> end = Q_end.AllocTensor<TYPE_END>();
        //LocalTensor<TYPE_WEIGHT> weight = Q_weight.AllocTensor<TYPE_WEIGHT>();
        DataCopy(start, Gm_start[progress * this->tileLength], length);
        DataCopy(end, Gm_end[progress * this->tileLength], length);
        //DataCopy(weight, Gm_weight[progress * this->tileLength], length);
        Q_start.EnQue(start);
        Q_end.EnQue(end);
        //Q_weight.EnQue(weight);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_START> start = Q_start.DeQue<TYPE_START>();
        LocalTensor<TYPE_END> end = Q_end.DeQue<TYPE_END>();
        //LocalTensor<TYPE_WEIGHT> weight = Q_weight.DeQue<TYPE_WEIGHT>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();        

        if constexpr (std::is_same_v<TYPE_START, half>) {
            auto p1 = tmp1.Get<float>();
            auto p2 = tmp2.Get<float>();
            //auto p3 = tmp3.Get<float>();
            float weightfl = this->weight;
            Cast(p1, start, RoundMode::CAST_NONE, length);
            Cast(p2, end, RoundMode::CAST_NONE, length);
            //Cast(p3, weight, RoundMode::CAST_NONE, length);
            Sub(p2, p2, p1, length);
            Muls(p2, p2, weightfl, length);
            Add(p2, p2, p1, length);
            Cast(y, p2, RoundMode::CAST_NONE, length);
        } else {
            Sub(end, end, start, length);
            Muls(end, end, this->weight, length);
            Add(y, start, end, length);
        }

        Q_start.FreeTensor(start);
        Q_end.FreeTensor(end);
        //Q_weight.FreeTensor(weight);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_start, Q_end;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1, tmp2, tmp3;
    GlobalTensor<TYPE_START> Gm_start;
    GlobalTensor<TYPE_END> Gm_end;
    GlobalTensor<TYPE_WEIGHT> Gm_weight;
    
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    TYPE_START weight;
    
};

extern "C" __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // if (tiling_data.start_length == tiling_data.total_length && tiling_data.end_length == tiling_data.total_length && tiling_data.weight_length == 1) {
    //     KernelLerp_Wi<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
    //     op.Init(start, end, weight, y, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    //     op.Process();
    //     return;
    // }
    if (tiling_data.start_length == tiling_data.total_length && tiling_data.end_length == tiling_data.total_length && tiling_data.weight_length == tiling_data.total_length) {
        KernelLerp<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
        op.Init(start, end, weight, y, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    else {
        KernelLerp_Broadcast<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
        op.Init(start, end, weight, y, tiling_data.start_length, tiling_data.end_length, tiling_data.weight_length, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.shape, tiling_data.numshapes, tiling_data.shapefull);
        op.Process();
    }
}
