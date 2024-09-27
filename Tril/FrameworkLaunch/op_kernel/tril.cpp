#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X, typename TYPE_Y> class KernelTril {
    using T = TYPE_X;
public:
    __aicore__ inline KernelTril() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, int32_t diagonal, int32_t dimnuma, int32_t dimnumb, int32_t dimnumbalign) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        //this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM: 0);
        this->diagonal = diagonal;
        this->dimnuma = dimnuma;
        this->dimnumb = dimnumb;
        this->totalLength = totalLength;
        this->dimnumbalign = dimnumbalign;
        //auto startPointer = core_size * GetBlockIdx();
        this->startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->dimnumbalign * sizeof(TYPE_X));        
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->dimnumbalign * sizeof(TYPE_Y));


    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->totalLength / this->dimnumb;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->dimnumbalign);
            Compute(i, this->dimnumbalign);
            CopyOut(i, this->dimnumbalign);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        DataCopy(x, Gm_x[progress * this->dimnumb], length);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {

        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();

        int32_t curA =  progress % this->dimnuma;
        int32_t zerocount = this->dimnumb - curA - 1 - this->diagonal;
        if(zerocount <= 0) {
            Adds(y, x, TYPE_X(0), this->dimnumb);
        } else if (zerocount >= this->dimnumb){
            Duplicate(y, TYPE_X(0), this->dimnumb);
        } else {
            Duplicate(y, TYPE_X(0), this->dimnumb);
            Duplicate(y, TYPE_X(1), this->dimnumb - zerocount);
            Mul(y, y, x, this->dimnumb);
        }

        //Duplicate(y, TYPE_X(1), this->dimnumb);

        //Mins(x, x, this->clip_value_max, length);
        //Maxs(y, x, this->clip_value_min, length);
        
        //uint32_t baseidx = this->startPointer + progress*length;
        // uint32_t baseidx = this->startPointer + progress*this->tileLength;
        // for(int j = 0; j < length; j++){
        //     //uint32_t posa = (baseidx + j + 1)%(this->dimnuma * this->dimnumb)/this->dimnuma + 1; 
        //     int32_t posa = (baseidx + j + 1)%(this->dimnuma * this->dimnumb)/this->dimnumb;
        //     int32_t posb = (baseidx + j + 1)%(this->dimnuma * this->dimnumb)%this->dimnumb;
        //     if(posa == 0 && posb == 0) {
        //     	posa = dimnuma;
        //     	posb = dimnumb;
        //     }else if(posb == 0) {
        //     	posb = dimnumb;
        //     } else {
        //     	posa += 1;
        //     }
        //     if(posa + this->diagonal > posb) {
        //     	x.SetValue(j, 0);
        //     } 
        // }
        // TYPE_X zero = 0;
        // Adds(y, x, zero, length);
        
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->dimnumb], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    GlobalTensor<TYPE_X> Gm_x;

    GlobalTensor<TYPE_Y> Gm_y;

    uint32_t startPointer;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    int32_t diagonal;
    int32_t dimnuma;
    int32_t dimnumb;
    uint32_t totalLength;
    int32_t dimnumbalign;
};

extern "C" __global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelTril<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.diagonal, tiling_data.dimnuma, tiling_data.dimnumb, tiling_data.dimnumbalign);
    op.Process();
}

/*
extern "C" __global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
*/