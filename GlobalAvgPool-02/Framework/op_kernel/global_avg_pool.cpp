#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

template<typename TYPE_X, typename TYPE_Y> class KernelGlobalAvgPool {
	using T = TYPE_Y;
public:
    __aicore__ inline KernelGlobalAvgPool() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchsize, 
                                uint32_t inputNum, uint32_t tileLength) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->batchsize = batchsize;
        this->inputNum = inputNum;
        this->tileLength = tileLength;
        this->size_y = this->inputNum / this->batchsize;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->inputNum + 8);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->inputNum / this->batchsize);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        //pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));  
        //pipe.InitBuffer(tmpBuffer2, this->availableTileLength * sizeof(half));      

        int typeSize = sizeof(TYPE_X);                           // half类型为2Bytes，float类型为4Bytes，按需填入
        // 再根据数据类型定义两个单位
        int elementsPerBlock = 32 / typeSize;       // 1个block存放的元素个数
        int elementsPerRepeat = 256 / typeSize;     // 1次repeat可以处理的元素个数

        // 最后确定首次最大repeat值
        int firstMaxRepeat = this->tileLength;           // 此处需要注意：对于tensor高维切分计算接口，firstMaxRepeat就是	repeatTimes；对于tensor前n个数据计算接口，firstMaxRepeat为count/elementsPerRepeat，比如在half类型下firstMaxRepeat就是count/128，在float类型下为count/64，按需填入，对于count<elementsPerRepeat的场景，firstMaxRepeat就是1

        int iter1OutputCount = firstMaxRepeat;                                              // 第一轮操作产生的元素个数
        int iter1AlignEnd = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock; // 第一轮产生的元素个数做向上取整
        int finalWorkLocalNeedSize = iter1AlignEnd;   
        pipe.InitBuffer(workQueue, BUFFER_NUM, finalWorkLocalNeedSize * sizeof(TYPE_X));          
       
    }
    __aicore__ inline void ProcessNew() {
        if constexpr (std::is_same_v<TYPE_X, half>) {
            Process();
            return;
        }
        if(this->batchsize < this->tileLength) {
            int32_t length = (this->batchsize + 7) / 8 * 8;
            for(int k = 0; k < this->size_y; k++) {
                this->sumtmp = 0;
                this->meantmp = 0;
                CopyInSZ(k, length);
                ComputeSZ(k, this->batchsize);
            }
        } else {
            int32_t loopnum = this->batchsize / this->tileLength;
            for(int k = 0; k < this->size_y; k++) {
                this->sumtmp = 0;
                this->meantmp = 0;
                for(int m = 0; m < loopnum; m++) {
                    CopyInMT(k, m, this->tileLength);
                    ComputeMT(k, m, this->tileLength);
                }
                if(this->batchsize % this->tileLength != 0) {
                    int32_t taillength = this->batchsize % this->tileLength;
                    int32_t taillengthalign = (taillength + 7) / 8 * 8;
                    CopyInMT(k, loopnum, taillengthalign);
                    ComputeMT(k, loopnum, taillength);
                }
                this->meantmp = this->sumtmp / this->batchsize;
                yGm.SetValue(k, this->meantmp);
            }
        }
    }

    __aicore__ inline void Process() {
        if constexpr (std::is_same_v<TYPE_X, half>) {
            for (int32_t c = 0; c < this-> size_y; c++) {
                float sumtmpfl = 0;
                TYPE_X sumtmphalf = 0;
                int32_t baseindex = c * this->batchsize;
                for (int32_t i = 0; i < this->batchsize; i++) {
                    TYPE_X tmp = xGm.GetValue(baseindex + i);	
                    float tmpfl = tmp;
                    sumtmpfl += tmpfl;
                    sumtmphalf = sumtmpfl;
                    sumtmpfl = sumtmphalf;	           	
                }
                int32_t count = this->batchsize;
                float meantmp = sumtmpfl / count;
                TYPE_X meantmphalf = meantmp;
                yGm.SetValue(c, meantmphalf);
            }            
        } 
    }

private:
    __aicore__ inline void CopyInSZ(int32_t progress, int32_t length) {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->batchsize], length);
        inQueueX.EnQue(xLocal);
    } 
    __aicore__ inline void ComputeSZ(int32_t progress, uint32_t reallength) {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> workLocal = workQueue.AllocTensor<TYPE_X>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();

        if constexpr (std::is_same_v<TYPE_X, float>) {
            ReduceSum(tmpLocal, xLocal, workLocal, reallength); 
            this->sumtmp += tmpLocal.GetValue(0);      
            this->meantmp = this->sumtmp / this->batchsize;
            yGm.SetValue(progress, this->meantmp);
        }
        
        inQueueX.FreeTensor(xLocal);
        workQueue.FreeTensor(workLocal);
    }
    __aicore__ inline void CopyInMT(int32_t channel, int32_t progress, int32_t length) {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[channel * this->batchsize + progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }    

    __aicore__ inline void ComputeMT(int32_t channel, int32_t progress, uint32_t reallength) {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_X> workLocal = workQueue.AllocTensor<TYPE_X>();
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        if constexpr (std::is_same_v<TYPE_X, float>) {
            ReduceSum(tmpLocal, xLocal, workLocal, reallength); 
            this->sumtmp += tmpLocal.GetValue(0);      
        }
        
        inQueueX.FreeTensor(xLocal);
        workQueue.FreeTensor(workLocal);
    }
    __aicore__ inline int RoundUp(int a, int b){ 
    	return (a + b - 1) / b;
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer; 
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, 1> workQueue;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchsize;
    uint32_t inputNum;
    uint32_t size_y;
    uint32_t tileLength;
    float sumtmp;
    float meantmp;

};
extern "C" __global__ __aicore__ void global_avg_pool(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelGlobalAvgPool<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.batchsize, tiling_data.inputNum, tiling_data.tileLength);
    op.ProcessNew();
}