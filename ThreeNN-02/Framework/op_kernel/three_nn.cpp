#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
using namespace AscendC;
class KernelThreeNN {
public:
    __aicore__ inline KernelThreeNN() {}
    __aicore__ inline void Init(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR indices,
                                uint32_t bDim, uint32_t inputNum1, uint32_t inputNum2, uint32_t n, uint32_t m,
                                uint32_t tileLength)
    {

        this->bDim = bDim;
        this->inputNum1 = inputNum1;
        this->inputNum2 = inputNum2;
        this->n = n;
        this->m = m;
        this->tileLength = tileLength;

        xyz1Gm.SetGlobalBuffer((__gm__ float *)xyz1, this->inputNum1);
        xyz2Gm.SetGlobalBuffer((__gm__ float *)xyz2, this->inputNum2);
        distGm.SetGlobalBuffer((__gm__ float *)dist, this->inputNum1);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, this->inputNum1);
        //pipe.InitBuffer(inQueuexyz1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueuexyz2, BUFFER_NUM, this->tileLength * sizeof(float));
        //pipe.InitBuffer(outQueuedist, BUFFER_NUM, this->tileLength * sizeof(float));
        //pipe.InitBuffer(outQueueindices, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(tmpBufferA1, this->tileLength * sizeof(float) / 3);
        pipe.InitBuffer(tmpBufferA2, this->tileLength * sizeof(float) / 3);
        pipe.InitBuffer(tmpBufferA3, this->tileLength * sizeof(float) / 3);
        pipe.InitBuffer(tmpBufferP1, 2 * sizeof(uint32_t));
        pipe.InitBuffer(tmpBufferP2, 2 * sizeof(uint32_t));
        pipe.InitBuffer(tmpBufferP3, 2 * sizeof(uint32_t));
        pipe.InitBuffer(tmpBufferX1, this->tileLength * sizeof(float) / 3);
        pipe.InitBuffer(tmpBufferX2, this->tileLength * sizeof(float) / 3);
        pipe.InitBuffer(tmpBufferX3, this->tileLength * sizeof(float) / 3);

        int typeSize = 4; 
        int elementsPerBlock = 32 / typeSize;       // 1个block存放的元素个数
        int elementsPerRepeat = 256 / typeSize;     // 1次repeat可以处理的元素个数

        // 最后确定首次最大repeat值
        int firstMaxRepeat = this->m;           // 此处需要注意：对于tensor高维切分计算接口，firstMaxRepeat就是	repeatTimes；对于tensor前n个数据计算接口，firstMaxRepeat为count/elementsPerRepeat，比如在half类型下firstMaxRepeat就是count/128，在float类型下为count/64，按需填入，对于count<elementsPerRepeat的场景，firstMaxRepeat就是1

        int iter1OutputCount = firstMaxRepeat * 2;                                              // 第一轮操作产生的元素个数
        int iter2AlignStart = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock; // 第二轮操作起始位置偏移，即第一轮产生的元素个数按照datablock(32字节)向上对齐的结果
        // 第一轮计算完成后，后续可能还需要多轮迭代，此时不可以复用同一块空间，因为第一轮的中间结果索引还需要再进行使用，所以需要继续准备第二轮和第三轮的空间
        int iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;              // 第二轮操作产生的元素个数
        int iter3AlignStart = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock; // 第三轮操作起始位置偏移，即第二轮产生的元素个数按照datablock(32字节)向上对齐的结果
        int iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;              // 第三轮操作产生的元素个数
        int iter3AlignEnd = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;   // 第三轮产生的元素个数按照datablock(32字节)向上对齐的结果
        int finalWorkLocalNeedSize = iter2AlignStart + iter3AlignStart + iter3AlignEnd;       // 最终workLocal所需的空间大小  
        pipe.InitBuffer(workQueue, BUFFER_NUM, finalWorkLocalNeedSize * sizeof(float));  
        
    }
    __aicore__ inline void Process() {
        int32_t loopnum = this->m * 3 / this->tileLength;
        int32_t tailLength = this->m * 3 % this->tileLength == 0 ? 0: this->m * 3 - loopnum * this->tileLength;
        
        for(int32_t b = 0; b < this->bDim ;b++) {
            if(loopnum == 0) {
                uint32_t length =  this->m * 3;
                int32_t i = 1;
                CopyInXYZ2(b * this->m * 3, length);
                Compute(b * this->n * 3, length);
 
            } else {
                for (int32_t i = 0; i < loopnum; i++) {
                    CopyInXYZ2(b * this->m * 3 + i * this->tileLength, this->tileLength);
                    Compute(b * this->n * 3 + i * this->tileLength, this->tileLength);

                }
                if(tailLength > 0) {
                    tailLength = (tailLength + 47) / 48 * 48;
                    CopyInXYZ2(b * this->m * 3 + loopnum * this->tileLength, tailLength);
                    Compute(b * this->n * 3 + loopnum * this->tileLength, tailLength);
                }
            }
        }    
    }

private:
    __aicore__ inline void CopyInXYZ2(int32_t startpos, uint32_t length)
    {
        uint32_t lengthalign = 0;
        if(length % 8 == 0) {
            lengthalign = length;
        } else {
            lengthalign = length / 8 * 8 + 8;
        }
        LocalTensor<float> xyz2Local = inQueuexyz2.AllocTensor<float>();
        DataCopy(xyz2Local, xyz2Gm[startpos], lengthalign);
        inQueuexyz2.EnQue(xyz2Local);
    }

    __aicore__ inline void Compute(int32_t startpos, uint32_t length)
    {
        LocalTensor<float> xyz2Local = inQueuexyz2.DeQue<float>();
        LocalTensor<uint32_t> tmpP1 = tmpBufferP1.Get<uint32_t>();
        LocalTensor<uint32_t> tmpP2 = tmpBufferP2.Get<uint32_t>();
        LocalTensor<uint32_t> tmpP3 = tmpBufferP3.Get<uint32_t>();
        LocalTensor<float> tmpX = tmpBufferA1.Get<float>();
        LocalTensor<float> tmpY = tmpBufferA2.Get<float>();
        LocalTensor<float> tmpZ = tmpBufferA3.Get<float>();
        LocalTensor<float> xLocal = tmpBufferX1.Get<float>();
        LocalTensor<float> yLocal = tmpBufferX2.Get<float>();
        LocalTensor<float> zLocal = tmpBufferX3.Get<float>();
        LocalTensor<float> workLocal = workQueue.AllocTensor<float>();

        uint32_t mask = 48;
        uint64_t rsvdCnt = 0;
        uint32_t patternele1 = 0b01001001001001001001001001001001;
        uint32_t patternele2 = 0b10010010010010010010010010010010;
        uint32_t patternele3 = 0b10010010010010010010010010010010;
        uint32_t patternele4 = 0b00100100100100100100100100100100;
        uint32_t patternele5 = 0b00100100100100100100100100100100;
        uint32_t patternele6 = 0b01001001001001001001001001001001;
        tmpP1.SetValue(0, patternele1);
        tmpP1.SetValue(1, patternele2);
        tmpP2.SetValue(0, patternele3);
        tmpP2.SetValue(1, patternele4);
        tmpP3.SetValue(0, patternele5);
        tmpP3.SetValue(1, patternele6);

        uint16_t repeatTimes = this->m * 3 % 48 ? this->m * 3 / 48 + 1: this->m * 3 / 48;
        GatherMask(tmpX, xyz2Local, tmpP1, true, mask, { 1, repeatTimes, 6, 0 }, rsvdCnt);
        GatherMask(tmpY, xyz2Local, tmpP2, true, mask, { 1, repeatTimes, 6, 0 }, rsvdCnt);
        GatherMask(tmpZ, xyz2Local, tmpP3, true, mask, { 1, repeatTimes, 6, 0 }, rsvdCnt);

        for(int n = 0; n < this->n; n++) {
            float tmp1 = xyz1Gm.GetValue(startpos + n * 3);
            Duplicate(xLocal, tmp1, this->m);
            float tmp2 = xyz1Gm.GetValue(startpos + n * 3 + 1);
            Duplicate(yLocal, tmp2, this->m);
            float tmp3 = xyz1Gm.GetValue(startpos + n * 3 + 2);
            Duplicate(zLocal, tmp3, this->m);

            Sub(xLocal, tmpX, xLocal, this->m);
            Mul(xLocal, xLocal, xLocal, this->m);
            Sub(yLocal, tmpY, yLocal, this->m);
            Mul(yLocal, yLocal, yLocal, this->m);
            Sub(zLocal, tmpZ, zLocal, this->m);
            Mul(zLocal, zLocal, zLocal, this->m);

            Add(xLocal, xLocal, yLocal, this->m);
            Add(xLocal, xLocal, zLocal, this->m);

            ReduceMin(yLocal, xLocal, workLocal, this->m, true);
            float maxdist1 = yLocal.GetValue(0);
            //float maxIndex1 = yLocal.GetValue(1);
            //uint32_t realIndex1 = *reinterpret_cast<uint32_t>(&maxIndex1);
            float tmpvalue1 = yLocal.GetValue(1);
            uint32_t* ptr1 = reinterpret_cast<uint32_t*>(&tmpvalue1); // 获取 half 的位模式
            uint32_t realIndex1 = *reinterpret_cast<int*>(ptr1);
            xLocal.SetValue(realIndex1, float(3.40282e+38));

            ReduceMin(yLocal, xLocal, workLocal, this->m, true);
            float maxdist2 = yLocal.GetValue(0);
            //float maxIndex2 = yLocal.GetValue(1);
            //uint32_t realIndex2 = *reinterpret_cast<uint32_t>(&maxIndex2);
            float tmpvalue2 = yLocal.GetValue(1);
            uint32_t* ptr2 = reinterpret_cast<uint32_t*>(&tmpvalue2); // 获取 half 的位模式
            uint32_t realIndex2 = *reinterpret_cast<int*>(ptr2);
            xLocal.SetValue(realIndex2, float(3.40282e+38));

            ReduceMin(yLocal, xLocal, workLocal, this->m, true);
            float maxdist3 = yLocal.GetValue(0);
            //float maxIndex3 = yLocal.GetValue(1);
            //uint32_t realIndex3 = *reinterpret_cast<uint32_t>(&maxIndex3);
            float tmpvalue3 = yLocal.GetValue(1);
            uint32_t* ptr3 = reinterpret_cast<uint32_t*>(&tmpvalue3); // 获取 half 的位模式
            uint32_t realIndex3 = *reinterpret_cast<int*>(ptr3);

            /*
            float maxdist1 = 3.40282e+38;
            float maxdist2 = 3.40282e+38;
            float maxdist3 = 3.40282e+38;
            int32_t index1 = 0;
            int32_t index2 = 0;
            int32_t index3 = 0;
            for(int32_t m = 0; m < this->m; m++) {
                float cmp1 = xLocal.GetValue(m);
                if(cmp1 < maxdist1) {
                    maxdist3 = maxdist2;
                    maxdist2 = maxdist1;
                    maxdist1 = cmp1;
                    index3 = index2;
                    index2 = index1;
                    index1 = m;
                } else if(cmp1 < maxdist2) {
                    maxdist3 = maxdist2;
                    maxdist2 = cmp1;
                    index3 = index2;
                    index2 = m;
                } else if(cmp1 < maxdist3) {
                    maxdist3 = cmp1;
                    index3 = m;
                }
            }
            */
            distGm.SetValue(startpos + n * 3, maxdist1);
            indicesGm.SetValue(startpos + n * 3, realIndex1);
            distGm.SetValue(startpos + n * 3 + 1, maxdist2);
            indicesGm.SetValue(startpos + n * 3 + 1, realIndex2);
            distGm.SetValue(startpos + n * 3 + 2, maxdist3);
            indicesGm.SetValue(startpos + n * 3  + 2, realIndex3);
        }
        inQueuexyz2.FreeTensor(xyz2Local);
        workQueue.FreeTensor(workLocal);
    }

    __aicore__ inline void CopyOut(int32_t startpos, uint32_t length)
    {

    }
    __aicore__ inline int RoundUp(int a, int b){ 
    	return (a + b - 1) / b;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuexyz2;
    //TQue<QuePosition::VECOUT, BUFFER_NUM> outQueuedist, outQueueindices;
    TBuf<QuePosition::VECCALC> tmpBufferA1, tmpBufferA2, tmpBufferA3, tmpBufferP1, tmpBufferP2, tmpBufferP3;
    TBuf<QuePosition::VECCALC> tmpBufferX1, tmpBufferX2, tmpBufferX3;
    TQue<QuePosition::VECOUT, 1> workQueue;
    GlobalTensor<float> xyz1Gm;
    GlobalTensor<float> xyz2Gm;
    GlobalTensor<float> distGm;
    GlobalTensor<int32_t> indicesGm;

    uint32_t bDim;                           
    uint32_t inputNum1;                                   
    uint32_t inputNum2;
    uint32_t n;
    uint32_t m;
    uint32_t tileLength;

};



extern "C" __global__ __aicore__ void three_nn(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelThreeNN op;
    op.Init(xyz1, xyz2, dist, indices,
            tiling_data.bDim, tiling_data.inputNum1, tiling_data.inputNum2, tiling_data.n, tiling_data.m,
            tiling_data.tileLength);
    op.Process();
}