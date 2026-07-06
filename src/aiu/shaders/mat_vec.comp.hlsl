/* OxC3(Oxsomi core 3), a general framework and toolset for cross-platform applications.
*  Copyright (C) 2023 - 2026 Oxsomi / Nielsbishere (Niels Brunekreef)
*
*  This program is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program. If not, see https://github.com/Oxsomi/core3/blob/main/LICENSE.
*  Be aware that GPL3 requires closed source products to be GPL3 too if released to the public.
*  To prevent this a separate license will have to be requested at contact@osomi.net for a premium;
*  This is called dual licensing.
*/

//aiu/matvec.comp.hlsl

//result[M] = mat[M, K] (quantized, tight GGUF layout) * vec[K] (F32).
//One threadgroup per output row; 256 threads
// (one per weight of a K-quant super block, or 8 blocks worth for Q8_0)
// stride the row's blocks, then subgroup + groupshared reduce.
//
//mat is the raw tensor byte stream viewed as U32; the tight layout means the tensor can come
// straight from disk (DirectStorage, optionally GDeflate) with no conversion. Buffer must be
// allocated padded to a multiple of 16 bytes (see aiu/quant_read.h allocation rule).
//
//QUANT_TYPE is an oxc uniform so one HLSL source yields one specialization per quant type;
// dead branches in Quant_dequantizeAt fold away at compile time.
//
//SM 6.10 note: the D3D12 linalg preview (dx::linalg MatVecMul) and VK cooperative vectors only
// accelerate native types (F16/F8/packed I8); GGUF K-quants need manual dequant, so this path
// dequantizes per thread.

#include "aiu/quant_read.h"

StructuredBuffer<U32x4> mat;                    //Tight blocks (see quant_types.h)
StructuredBuffer<F32> vec;                      //cols floats
RWStructuredBuffer<F32> result;                 //rows floats

struct MatVecCommand {
	U32 rows;                                   //M
	U32 cols;                                   //K, multiple of the block element count
};

PUSH_CONSTANT MatVecCommand cmd;

static const U32 MatVec_threads = 256;

groupshared F32 MatVec_partial[MatVec_threads / 4];          //Worst case subgroup size 4
groupshared F32 MatVec_partial2[MatVec_threads / 4 / 4];     //Double reduction (worst case; 16 left)
groupshared F32 MatVec_partial3[MatVec_threads / 4 / 4 / 4];

[[oxc::uniforms(
    U8 QUANT_TYPE = 2,         //EQuantType_Q4K
    U8 QUANT_STRIDE = 144,     //BlockQ4K_STRIDE
    U16 QUANT_ELEMENTS = 256   //BlockQ4K_ELEMENTS
)]]
[[oxc::uniforms(
    U8 QUANT_TYPE = 3,         //EQuantType_Q6K
    U8 QUANT_STRIDE = 210,     //BlockQ6K_STRIDE
    U16 QUANT_ELEMENTS = 256   //BlockQ6K_ELEMENTS
)]]
[[oxc::uniforms(
    U8 QUANT_TYPE = 4,         //EQuantType_Q80
    U8 QUANT_STRIDE = 34,      //BlockQ80_STRIDE
    U16 QUANT_ELEMENTS = 32    //BlockQ80_ELEMENTS
)]]
[[oxc::extension("SubgroupOperations")]]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(U32x3 groupId : SV_GroupID, U32 threadId : SV_GroupIndex) {

	U32 row = groupId.x;

	if(row >= cmd.rows)
		return;

	U32 blocksPerRow = cmd.cols / $$QUANT_ELEMENTS;
	U32 rowOffset = row * blocksPerRow * $$QUANT_STRIDE;       //Byte offset; rows can start misaligned, loads handle it

	//Each thread owns weight index (threadId % blockElements) within every
	// (threads / blockElements) apart block of the row.
    //For K-quants (256 wide) that's one weight per block;
    // for Q8_0 (32 wide) 8 consecutive blocks are covered per iteration.

	U32 blockOfThread = threadId / $$QUANT_ELEMENTS;
	U32 inBlock = threadId & ($$QUANT_ELEMENTS - 1);
	U32 blocksPerIter = MatVec_threads / $$QUANT_ELEMENTS;

	F32 sum = 0;

	for(U32 b = blockOfThread; b < blocksPerRow; b += blocksPerIter)
		sum +=
			Quant_dequantizeAt($$QUANT_TYPE, mat, rowOffset + b * $$QUANT_STRIDE, inBlock) *
			vec[b * $$QUANT_ELEMENTS + inBlock];

	//Groupshared reduce (4-128)

	sum = WaveActiveSum(sum);

    U32 lanes = WaveGetLaneCount();
	U32 wave = threadId / lanes;

	if(WaveIsFirstLane())
		MatVec_partial[wave] = sum;

    U32 threadsReduc = MatVec_threads / lanes;

	GroupMemoryBarrierWithGroupSync();

    //Another groupshared reduce into SGPR into memory once

    if(lanes >= 16) {       //>16x16 = 256, we're done
        
        if(threadId < threadsReduc) {

            F32 total = WaveActiveSum(MatVec_partial[threadId]);

            if(WaveIsFirstLane())
                result[row] = total;
        }

        return;
    }

    //Level 2

    if(threadId < threadsReduc) {

        F32 v = WaveActiveSum(MatVec_partial[threadId]);

        if(WaveIsFirstLane())
            MatVec_partial2[threadId / lanes] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    if(WaveGetLaneCount() >= 8) {

        if(threadId < threadsReduc2) {

            F32 total = WaveActiveSum(MatVec_partial2[threadId]);

            if(WaveIsFirstLane())
                result[row] = total;
        }

        return;
    }

    //Level 3 (lanes == 4 only)

    if(threadId < threadsReduc2) {

        F32 v = WaveActiveSum(MatVec_partial2[threadId]);

        if(WaveIsFirstLane())
            MatVec_partial3[threadId / lanes] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    if(threadId < threadsReduc2 / lanes) {

        F32 total = WaveActiveSum(MatVec_partial3[threadId]);

        if(WaveIsFirstLane())
            result[row] = total;
    }
}
