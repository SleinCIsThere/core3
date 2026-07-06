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

//aiu/quant.h

//C public API for GGUF quantized tensor data.
//
//Data stays in the tight on-disk GGUF layout end to end: no conversion, no padding. That keeps
// real ggufs loadable as-is and lets the GPU path stream tensors raw from disk (DirectStorage,
// optionally GDeflate compressed) without a CPU repack. The tradeoff is that Q6_K (210B) and
// Q8_0 (34B) blocks land on arbitrary byte offsets, handled by quant_read.h's unaligned loads.
//
//Pointer requirements for all functions taking quantized 'blocks':
//  - 4 byte aligned base (GGUF tensor data alignment is 32 by default, so tensors qualify)
//  - allocation padded to a multiple of 16 bytes (reads near the end of the last block may
//    touch up to 3 bytes past the tight size; see quant_read.h)

#pragma once
#include "aiu/shared/quant_read.h"

#ifdef __cplusplus
	extern "C" {
#endif

typedef struct Error Error;

//Block size in bytes and weights per block, or 0 if unknown/unsupported

static inline U64 Quant_blockSize(EQuantType type) {
	switch(type) {
		case EQuantType_F32:    return sizeof(F32);
		case EQuantType_F16:    return sizeof(F16);
		case EQuantType_Q4K:    return BlockQ4K_STRIDE;
		case EQuantType_Q6K:    return BlockQ6K_STRIDE;
		case EQuantType_Q80:    return BlockQ80_STRIDE;
		default:                return 0;
	}
}

static inline U64 Quant_blockNumElements(EQuantType type) {
	switch(type) {
		case EQuantType_F32:    return 1;
		case EQuantType_F16:    return 1;
		case EQuantType_Q4K:    return BlockQ4K_ELEMENTS;
		case EQuantType_Q6K:    return BlockQ6K_ELEMENTS;
		case EQuantType_Q80:    return BlockQ80_ELEMENTS;
		default:                return 0;
	}
}

//Tight data size in bytes for 'elements' weights stored as 'type'
// (excludes the 16 byte allocation padding; use Quant_allocSize when allocating).
// elements must be a multiple of Quant_blockNumElements(type).
//Returns 0 on failure.

static inline U64 Quant_size(EQuantType type, U64 elements) {

	const U64 blockElements = Quant_blockNumElements(type);

	if(!blockElements || (elements % blockElements))
		return 0;

	return (elements / blockElements) * Quant_blockSize(type);
}

//Allocation size: tight size padded up to a multiple of 16 so unaligned tail reads stay in bounds

static inline U64 Quant_allocSize(EQuantType type, U64 elements) {

	const U64 size = Quant_size(type, elements);

	if(!size)
		return 0;

	return (size + 15) &~ (U64)15;
}

//Dequantize 'elements' weights into f32Out. elements must be a multiple of the type's block
//element count. blocks and f32Out must not overlap.

Bool Quant_dequantize(EQuantType type, const void *blocks, U64 elements, F32 *f32Out, Error *e_rr);

//Quantized dot product against an F32 vector; dst = sum(dequant(blocks)[i] * f32[i]).
//Never materializes the dequantized row; this is the matvec inner loop building block.

Bool Quant_dot(EQuantType type, const void *blocks, const F32 *f32, U64 elements, F32 *dst, Error *e_rr);

//Quantize F32 weights to Q8_0 in tight layout
// (runtime activation quant; weight quant types come from GGUF, never produced here).
// elements must be a multiple of 32. blocksOut must be 4 byte aligned and hold Quant_allocSize(EQuantType_Q80, elements) bytes

Bool Quant_quantizeQ80(const F32 *f32, U64 elements, void *blocksOut, Error *e_rr);

#ifdef __cplusplus
	}
#endif
