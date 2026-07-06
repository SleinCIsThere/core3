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

//aiu/quant.c

//CPU reference implementation over the shared quant_read.h accessors, so CPU and GPU literally
//run the same dequant source. The hoisted per-sub-block loops below only reorder work; per-weight
//results are bitwise identical to BlockQ*_dequantizeAt (asserted by tests).

#include "aiu/quant.h"
#include "types/base/error.h"
#include "types/base/mathf.h"
#include "types/base/buffer_base.h"

//Per block, blockOffset in bytes from the buffer view start

static void BlockQ4K_dequantize(QuantBuf buf, U32 blockOffset, F32 *y) {

	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ4K_OFF_D);
	const F32 dmin = Quant_loadF16AsF32(buf, blockOffset + BlockQ4K_OFF_DMIN);

	for(U32 g = 0; g < 4; ++g) {

		const F32 d1 = d * BlockQ4K_scaleD(buf, blockOffset, g * 2);
		const F32 m1 = dmin * BlockQ4K_scaleM(buf, blockOffset, g * 2);
		const F32 d2 = d * BlockQ4K_scaleD(buf, blockOffset, g * 2 + 1);
		const F32 m2 = dmin * BlockQ4K_scaleM(buf, blockOffset, g * 2 + 1);

		const U32 qs = blockOffset + BlockQ4K_OFF_QS + g * 32;

		for(U32 l = 0; l < 32; ++l)
			*y++ = d1 * (Quant_loadBits(buf, qs + l, 0, 4)) - m1;

		for(U32 l = 0; l < 32; ++l)
			*y++ = d2 * (Quant_loadBits(buf, qs + l, 4, 4)) - m2;
	}
}

static F32 BlockQ4K_dot(QuantBuf buf, U32 blockOffset, const F32 *y) {

	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ4K_OFF_D);
	const F32 dmin = Quant_loadF16AsF32(buf, blockOffset + BlockQ4K_OFF_DMIN);
	F32 sum = 0;

	for(U32 g = 0; g < 4; ++g) {

		const F32 d1 = d * BlockQ4K_scaleD(buf, blockOffset, g * 2);
		const F32 m1 = dmin * BlockQ4K_scaleM(buf, blockOffset, g * 2);
		const F32 d2 = d * BlockQ4K_scaleD(buf, blockOffset, g * 2 + 1);
		const F32 m2 = dmin * BlockQ4K_scaleM(buf, blockOffset, g * 2 + 1);

		const U32 qs = blockOffset + BlockQ4K_OFF_QS + g * 32;
		const F32 *y1 = y + g * 64;
		const F32 *y2 = y1 + 32;

		for(U32 l = 0; l < 32; ++l)
			sum += (d1 * (Quant_loadBits(buf, qs + l, 0, 4)) - m1) * y1[l];

		for(U32 l = 0; l < 32; ++l)
			sum += (d2 * (Quant_loadBits(buf, qs + l, 4, 4)) - m2) * y2[l];
	}

	return sum;
}

static void BlockQ6K_dequantize(QuantBuf buf, U32 blockOffset, F32 *y) {
	for(U32 i = 0; i < BlockQ6K_ELEMENTS; ++i)
		y[i] = BlockQ6K_dequantizeAt(buf, blockOffset, i);
}

static F32 BlockQ6K_dot(QuantBuf buf, U32 blockOffset, const F32 *y) {

	F32 sum = 0;

	for(U32 i = 0; i < BlockQ6K_ELEMENTS; ++i)
		sum += BlockQ6K_dequantizeAt(buf, blockOffset, i) * y[i];

	return sum;
}

static void BlockQ80_dequantize(QuantBuf buf, U32 blockOffset, F32 *y) {

	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ80_OFF_D);

	for(U32 i = 0; i < BlockQ80_ELEMENTS; ++i)
		y[i] = d * Quant_loadI8(buf, blockOffset + BlockQ80_OFF_QS + i);
}

static F32 BlockQ80_dot(QuantBuf buf, U32 blockOffset, const F32 *y) {

	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ80_OFF_D);
	F32 sum = 0;

	for(U32 i = 0; i < BlockQ80_ELEMENTS; ++i)
		sum += (F32) Quant_loadI8(buf, blockOffset + BlockQ80_OFF_QS + i) * y[i];

	return d * sum;
}

// -- Public API --------------------------------------------------------------------

static Bool Quant_validate(EQuantType type, const void *blocks, U64 elements, const void *other, Error *e_rr) {

	Bool s_uccess = true;

	if(!blocks || !other)
		retError(clean, Error_nullPointer(!blocks ? 1 : 3, "Quant_validate()::blocks and output are required"));

	if((U64)(uintptr_t)blocks & 3)
		retError(clean, Error_invalidParameter(
			1, 0, "Quant_validate()::blocks must be 4 byte aligned (quant_read.h reads it as U32[])"
		));

	if(type >= EQuantType_Count)
		retError(clean, Error_invalidEnum(0, (U64)type, EQuantType_Count, "Quant_validate()::type is invalid"));

	const U64 blockElements = Quant_blockNumElements(type);

	if(!elements || (elements % blockElements))
		retError(clean, Error_invalidParameter(
			2, 0, "Quant_validate()::elements must be a non zero multiple of the block element count"
		));

clean:
	return s_uccess;
}

Bool Quant_dequantize(EQuantType type, const void *blocks, U64 elements, F32 *f32Out, Error *e_rr) {

	Bool s_uccess = true;

	gotoIfError3(clean, Quant_validate(type, blocks, elements, f32Out, e_rr));

	const QuantBuf buf = (QuantBuf) blocks;

	switch(type) {

		case EQuantType_F32:

			Buffer_memcpy(
				Buffer_createRef(f32Out, elements * sizeof(F32)),
				Buffer_createRefConst(blocks, elements * sizeof(F32))
			);

			break;

		case EQuantType_F16: {

			const F16 *src = (const F16*) blocks;

			for(U64 i = 0; i < elements; ++i)
				f32Out[i] = F16_castF32(src[i]);

			break;
		}

		case EQuantType_Q4K:

			for(U64 i = 0; i < elements / BlockQ4K_ELEMENTS; ++i)
				BlockQ4K_dequantize(buf, (U32)(i * BlockQ4K_STRIDE), f32Out + i * BlockQ4K_ELEMENTS);

			break;

		case EQuantType_Q6K:

			for(U64 i = 0; i < elements / BlockQ6K_ELEMENTS; ++i)
				BlockQ6K_dequantize(buf, (U32)(i * BlockQ6K_STRIDE), f32Out + i * BlockQ6K_ELEMENTS);

			break;

		case EQuantType_Q80:

			for(U64 i = 0; i < elements / BlockQ80_ELEMENTS; ++i)
				BlockQ80_dequantize(buf, (U32)(i * BlockQ80_STRIDE), f32Out + i * BlockQ80_ELEMENTS);

			break;

		default:
			retError(clean, Error_unsupportedOperation(0, "Quant_dequantize()::type is not supported"));
	}

clean:
	return s_uccess;
}

Bool Quant_dot(EQuantType type, const void *blocks, const F32 *f32, U64 elements, F32 *dst, Error *e_rr) {

	Bool s_uccess = true;
	F32 sum = 0;

	gotoIfError3(clean, Quant_validate(type, blocks, elements, dst, e_rr));

	if(!f32)
		retError(clean, Error_nullPointer(2, "Quant_dot()::f32 is required"));

	const QuantBuf buf = (QuantBuf) blocks;

	switch(type) {

		case EQuantType_F32: {

			const F32 *src = (const F32*) blocks;

			for(U64 i = 0; i < elements; ++i)
				sum += src[i] * f32[i];

			break;
		}

		case EQuantType_F16: {

			const F16 *src = (const F16*) blocks;

			for(U64 i = 0; i < elements; ++i)
				sum += F16_castF32(src[i]) * f32[i];

			break;
		}

		case EQuantType_Q4K:

			for(U64 i = 0; i < elements / BlockQ4K_ELEMENTS; ++i)
				sum += BlockQ4K_dot(buf, (U32)(i * BlockQ4K_STRIDE), f32 + i * BlockQ4K_ELEMENTS);

			break;

		case EQuantType_Q6K:

			for(U64 i = 0; i < elements / BlockQ6K_ELEMENTS; ++i)
				sum += BlockQ6K_dot(buf, (U32)(i * BlockQ6K_STRIDE), f32 + i * BlockQ6K_ELEMENTS);

			break;

		case EQuantType_Q80:

			for(U64 i = 0; i < elements / BlockQ80_ELEMENTS; ++i)
				sum += BlockQ80_dot(buf, (U32)(i * BlockQ80_STRIDE), f32 + i * BlockQ80_ELEMENTS);

			break;

		default:
			retError(clean, Error_unsupportedOperation(0, "Quant_dot()::type is not supported"));
	}

	*dst = sum;

clean:
	return s_uccess;
}

Bool Quant_quantizeQ80(const F32 *f32, U64 elements, void *blocksOut, Error *e_rr) {

	Bool s_uccess = true;

	gotoIfError3(clean, Quant_validate(EQuantType_Q80, blocksOut, elements, f32, e_rr));

	U8 *out = (U8*) blocksOut;

	for(U64 i = 0; i < elements / BlockQ80_ELEMENTS; ++i) {

		const F32 *x = f32 + i * BlockQ80_ELEMENTS;
		U8 *block = out + i * BlockQ80_STRIDE;
		F32 amax = 0;

		for(U32 j = 0; j < BlockQ80_ELEMENTS; ++j) {

			const F32 v = F32_abs(x[j]);

			if(v > amax)
				amax = v;
		}

		const F32 d = amax / 127;
		const F32 id = d ? 1 / d : 0;
		const F16 dF16 = F32_castF16(d);

		//Tight layout writes: d at 0..1, qs at 2..33. memcpy for d avoids any alignment concern;
		//qs bytes are written directly (writing bytes is fine, only the U32 read view needs
		//the aligned base)

		*(F16*)(block + BlockQ80_OFF_D) = dF16;

		for(U32 j = 0; j < BlockQ80_ELEMENTS; ++j) {
			const I32 q = (I32) F32_round(x[j] * id);
			block[BlockQ80_OFF_QS + j] = (U8)(q & 0xFF);
		}
	}

clean:
	return s_uccess;
}
