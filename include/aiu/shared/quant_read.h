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

//aiu/shared/quant_read.h

#pragma once

//Shared C/HLSL dequantization over tight (unpadded) GGUF block data.
//
//Blocks are read through QuantBuf, a word-indexable view of the raw tensor bytes:
//  C:      const U32*             (tensor base must be 4 byte aligned; GGUF tensor alignment is
//                                  32 by default so any tensor start qualifies)
//  HLSL:   StructuredBuffer<U32x4> (16 byte loads; see Quant_loadU32)
//
//Q6_K and Q8_0 strides are not multiples of 4, so any byte/half within a block can
//straddle a U32 boundary; the load helpers below handle that with two word reads worst case.
//
//ALLOCATION RULE: buffers must be allocated padded up to a multiple of 16 bytes. Two reasons:
// a load near the end of the last block may touch up to 3 bytes past the tight size, and the HLSL
// view element is U32x4 (16 bytes), so the buffer element count is allocSize / 16. The pad content
// is never interpreted, only read.

#include "aiu/shared/quant_types.h"

#ifdef __HLSL_VERSION
	#include "@flp.h"
	#define QuantBuf StructuredBuffer<U32x4>
#else
	#include "types/math/flp.h"
	typedef const U32 *QuantBuf;
#endif

//Load primitive; the ONLY place where the two views differ. Everything above it is shared verbatim.
//
//HLSL views the tensor as StructuredBuffer<U32x4>: one 16 byte load per element, fewer DXIL/SPIR-V
// load ops than 4x U32, and the natural shape for future sub-block-per-thread kernels (a Q4_K
// header (d, dmin, scales[12]) is exactly one U32x4; 1 element = 32 nibbles of qs).
//
//C views it as const U32*; word index = byte / 4.

#ifdef __HLSL_VERSION
	static inline U32 Quant_loadU32(QuantBuf buf, U32 wordIndex) {
		return buf[wordIndex >> 2][wordIndex & 3];
	}
#else
	static inline U32 Quant_loadU32(QuantBuf buf, U32 wordIndex) {
		return buf[wordIndex];
	}
#endif

//Generic unaligned bit loader on top of the primitive: count bits starting at
// byteOffset * 8 + bitOffset, straddling at most one U32 boundary (count <= 32). byteOffset is
// relative to the view start; bitOffset may exceed 8 for convenience (e.g. nibble i = bitOffset i * 4).
//Fully inlined; with constant count/bitOffset this folds to the same shift+mask as hand written code.

static inline U32 Quant_loadBits(QuantBuf buf, U32 byteOffset, U32 bitOffset, U32 count) {

	U32 bit = ((byteOffset & 3) << 3) + bitOffset;
	U32 word = (byteOffset >> 2) + (bit >> 5);
	bit &= 31;

	U32 v = Quant_loadU32(buf, word) >> bit;

	if(bit + count > 32)                                        //Straddle: bit >= 1 here, shift < 32
		v |= Quant_loadU32(buf, word + 1) << (32 - bit);

	return count == 32 ? v : v & ((1u << count) - 1);
}

//Common scalar loads

static inline U32 Quant_loadU8(QuantBuf buf, U32 byteOffset) {
	return Quant_loadBits(buf, byteOffset, 0, 8);
}

static inline U32 Quant_loadU16(QuantBuf buf, U32 byteOffset) {
	return Quant_loadBits(buf, byteOffset, 0, 16);
}

//Sign extend low 8 bits; xor+sub instead of shifting a signed int (implementation-defined in C)

static inline I32 Quant_loadI8(QuantBuf buf, U32 byteOffset) {
	return (I32)(Quant_loadU8(buf, byteOffset) ^ 0x80u) - 128;
}

//F16 loads. The bits must be reinterpreted, NOT value-cast: in C, F16 is a U16 storage typedef so
// the cast is the reinterpretation, but in HLSL F16 is float16_t and (F16)bits would convert the
// number.
//Quant_loadF16 returns the native F16 so 16-bit-typed kernels can compute without a round trip
// through F32; Quant_loadF16AsF32 is the expanded form the F32 dequant paths use.

#ifdef __HLSL_VERSION
	static inline F16 Quant_loadF16(QuantBuf buf, U32 byteOffset) {
		return F16_fromU32Bits(Quant_loadU16(buf, byteOffset));
	}
#else
	static inline F16 Quant_loadF16(QuantBuf buf, U32 byteOffset) {
		return (F16) Quant_loadU16(buf, byteOffset);
	}
#endif

static inline F32 Quant_loadF16AsF32(QuantBuf buf, U32 byteOffset) {
	return F16_castF32(Quant_loadF16(buf, byteOffset));
}

//Q4_K:

//Unpack the 6-bit sub-block scale (D) / min (M) for scale index j in [0, 8>.
//Layout reference: ggml-quants.c get_scale_min_k4. Two functions instead of out-params so the
// call sites read identically in C and HLSL; the shared byte loads CSE away.

static inline U32 BlockQ4K_scaleD(QuantBuf buf, U32 blockOffset, U32 j) {

	const U32 s = blockOffset + BlockQ4K_OFF_SCALES;

	if(j < 4)
		return Quant_loadBits(buf, s + j, 0, 6);

	return Quant_loadBits(buf, s + j + 4, 0, 4) | (Quant_loadBits(buf, s + j - 4, 6, 2) << 4);
}

static inline U32 BlockQ4K_scaleM(QuantBuf buf, U32 blockOffset, U32 j) {

	const U32 s = blockOffset + BlockQ4K_OFF_SCALES;

	if(j < 4)
		return Quant_loadBits(buf, s + j + 4, 0, 6);

	return Quant_loadBits(buf, s + j + 4, 4, 4) | (Quant_loadBits(buf, s + j, 6, 2) << 4);
}

//Dequantize weight i in [0, 256> of the block starting at byte blockOffset

static inline F32 BlockQ4K_dequantizeAt(QuantBuf buf, U32 blockOffset, U32 i) {

	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ4K_OFF_D);
	const F32 dmin = Quant_loadF16AsF32(buf, blockOffset + BlockQ4K_OFF_DMIN);

	//Which 64-wide group (picks a sub-scale pair), which sub-scale of the pair (low vs high nibble region),
	// then the byte within the group

	const U32 sub64 = i >> 6;
	const U32 j = (sub64 << 1) | ((i >> 5) & 1);
	const U32 sc = BlockQ4K_scaleD(buf, blockOffset, j);
	const U32 m = BlockQ4K_scaleM(buf, blockOffset, j);

	const U32 q = Quant_loadBits(buf, blockOffset + BlockQ4K_OFF_QS + (sub64 << 5) + (i & 31), (i & 32) ? 4 : 0, 4);

	return d * sc * q - dmin * m;
}

//Q6_K:

static inline F32 BlockQ6K_dequantizeAt(QuantBuf buf, U32 blockOffset, U32 i) {

	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ6K_OFF_D);

	//2 groups of 128; each group has 4 quarters of 32. Quarts 0/1 use ql low nibbles,
	//2/3 use ql high nibbles. qh contributes 2 bits per weight, field picked by quart

	const U32 half128 = i >> 7;
	const U32 r = i & 127;
	const U32 quart = r >> 5;
	const U32 l = r & 31;

	const U32 ql = Quant_loadBits(
		buf, blockOffset + BlockQ6K_OFF_QL + (half128 << 6) + ((quart & 1) << 5) + l, quart < 2 ? 0 : 4, 4
	);

	const U32 qh = Quant_loadBits(buf, blockOffset + BlockQ6K_OFF_QH + (half128 << 5) + l, quart << 1, 2);

	const I32 q = (I32)(ql | (qh << 4)) - 32;
	const I32 sc = Quant_loadI8(buf, blockOffset + BlockQ6K_OFF_SCALES + (half128 << 3) + (quart << 1) + (l >> 4));

	return d * sc * q;
}

//Q8_0:

static inline F32 BlockQ80_dequantizeAt(QuantBuf buf, U32 blockOffset, U32 i) {
	const F32 d = Quant_loadF16AsF32(buf, blockOffset + BlockQ80_OFF_D);
	return d * Quant_loadI8(buf, blockOffset + BlockQ80_OFF_QS + i);
}

//Generic:

//Dispatch on EQuantType for one weight. Call with a compile time / oxc uniform type so the
//switch folds; per-weight runtime dispatch would be wasteful.

static inline F32 Quant_dequantizeAt(U32 type, QuantBuf buf, U32 blockOffset, U32 i) {

	switch(type) {
		case EQuantType_Q4K:    return BlockQ4K_dequantizeAt(buf, blockOffset, i);
		case EQuantType_Q6K:    return BlockQ6K_dequantizeAt(buf, blockOffset, i);
		case EQuantType_Q80:    return BlockQ80_dequantizeAt(buf, blockOffset, i);
		default:                return 0;
	}
}
