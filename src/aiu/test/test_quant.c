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

//aiu/test/test_quant.c

//Quantization unit tests.
//
//What is covered:
//  1. Q8_0         - quantize + dequantize error bound, dot vs dequantized dot
//  2. Q4_K         - synthetic block with analytic expected values, dot vs dequantized dot
//  3. Q6_K         - synthetic block with analytic expected values
//  4. Consistency  - per-weight Quant_dequantizeAt == bulk Quant_dequantize on random data,
//                    including misaligned (Q6_K/Q8_0) multi-block offsets that straddle U32 words
//  5. API          - input validation, Quant_size / Quant_allocSize
//
//Run in CI, no GPU required (GPU bit-compat is a separate test that dispatches
//quant_test.comp.hlsl and diffs against Quant_dequantize once the graphics harness lands).

#include "test_aiu_shared.h"
#include "aiu/quant.h"
#include "types/test/test.h"
#include "types/base/mathf.h"
#include "types/base/buffer_base.h"
#include "types/math/type_cast.h"
#include "types/math/rand.h"

#include <string.h>

//Deterministic PRNG using OxC3 Random_seed / Random_sample so results reproduce across platforms.
//Random_sample returns [0, 1>; scale into [-scale, scale>.

static inline F32 Test_quantRandF32(U32 *seed, F32 scale) {
	return (Random_sample(seed) * 2 - 1) * scale;
}

static inline U8 Test_quantRandU8(U32 *seed) {
	return (U8)(Random_sample(seed) * 256);
}

//Tight blocks live in byte arrays; back them with U32 so the base is 4 byte aligned and padded.
//BYTES rounded up to 16 per the allocation rule in quant.h.

#define Test_quantBacking(name, bytes)          \
	U32 name##Backing[((bytes) + 15) / 16 * 4]; \
	U8 *name = (U8*) name##Backing

// -- 1. Q8_0 --------------------------------------------------------------------

static void Test_quantQ80(Test *t) {

	Test_setModule(t, "Aiu quant Q8_0");

	enum { N = 1024, BLOCKS = N / BlockQ80_ELEMENTS };

	F32 src[N], deq[N], y[N];
	Test_quantBacking(blocks, BLOCKS * BlockQ80_STRIDE);
	U32 seed = Random_seed(0x80, 0xB);

	for(U64 i = 0; i < N; ++i) {
		src[i] = Test_quantRandF32(&seed, 4);
		y[i] = Test_quantRandF32(&seed, 1);
	}

	Test_assert(t, "quantize", Quant_quantizeQ80(src, N, blocks, &t->err));
	Test_assert(t, "dequantize", Quant_dequantize(EQuantType_Q80, blocks, N, deq, &t->err));

	//Max quantization error is d / 2 = amax / 254 per element (+ f16 scale rounding)

	F32 maxErr = 0;

	for(U64 i = 0; i < N; ++i)
		maxErr = F32_max(maxErr, F32_abs(deq[i] - src[i]));

	Test_assert(t, "error bound", maxErr <= 4.f / 127 * 0.51f + 1e-3f);

	F32 dot = 0, ref = 0;

	for(U64 i = 0; i < N; ++i)
		ref += deq[i] * y[i];

	Test_assert(t, "dot", Quant_dot(EQuantType_Q80, blocks, y, N, &dot, &t->err));
	Test_assert(t, "dot matches dequant dot", F32_abs(dot - ref) <= F32_abs(ref) * 1e-4f + 1e-3f);
}

// -- 2. Q4_K --------------------------------------------------------------------

static void Test_quantQ4K(Test *t) {

	Test_setModule(t, "Aiu quant Q4_K");

	//Synthetic block: d = 0.5, dmin = 0.25, all sub scales/mins = 1, nibbles 3 (low) / 8 (high).
	// scales bytes 0..7 hold the 6-bit values for j < 4 (D in 0..3, M in 4..7); bytes 8..11 hold
	// the packed nibbles for j >= 4 (low nibble D, high nibble M); high 2 bits stay 0 in bytes 0..7
	// so the reconstructed 6-bit values for j >= 4 are also 1.

	Test_quantBacking(b, BlockQ4K_STRIDE);
	Buffer_unsetAllBits(Buffer_createRef(b, BlockQ4K_STRIDE), NULL);

	*(F16*)(b + BlockQ4K_OFF_D)    = F32_castF16(0.5f);     //d
	*(F16*)(b + BlockQ4K_OFF_DMIN) = F32_castF16(0.25f);    //dmin

	for(U32 i = 0; i < 8; ++i)
		b[BlockQ4K_OFF_SCALES + i] = 1;

	for(U32 i = 8; i < 12; ++i)
		b[BlockQ4K_OFF_SCALES + i] = 0x11;

	for(U32 i = 0; i < 128; ++i)
		b[BlockQ4K_OFF_QS + i] = 0x83;              //Low nibble 3, high nibble 8

	F32 out[BlockQ4K_ELEMENTS];

	Test_assert(t, "dequantize", Quant_dequantize(EQuantType_Q4K, b, BlockQ4K_ELEMENTS, out, &t->err));

	//Expected: d * sc * q - dmin * m = 0.5 * q - 0.25; q = 3 for low nibbles, 8 for high

	Bool match = true;

	for(U64 i = 0; i < BlockQ4K_ELEMENTS; ++i) {
		const F32 q = (i & 32) ? 8 : 3;
		match &= F32_abs(out[i] - (0.5f * q - 0.25f)) < 1e-6f;
	}

	Test_assert(t, "synthetic block values", match);

	F32 y[BlockQ4K_ELEMENTS], dot = 0, ref = 0;
	U32 seed = Random_seed(0x4C, 0xD);

	for(U64 i = 0; i < BlockQ4K_ELEMENTS; ++i) {
		y[i] = Test_quantRandF32(&seed, 1);
		ref += out[i] * y[i];
	}

	Test_assert(t, "dot", Quant_dot(EQuantType_Q4K, b, y, BlockQ4K_ELEMENTS, &dot, &t->err));
	Test_assert(t, "dot matches dequant dot", F32_abs(dot - ref) <= F32_abs(ref) * 1e-4f + 1e-3f);
}

// -- 3. Q6_K --------------------------------------------------------------------

static void Test_quantQ6K(Test *t) {

	Test_setModule(t, "Aiu quant Q6_K");

	//All scales 2, all 6-bit values 35 (ql nibble 3, qh 2-bit field 2 -> 3 | (2 << 4) = 35, bias -32 -> 3)

	Test_quantBacking(b, BlockQ6K_STRIDE);
	Buffer_unsetAllBits(Buffer_createRef(b, BlockQ6K_STRIDE), NULL);

	Buffer_setAllToU8(Buffer_createRef(b + BlockQ6K_OFF_QL,     128), 0x33, NULL);
	Buffer_setAllToU8(Buffer_createRef(b + BlockQ6K_OFF_QH,      64), 0xAA, NULL);
	Buffer_setAllToU8(Buffer_createRef(b + BlockQ6K_OFF_SCALES,  16), 0x02, NULL);

	*(F16*)(b + BlockQ6K_OFF_D) = F32_castF16(0.25f);     //d

	F32 out[BlockQ6K_ELEMENTS];

	Test_assert(t, "dequantize", Quant_dequantize(EQuantType_Q6K, b, BlockQ6K_ELEMENTS, out, &t->err));

	//Expected: d * sc * q = 0.25 * 2 * 3 = 1.5 everywhere

	Bool match = true;

	for(U64 i = 0; i < BlockQ6K_ELEMENTS; ++i)
		match &= F32_abs(out[i] - 1.5f) < 1e-6f;

	Test_assert(t, "synthetic block values", match);
}

// -- 4. Per-weight vs bulk consistency -------------------------------------------

//Random multi-block data; block 1+ of Q6_K/Q8_0 start at non-4-byte-aligned offsets (210, 34...),
// so this exercises the straddling U32 loads. Compared bitwise (memcmp) so NaN patterns count too.

static void Test_quantConsistency(Test *t) {

	Test_setModule(t, "Aiu quant consistency");

	U32 seed = Random_seed(0xC0, 0xE);

	//Q4_K, 4 blocks

	{
		enum { BLOCKS = 4, N = BLOCKS * BlockQ4K_ELEMENTS };
		Test_quantBacking(b, BLOCKS * BlockQ4K_STRIDE);

		for(U32 i = 0; i < BLOCKS * BlockQ4K_STRIDE; ++i)
			b[i] = Test_quantRandU8(&seed);

		F32 bulk[N], per[N];
		Test_assert(t, "Q4_K dequantize", Quant_dequantize(EQuantType_Q4K, b, N, bulk, &t->err));

		for(U32 i = 0; i < N; ++i)
			per[i] = Quant_dequantizeAt(
				EQuantType_Q4K, (QuantBuf) b,
				(i / BlockQ4K_ELEMENTS) * BlockQ4K_STRIDE, i & (BlockQ4K_ELEMENTS - 1)
			);

		Buffer bulkb = Buffer_createRef(bulk, sizeof(bulk));
		Buffer perb  = Buffer_createRef(per, sizeof(per));
		Test_assert(t, "Q4_K per-weight == bulk", Buffer_cmp(bulkb, perb) == ECompareResult_Eq);
	}

	//Q6_K, 4 blocks (blocks 1..3 start at byte 210, 420, 630, all misaligned)

	{
		enum { BLOCKS = 4, N = BLOCKS * BlockQ6K_ELEMENTS };
		Test_quantBacking(b, BLOCKS * BlockQ6K_STRIDE);

		for(U32 i = 0; i < BLOCKS * BlockQ6K_STRIDE; ++i)
			b[i] = Test_quantRandU8(&seed);

		F32 bulk[N], per[N];
		Test_assert(t, "Q6_K dequantize", Quant_dequantize(EQuantType_Q6K, b, N, bulk, &t->err));

		for(U32 i = 0; i < N; ++i)
			per[i] = Quant_dequantizeAt(
				EQuantType_Q6K, (QuantBuf) b,
				(i / BlockQ6K_ELEMENTS) * BlockQ6K_STRIDE, i & (BlockQ6K_ELEMENTS - 1)
			);

		Buffer bulkb = Buffer_createRef(bulk, sizeof(bulk));
		Buffer perb  = Buffer_createRef(per, sizeof(per));
		Test_assert(t, "Q6_K per-weight == bulk (misaligned blocks)", Buffer_cmp(bulkb, perb) == ECompareResult_Eq);
	}

	//Q8_0, 16 blocks (strides of 34 hit every alignment mod 4: 0, 2, 0, 2, ... and odd via 34 * k)

	{
		enum { BLOCKS = 16, N = BLOCKS * BlockQ80_ELEMENTS };
		Test_quantBacking(b, BLOCKS * BlockQ80_STRIDE);

		for(U32 i = 0; i < BLOCKS * BlockQ80_STRIDE; ++i)
			b[i] = Test_quantRandU8(&seed);

		F32 bulk[N], per[N];
		Test_assert(t, "Q8_0 dequantize", Quant_dequantize(EQuantType_Q80, b, N, bulk, &t->err));

		for(U32 i = 0; i < N; ++i)
			per[i] = Quant_dequantizeAt(
				EQuantType_Q80, (QuantBuf) b,
				(i / BlockQ80_ELEMENTS) * BlockQ80_STRIDE, i & (BlockQ80_ELEMENTS - 1)
			);

		Buffer bulkb = Buffer_createRef(bulk, sizeof(bulk));
		Buffer perb  = Buffer_createRef(per, sizeof(per));
		Test_assert(t, "Q8_0 per-weight == bulk (misaligned blocks)", Buffer_cmp(bulkb, perb) == ECompareResult_Eq);
	}
}

// -- 5. API validation ------------------------------------------------------------

static void Test_quantValidation(Test *t) {

	Test_setModule(t, "Aiu quant validation");

	F32 out[BlockQ80_ELEMENTS];
	Test_quantBacking(b, BlockQ80_STRIDE);
	Buffer_unsetAllBits(Buffer_createRef(b, BlockQ80_STRIDE), NULL);

	Test_assert(t, "null blocks", !Quant_dequantize(EQuantType_Q80, NULL, BlockQ80_ELEMENTS, out, NULL));
	Test_assert(t, "misaligned count", !Quant_dequantize(EQuantType_Q80, b, BlockQ80_ELEMENTS - 1, out, NULL));
	Test_assert(t, "invalid type", !Quant_dequantize(EQuantType_Count, b, BlockQ80_ELEMENTS, out, NULL));
	Test_assert(t, "null f32", !Quant_dot(EQuantType_Q80, b, NULL, BlockQ80_ELEMENTS, out, NULL));
	Test_assert(t, "misaligned base", !Quant_dequantize(EQuantType_Q80, b + 1, BlockQ80_ELEMENTS, out, NULL));

	Test_assert(t, "Quant_size Q4_K", Quant_size(EQuantType_Q4K, 512) == 2 * 144);
	Test_assert(t, "Quant_size Q6_K tight", Quant_size(EQuantType_Q6K, 256) == 210);
	Test_assert(t, "Quant_size Q8_0 tight", Quant_size(EQuantType_Q80, 32) == 34);
	Test_assert(t, "Quant_size misaligned", !Quant_size(EQuantType_Q4K, 100));
	Test_assert(t, "Quant_size F16", Quant_size(EQuantType_F16, 3) == 6);

	Test_assert(t, "Quant_allocSize pads to 16", Quant_allocSize(EQuantType_Q80, 32) == 48);
	Test_assert(t, "Quant_allocSize Q6_K", Quant_allocSize(EQuantType_Q6K, 256) == 224);
	Test_assert(t, "Quant_allocSize multiple of 16 stays", Quant_allocSize(EQuantType_Q4K, 256) == 144);
}

void Test_quant(Test *t) {
	Test_quantQ80(t);
	Test_quantQ4K(t);
	Test_quantQ6K(t);
	Test_quantConsistency(t);
	Test_quantValidation(t);
}
