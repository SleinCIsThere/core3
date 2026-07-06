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

//aiu/shared/quant_types.h

#pragma once

//Quant type enum and GGUF block layout constants, shared between C and HLSL.
//
//Blocks are kept in the tight on-disk GGUF layout (no padding, no conversion), so raw reads
// (e.g. DirectStorage, optionally GDeflate-compressed) can go straight into the GPU buffer.
// Q6_K/Q8_0 strides aren't multiples of 4, so blocks land on arbitrary byte offsets; see
// quant_read.h for the alignment rule this puts on allocations.
//
//Layout reference: ggml-quants.c

#ifdef __HLSL_VERSION
	#include "@types.hlsli"
#else
	#include "types/base/types.h"
#endif

enum EQuantType {
	EQuantType_F32,
	EQuantType_F16,
	EQuantType_Q4K,
	EQuantType_Q6K,
	EQuantType_Q80,
	EQuantType_Count
};

//Per block: stride in bytes (tight GGUF layout), weight count, and field byte offsets.
//
//Field glossary: every format stores weights as small integers plus F16 scale(s), and
//dequantization is a fused multiply: weight = scale * q (+ optional offset term):
//  d       Super-block scale (F16). Every format has one.
//  dmin    Q4_K only: second F16 scale applied to the per-sub-block minimum (offset term),
//          weight = (d * sc) * q - (dmin * m). Asymmetric quant without storing f32 minimums.
//  scales  Per-sub-block quantized scales. Q4_K: 8 pairs (scale sc + min m), 6 bit each, packed
//          into 12 bytes. Q6_K: 16 signed 8 bit scales (symmetric, so no mins).
//  qs      The quantized weights. Q4_K: 4 bit nibbles, 2 per byte. Q8_0: signed 8 bit.
//  ql/qh   Q6_K splits its 6 bit weights across two arrays: ql = low 4 bits (nibble packed),
//          qh = high 2 bits (4 weights per byte). q = (ql | qh << 4) - 32.

enum EQuantLayout {

	//Q4_K: ~4.5 bpw. 256 weights per block, 8 sub-blocks of 32. 144 bytes.
	//
	// struct BlockQ4K {          offset  size
	//     F16  d;                //   0     2   super-block scale
	//     F16  dmin;             //   2     2   super-block min scale
	//     U8   scales[12];       //   4    12   6-bit (sc,m) pairs for 8 sub-blocks, packed
	//     U8   qs[128];          //  16   128   4-bit weights, 2 per byte (low nibble first)
	// };                         // 144 total, 4-byte aligned, on-disk == in-memory

	BlockQ4K_STRIDE = 144,
	BlockQ4K_ELEMENTS = 256,
	BlockQ4K_OFF_D = 0,
	BlockQ4K_OFF_DMIN = 2,
	BlockQ4K_OFF_SCALES = 4,
	BlockQ4K_OFF_QS = 16,

	//Q6_K: ~6.5 bpw. 256 weights per block, 16 sub-blocks of 16. 210 bytes (2-byte misalign).
	//
	// struct BlockQ6K {          offset  size
	//     U8   ql[128];          //   0   128   low 4 bits of each weight (nibble packed)
	//     U8   qh[64];           // 128    64   high 2 bits, 4 weights per byte
	//     I8   scales[16];       // 192    16   signed 8-bit per-sub-block scale
	//     F16  d;                // 208     2   super-block scale
	// };                         // 210 total, NOT 4-byte aligned; pad gpu allocation to 16B!

	BlockQ6K_STRIDE = 210,
	BlockQ6K_ELEMENTS = 256,
	BlockQ6K_OFF_QL = 0,
	BlockQ6K_OFF_QH = 128,
	BlockQ6K_OFF_SCALES = 192,
	BlockQ6K_OFF_D = 208,

	//Q8_0: ~8.5 bpw. 32 weights per block, no sub-blocks. 34 bytes (2-byte misalign).
	//
	// struct BlockQ80 {          offset  size
	//     F16  d;                //   0     2   block scale
	//     I8   qs[32];           //   2    32   signed 8-bit weights
	// };                         //  34 total, NOT 4-byte aligned; pad gpu allocation to 16B!

	BlockQ80_STRIDE = 34,
	BlockQ80_ELEMENTS = 32,
	BlockQ80_OFF_D = 0,
	BlockQ80_OFF_QS = 2
};

#if !defined(__HLSL_VERSION) && !defined(__cplusplus)
	typedef enum EQuantType EQuantType;
#endif
