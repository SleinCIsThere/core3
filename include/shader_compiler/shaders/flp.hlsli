R"(
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

#pragma once
#include "@types.hlsli"

static inline U32 U32_fromF32Bits(F32 f) { return asuint(f); }
static inline U32x2 U32x2_fromF32x2Bits(F32x2 f) { return asuint(f); }
static inline U32x3 U32x3_fromF32x3Bits(F32x3 f) { return asuint(f); }
static inline U32x4 U32x4_fromF32x4Bits(F32x4 f) { return asuint(f); }

static inline F32 F32_fromU32Bits(U32 u) { return asfloat(u); }
static inline F32x2 F32x2_fromU32x2Bits(U32x2 u) { return asfloat(u); }
static inline F32x3 F32x3_fromU32x3Bits(U32x3 u) { return asfloat(u); }
static inline F32x4 F32x4_fromU32x4Bits(U32x4 u) { return asfloat(u); }

static inline F32 F32_fromU32Bits16(U32 bits) {
    return f16tof32(bits);
}

static inline U32 U32_fromF32Bits(F32 bits) {
    return f32tof16(bits);
}

#ifdef __OXC_EXT_16BITTYPES

    static inline F16 F16_castF32(F32 bits) { return (F16) bits; }
    static inline F16 F16_fromU16Bits(U16 bits) { return (F16) f16tof32(bits); }
    static inline F16 F16_fromU32Bits(U32 bits) { return (F16) f16tof32(bits); }

    static inline F32 F32_castF16(F16 bits) { return (F32) bits; }
    static inline U16 U16_fromF16Bits(F16 bits) { return f32tof16(bits); }
    static inline U16 U16_fromF32Bits(F32 bits) { return f32tof16(bits); }
    static inline U32 U32_fromF16Bits(F16 bits) { return f32tof16(bits); }

#endif

)"
