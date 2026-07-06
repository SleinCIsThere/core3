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

#ifdef __OXC_EXT_16BITTYPES

	typedef float16_t F16;
	typedef float16_t2 F16x2;
	typedef float16_t3 F16x3;
	typedef float16_t4 F16x4;

	typedef int16_t I16;
	typedef int16_t2 I16x2;
	typedef int16_t3 I16x3;
	typedef int16_t4 I16x4;

	typedef uint16_t U16;
	typedef uint16_t2 U16x2;
	typedef uint16_t3 U16x3;
	typedef uint16_t4 U16x4;

#endif

#ifdef __OXC_EXT_F64

	typedef double F64;
	typedef double2 F64x2;
	typedef double3 F64x3;
	typedef double4 F64x4;

#endif

typedef uint U32;
typedef float F32;
typedef int I32;

typedef uint2 U32x2;
typedef float2 F32x2;
typedef int2 I32x2;

typedef uint3 U32x3;
typedef float3 F32x3;
typedef int3 I32x3;

typedef uint4 U32x4;
typedef float4 F32x4;
typedef int4 I32x4;

#ifdef __OXC_EXT_I64

	typedef uint64_t U64;
	typedef uint64_t2 U64x2;
	typedef uint64_t3 U64x3;
	typedef uint64_t4 U64x4;

	typedef int64_t I64;
	typedef int64_t2 I64x2;
	typedef int64_t3 I64x3;
	typedef int64_t4 I64x4;

#endif

//Bool

typedef bool Bool;

typedef bool2 Boolx2;
typedef bool3 Boolx3;
typedef bool4 Boolx4;

//Binding graphics shader outputs (e.g. uv : _bind(0))

#define _bind(x) TEXCOORD##x
#define _flat nointerpolation

//Indirect draws

struct IndirectDraw {
	U32 vertexCount, instanceCount, vertexOffset, instanceOffset;
};

struct IndirectDrawIndexed {

	U32 indexCount, instanceCount, indexOffset;
	I32 vertexOffset;

	U32 instanceOffset;
	U32 padding[3];			//For alignment
};

struct IndirectDispatch {
	U32 x, y, z, pad;
};

#ifdef __spirv__
	#define UNKNOWN_FORMAT [[vk::image_format("unknown")]]
	#define PUSH_CONSTANT [[vk::push_constant]] 
#else
	#define UNKNOWN_FORMAT
	#define PUSH_CONSTANT
#endif

)"
