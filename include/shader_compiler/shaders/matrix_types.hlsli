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

	//Float16 matrices

	typedef float16_t4x4 F16x4x4;
	typedef float16_t3x4 F16x3x4;
	typedef float16_t2x4 F16x2x4;

	typedef float16_t4x3 F16x4x3;
	typedef float16_t3x3 F16x3x3;
	typedef float16_t2x3 F16x2x3;

	typedef float16_t4x2 F16x4x2;
	typedef float16_t3x2 F16x3x2;
	typedef float16_t2x2 F16x2x2;

	typedef int16_t2 I16x2;
	typedef int16_t3 I16x3;
	typedef int16_t4 I16x4;

	typedef uint16_t2 U16x2;
	typedef uint16_t3 U16x3;
	typedef uint16_t4 U16x4;

	//Int16 matrices

	typedef int16_t4x4 I16x4x4;
	typedef int16_t3x4 I16x3x4;
	typedef int16_t2x4 I16x2x4;

	typedef int16_t4x3 I16x4x3;
	typedef int16_t3x3 I16x3x3;
	typedef int16_t2x3 I16x2x3;

	typedef int16_t4x2 I16x4x2;
	typedef int16_t3x2 I16x3x2;
	typedef int16_t2x2 I16x2x2;

	//Uint16 matrices

	typedef uint16_t4x4 U16x4x4;
	typedef uint16_t3x4 U16x3x4;
	typedef uint16_t2x4 U16x2x4;

	typedef uint16_t4x3 U16x4x3;
	typedef uint16_t3x3 U16x3x3;
	typedef uint16_t2x3 U16x2x3;

	typedef uint16_t4x2 U16x4x2;
	typedef uint16_t3x2 U16x3x2;
	typedef uint16_t2x2 U16x2x2;

#endif

#ifdef __OXC_EXT_F64

	//Float64 matrices

	typedef double4x4 F64x4x4;
	typedef double3x4 F64x3x4;
	typedef double2x4 F64x2x4;

	typedef double4x3 F64x4x3;
	typedef double3x3 F64x3x3;
	typedef double2x3 F64x2x3;

	typedef double4x2 F64x4x2;
	typedef double3x2 F64x3x2;
	typedef double2x2 F64x2x2;

#endif

//Float matrices

typedef float4x4 F32x4x4;
typedef float3x4 F32x3x4;
typedef float2x4 F32x2x4;

typedef float4x3 F32x4x3;
typedef float3x3 F32x3x3;
typedef float2x3 F32x2x3;

typedef float4x2 F32x4x2;
typedef float3x2 F32x3x2;
typedef float2x2 F32x2x2;

//Int matrices

typedef int4x4 I32x4x4;
typedef int3x4 I32x3x4;
typedef int2x4 I32x2x4;

typedef int4x3 I32x4x3;
typedef int3x3 I32x3x3;
typedef int2x3 I32x2x3;

typedef int4x2 I32x4x2;
typedef int3x2 I32x3x2;
typedef int2x2 I32x2x2;

//Uint matrices

typedef uint4x4 U32x4x4;
typedef uint3x4 U32x3x4;
typedef uint2x4 U32x2x4;

typedef uint4x3 U32x4x3;
typedef uint3x3 U32x3x3;
typedef uint2x3 U32x2x3;

typedef uint4x2 U32x4x2;
typedef uint3x2 U32x3x2;
typedef uint2x2 U32x2x2;

#ifdef __OXC_EXT_I64

	//Uint64 matrices

	typedef uint64_t4x4 U64x4x4;
	typedef uint64_t3x4 U64x3x4;
	typedef uint64_t2x4 U64x2x4;

	typedef uint64_t4x3 U64x4x3;
	typedef uint64_t3x3 U64x3x3;
	typedef uint64_t2x3 U64x2x3;

	typedef uint64_t4x2 U64x4x2;
	typedef uint64_t3x2 U64x3x2;
	typedef uint64_t2x2 U64x2x2;

	//Int64 matrices

	typedef int64_t4x4 I64x4x4;
	typedef int64_t3x4 I64x3x4;
	typedef int64_t2x4 I64x2x4;

	typedef int64_t4x3 I64x4x3;
	typedef int64_t3x3 I64x3x3;
	typedef int64_t2x3 I64x2x3;

	typedef int64_t4x2 I64x4x2;
	typedef int64_t3x2 I64x3x2;
	typedef int64_t2x2 I64x2x2;

#endif

)"
