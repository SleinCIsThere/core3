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

//aiu/gguf_headers.h

//GGUF on-disk format constants (little endian).
//
//File layout (v2/v3):
//  GgufHeader
//  metadataKvCount x { string key, U32 valueType, value }
//  tensorCount x { string name, U32 nDims, U64 dims[nDims], U32 ggmlType, U64 offset }
//  padding to 'general.alignment' (metadata KV, default 32)
//  data section (each tensor blob at dataStart + tensor.offset)
//
//Strings are U64 length + bytes, not null terminated. dims[0] is the fastest moving dimension
// (row length). Tensor offsets are relative to the start of the data section and pre-aligned.
//
//Version note: v2 and v3 share this layout (v1 used U32 counts/lengths and is not accepted;
// the v3 bump was about big endian support, which we also don't accept).

#pragma once
#include "types/base/c8.h"

#ifdef __cplusplus
	extern "C" {
#endif

#define GGUF_MAGIC C8x4('G', 'G', 'U', 'F')

typedef enum EGgufVersion {
	EGgufVersion_V2 = 2,
	EGgufVersion_V3 = 3
} EGgufVersion;

//Metadata value type ids as stored on disk

typedef enum EGgufValueType {
	EGgufValueType_U8,
	EGgufValueType_I8,
	EGgufValueType_U16,
	EGgufValueType_I16,
	EGgufValueType_U32,
	EGgufValueType_I32,
	EGgufValueType_F32,
	EGgufValueType_Bool,
	EGgufValueType_String,
	EGgufValueType_Array,
	EGgufValueType_U64,
	EGgufValueType_I64,
	EGgufValueType_F64,
	EGgufValueType_Count
} EGgufValueType;

//Scalar size in bytes, or 0 for string/array/invalid

static inline U64 EGgufValueType_size(EGgufValueType type) {
	switch(type) {

		case EGgufValueType_U8:  case EGgufValueType_I8:  case EGgufValueType_Bool:
			return 1;

		case EGgufValueType_U16: case EGgufValueType_I16:
			return 2;

		case EGgufValueType_U32: case EGgufValueType_I32: case EGgufValueType_F32:
			return 4;

		case EGgufValueType_U64: case EGgufValueType_I64: case EGgufValueType_F64:
			return 8;

		default:
			return 0;
	}
}

//ggml tensor type ids as stored in tensor infos.
//Only the ids we can map to EQuantType are listed; anything else parses as unsupported
// (kept in the tensor list with data unset, never a whole-file error: real model mixes
// routinely contain a few tensors of types we don't run).

typedef enum EGgmlType {
	EGgmlType_F32  = 0,
	EGgmlType_F16  = 1,
	EGgmlType_Q8_0 = 8,
	EGgmlType_Q4_K = 12,
	EGgmlType_Q6_K = 14
} EGgmlType;

#define GGUF_DEFAULT_ALIGNMENT 32    //Used when the general.alignment KV is absent
#define GGUF_MAX_DIMS 4

#ifdef __cplusplus
	}
#endif
