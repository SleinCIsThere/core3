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

//aiu/gguf_file.h

//GGUF model file reader.
//
//GgufFile_read parses the metadata and tensor infos through a StreamCursor,
// but never touches tensor data: tensors come back as stream descriptors
// (streamOff/streamLen, like oiDL's DLEntryStream / dds' SubResourceData), so multi-GB models
// don't get loaded into memory and the GPU path can stream tensor blobs straight from disk.
//GgufTensor_readData loads a single tensor when the CPU path needs it, already padded per the
// quant.h allocation contract.
//
//The GgufFile holds a reference on the stream (RefPtr_inc) until GgufFile_free.
//
//Metadata strings and scalar array payloads are owned copies (freed by GgufFile_free); the
// metadata section is small compared to tensor data so this is fine.

#pragma once
#include "aiu/quant.h"
#include "aiu/gguf_headers.h"
#include "types/base/string_base.h"
#include "types/base/buffer_base.h"
#include "types/container/list.h"

#ifdef __cplusplus
	extern "C" {
#endif

typedef struct Allocator Allocator;

typedef struct RefPtr RefPtr;
typedef RefPtr StreamRef;

typedef struct GgufValue {

	U8 type;                    //EGgufValueType
	U8 arrayType;               //EGgufValueType of the elements, if type == Array
	U8 padding[6];
	U64 arrayCount;             //Element count, if type == Array

	union {
		U64 u64;                //U8/U16/U32/U64/Bool (zero extended)
		I64 i64;                //I8/I16/I32/I64 (sign extended)
		F32 f32;
		F64 f64;
		CharString string;      //Owned
	};

    //Array storage (type == Array). One of these is populated by arrayType:
    //  scalar elemType  -> arrayBacking holds the tight payload; arrayStrings/arrayValues null
    //  String           -> arrayBacking holds all [len][bytes]; arrayStrings = refs into it
    //  Array (nested)   -> arrayValues = arrayCount owned GgufValue elements
    Buffer arrayBacking;        //Owned; raw bytes for scalar or string arrays
    CharString *arrayStrings;   //Refs into arrayBacking (string arrays); part of arrayBacking's tail alloc or a second alloc
    GgufValue *arrayValues;     //Owned (nested arrays only)

} GgufValue;

typedef struct GgufKv {
	CharString key;             //Owned
	GgufValue value;
} GgufKv;

TList(GgufKv);

typedef struct GgufTensor {

	CharString name;            //Owned

	U8 nDims;                   //1..GGUF_MAX_DIMS
	U8 quantType;               //EQuantType, or EQuantType_Count if the ggml type is unsupported
	U8 padding[2];
	U32 ggmlType;               //Raw on-disk type id (also valid when unsupported)

	U64 dims[GGUF_MAX_DIMS];    //dims[0] is the fastest moving (row length); unused dims are 1
	U64 elements;               //Product of dims
	U64 offset;                 //Relative to the start of the data section (as on disk)

	U64 streamOff;              //Absolute stream offset of the tensor blob
	U64 streamLen;              //Tight Quant_size() bytes; 0 if quantType == EQuantType_Count (unknown block size)

} GgufTensor;

TList(GgufTensor);

typedef struct GgufFile {

	U32 version;                //EGgufVersion
	U32 alignment;              //general.alignment KV or GGUF_DEFAULT_ALIGNMENT

	ListGgufKv metadata;
	ListGgufTensor tensors;

	StreamRef *stream;          //Referenced (inc'd); tensor streamOffs point into it
	U64 dataStart;              //Absolute stream offset of the data section

} GgufFile;

//Parse a gguf from a stream starting at *streamOff.
//On success *streamOff is advanced to the end of the stream: gguf doesn't encode its total
// size, so the data section is assumed to run to the end
// (fine for standalone files; a gguf embedded mid-stream with trailing data can't be detected).
//Unsupported tensor types don't fail the parse; check GgufTensor::quantType per tensor.

Bool GgufFile_read(
	StreamRef *stream,
	U64 *streamOff,
	const Allocator *alloc,
	GgufFile *ggufFile,
	Error *e_rr
);

void GgufFile_free(GgufFile *ggufFile, const Allocator *alloc);

//Load one tensor's data for the CPU path. Allocates Quant_allocSize() bytes
// (the tight data plus the 16 byte padding required by the quant.h contract, pad zeroed),
// so the result can go straight into Quant_dequantize / Quant_dot.
//Fails for unsupported tensor types.

Bool GgufTensor_readData(
	const GgufFile *ggufFile,
	const GgufTensor *tensor,
	const Allocator *alloc,
	Buffer *result,
	Error *e_rr
);

//Lookups (linear scan; NULL if missing). Key/name compares are case sensitive.

const GgufKv *GgufFile_findKv(const GgufFile *ggufFile, CharString key);
const GgufTensor *GgufFile_findTensor(const GgufFile *ggufFile, CharString name);

//Value helpers: fetch with integer type coercion, so e.g. general.alignment being U32 or U64
// on disk doesn't matter to the caller. Return false if missing or not the requested kind.

Bool GgufValue_asU64(const GgufValue *value, U64 *result);   //Any unsigned int or bool
Bool GgufValue_asI64(const GgufValue *value, I64 *result);   //Any int (unsigned must fit)
Bool GgufValue_asF64(const GgufValue *value, F64 *result);   //F32 or F64
Bool GgufValue_asString(const GgufValue *value, CharString *result);

#ifdef __cplusplus
	}
#endif
