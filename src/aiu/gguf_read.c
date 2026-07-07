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

//aiu/gguf_read.c

#include "aiu/gguf_file.h"
#include "types/base/error.h"
#include "types/base/constants.h"
#include "types/base/string_read_helper.h"
#include "types/container/buffer.h"
#include "types/container/string.h"
#include "types/container/container_types.h"
#include "types/container/ref_ptr.h"
#include "types/container/stream.h"
#include "types/container/list_impl.h"

TListImpl(GgufKv);
TListImpl(GgufTensor);

//All lengths and counts in the file are attacker controlled; every consume below is bounds
// checked against the remaining stream, and every multiply/add that feeds an allocation or an
// offset is overflow checked before use.

//Nested arrays exist in the wild but only shallowly; bound recursion so a crafted file can't
// blow the stack.

#define GGUF_MAX_ARRAY_DEPTH 8

static Bool Gguf_mulU64(U64 a, U64 b, U64 *result, Error *e_rr) {

	Bool s_uccess = true;

	if(b && a > U64_MAX / b)
		retError(clean, Error_overflow(0, a, U64_MAX / b, "Gguf_mulU64() overflow"));

	*result = a * b;

clean:
	return s_uccess;
}

//Consume a string at *it into an owned CharString.
//The length is bounds checked against the remaining stream before allocating.

static Bool Gguf_consumeString(
	StreamCursor *cursor, U64 *it, const Allocator *alloc, CharString *result, Error *e_rr
) {

	Bool s_uccess = true;
	U64 len;

	gotoIfError3(clean, StreamCursor_consumeU64(cursor, it, &len, alloc, e_rr));

	const OxStream *stream = RefPtr_data(cursor->stream, OxStream);

	if(len > stream->size - *it)
		retError(clean, Error_outOfBounds(
			0, len, stream->size - *it, "GgufFile_read() string length exceeds remaining stream"
		));

	if(len) {
		gotoIfError3(clean, CharString_create('\0', len, alloc, result, e_rr));
		gotoIfError3(clean, StreamCursor_consume(cursor, it, result->ptrNonConst, len, alloc, e_rr));
	}

clean:

	if(!s_uccess)
		CharString_free(result, alloc);

	return s_uccess;
}

static void GgufValue_free(GgufValue *value, const Allocator *alloc) {

	if(!value)
		return;

	switch((EGgufValueType) value->type) {

		case EGgufValueType_String:
			CharString_free(&value->string, alloc);
			break;

		case EGgufValueType_Array:

			//Nested arrays own their GgufValue elements; free them recursively first.
			//String and scalar arrays are backed by one bulk buffer (arrayScalars) plus, for
			// strings, a ref table (arrayStrings) whose CharStrings point into it and must NOT
			// be freed individually.

			if(value->arrayType == EGgufValueType_Array) {

				GgufValue *values = (GgufValue*) value->arrayValues.ptrNonConst;

				for(U64 i = 0; i < value->arrayCount && values; ++i)
					GgufValue_free(&values[i], alloc);

				Buffer_free(&value->arrayValues, alloc);
			}

			else {
				Buffer_free(&value->arrayStrings, alloc);   //Null for scalar arrays; ref table for strings
				Buffer_free(&value->arrayScalars, alloc);
			}

			break;

		default:
			break;
	}

	*value = (GgufValue) { 0 };
}

//String array fast path: one bulk read of the whole [U64 len][bytes]* payload, then a table of
// CharString refs into it. Avoids one allocation per element (tokenizer vocabularies are ~49K
// strings; the per-element path made debug builds crawl and stressed the allocator in release).

static Bool Gguf_consumeStringArray(
	StreamCursor *cursor, U64 *it, U64 count, const Allocator *alloc, GgufValue *value, Error *e_rr
) {

	Bool s_uccess = true;

	const OxStream *stream = RefPtr_data(cursor->stream, OxStream);

	//Pass 1: walk the payload with a scratch offset to size it and validate every length,
	// without allocating. Each element is U64 len + len bytes; all bounds checked against the
	// remaining stream, the running total overflow checked.

	U64 scan = *it;
	U64 payload = 0;

	for(U64 i = 0; i < count; ++i) {

		U64 len;
		gotoIfError3(clean, StreamCursor_consumeU64(cursor, &scan, &len, alloc, e_rr));

		if(len > stream->size - scan)
			retError(clean, Error_outOfBounds(
				0, len, stream->size - scan, "GgufFile_read() string array element exceeds remaining stream"
			));

		scan += len;

		//payload += sizeof(U64) + len, overflow checked

		if(payload > U64_MAX - sizeof(U64) - len)
			retError(clean, Error_overflow(0, payload, U64_MAX, "GgufFile_read() string array payload overflow"));

		payload += sizeof(U64) + len;
	}

	//Ref-table size: count CharStrings. count is already bounded by remaining / sizeof(U64).

	U64 tableSize;
	gotoIfError3(clean, Gguf_mulU64(count, sizeof(CharString), &tableSize, e_rr));

	//Bulk read the payload, then build the ref table pointing into it.

	if(payload) {
		gotoIfError3(clean, Buffer_createUninitializedBytes(payload, alloc, &value->arrayScalars, e_rr));
		gotoIfError3(clean, StreamCursor_consumeBuffer(cursor, it, value->arrayScalars, alloc, e_rr));
	}

	if(tableSize)
		gotoIfError3(clean, Buffer_createUninitializedBytes(tableSize, alloc, &value->arrayStrings, e_rr));

	{
		const U8 *ptr = value->arrayScalars.ptr;
		CharString *strings = (CharString*) value->arrayStrings.ptrNonConst;
		U64 off = 0;

		for(U64 i = 0; i < count; ++i) {

			//Length is little endian and may sit at an unaligned offset (previous elements
			// aren't padded), so assemble it byte-wise rather than dereferencing a U64*.

			U64 len = 0;

			for(U8 b = 0; b < 8; ++b)
				len |= (U64) ptr[off + b] << (b * 8);

			off += sizeof(U64);

			strings[i] = CharString_createRefSizedConst((const C8*) (ptr + off), len, false);
			off += len;
		}
	}

	*it = scan;

clean:

	if(!s_uccess) {
		Buffer_free(&value->arrayStrings, alloc);
		Buffer_free(&value->arrayScalars, alloc);
	}

	return s_uccess;
}

static Bool Gguf_consumeValue(
	StreamCursor *cursor,
	U64 *it,
	U32 valueType,
	U64 depth,
	const Allocator *alloc,
	GgufValue *value,
	Error *e_rr
) {

	Bool s_uccess = true;
	*value = (GgufValue) { 0 };

	if(valueType >= EGgufValueType_Count)
		retError(clean, Error_invalidEnum(
			1, valueType, EGgufValueType_Count, "GgufFile_read() invalid metadata value type"
		));

	value->type = (U8) valueType;

	const OxStream *stream = RefPtr_data(cursor->stream, OxStream);

	switch((EGgufValueType) valueType) {

		case EGgufValueType_String:
			gotoIfError3(clean, Gguf_consumeString(cursor, it, alloc, &value->string, e_rr));
			break;

		case EGgufValueType_Array: {

			if(depth >= GGUF_MAX_ARRAY_DEPTH)
				retError(clean, Error_outOfBounds(
					0, depth, GGUF_MAX_ARRAY_DEPTH, "GgufFile_read() array nesting too deep"
				));

			U32 elemType;
			gotoIfError3(clean, StreamCursor_consumeU32(cursor, it, &elemType, alloc, e_rr));
			gotoIfError3(clean, StreamCursor_consumeU64(cursor, it, &value->arrayCount, alloc, e_rr));

			if(elemType >= EGgufValueType_Count)
				retError(clean, Error_invalidEnum(
					1, elemType, EGgufValueType_Count, "GgufFile_read() invalid array element type"
				));

			value->arrayType = (U8) elemType;

			const U64 elemSize = EGgufValueType_size((EGgufValueType) elemType);

			if(elemSize) {    //Scalar array: one bulk read of the tight payload

				U64 payload;
				gotoIfError3(clean, Gguf_mulU64(value->arrayCount, elemSize, &payload, e_rr));

				if(payload > stream->size - *it)
					retError(clean, Error_outOfBounds(
						0, payload, stream->size - *it,
						"GgufFile_read() array payload exceeds remaining stream"
					));

				if(payload) {
					gotoIfError3(clean, Buffer_createUninitializedBytes(payload, alloc, &value->arrayScalars, e_rr));
					gotoIfError3(clean, StreamCursor_consumeBuffer(cursor, it, value->arrayScalars, alloc, e_rr));
				}

				break;
			}

			//A string or nested array's element count is bounded first (each element is at
			// least a U64 on disk), then dispatched.

			if(value->arrayCount > (stream->size - *it) / sizeof(U64))
				retError(clean, Error_outOfBounds(
					0, value->arrayCount, (stream->size - *it) / sizeof(U64),
					"GgufFile_read() array count exceeds remaining stream"
				));

			if(elemType == EGgufValueType_String) {    //Bulk read + ref table, no per-element alloc
				gotoIfError3(clean, Gguf_consumeStringArray(cursor, it, value->arrayCount, alloc, value, e_rr));
				break;
			}

			//Nested array: parse elements out of line into owned GgufValues.

			U64 allocSize;
			gotoIfError3(clean, Gguf_mulU64(value->arrayCount, sizeof(GgufValue), &allocSize, e_rr));
			gotoIfError3(clean, Buffer_createEmptyBytes(allocSize, alloc, &value->arrayValues, e_rr));

			GgufValue *values = (GgufValue*) value->arrayValues.ptrNonConst;

			for(U64 i = 0; i < value->arrayCount; ++i)
				gotoIfError3(clean, Gguf_consumeValue(cursor, it, elemType, depth + 1, alloc, &values[i], e_rr));

			break;
		}

		//Scalars: consume the on disk size, widen to 64 bit in the union so callers don't
		// care which width the file happened to use

		case EGgufValueType_U8: {
			U8 v;
			gotoIfError3(clean, StreamCursor_consumeU8(cursor, it, &v, alloc, e_rr));
			value->u64 = v;
			break;
		}

		case EGgufValueType_Bool: {
			U8 v;
			gotoIfError3(clean, StreamCursor_consumeU8(cursor, it, &v, alloc, e_rr));
			value->u64 = !!v;
			break;
		}

		case EGgufValueType_U16: {
			U16 v;
			gotoIfError3(clean, StreamCursor_consumeU16(cursor, it, &v, alloc, e_rr));
			value->u64 = v;
			break;
		}

		case EGgufValueType_U32: {
			U32 v;
			gotoIfError3(clean, StreamCursor_consumeU32(cursor, it, &v, alloc, e_rr));
			value->u64 = v;
			break;
		}

		case EGgufValueType_U64:
			gotoIfError3(clean, StreamCursor_consumeU64(cursor, it, &value->u64, alloc, e_rr));
			break;

		case EGgufValueType_I8: {
			I8 v;
			gotoIfError3(clean, StreamCursor_consume(cursor, it, &v, 1, alloc, e_rr));
			value->i64 = v;
			break;
		}

		case EGgufValueType_I16: {
			I16 v;
			gotoIfError3(clean, StreamCursor_consume(cursor, it, &v, 2, alloc, e_rr));
			value->i64 = v;
			break;
		}

		case EGgufValueType_I32: {
			I32 v;
			gotoIfError3(clean, StreamCursor_consume(cursor, it, &v, 4, alloc, e_rr));
			value->i64 = v;
			break;
		}

		case EGgufValueType_I64:
			gotoIfError3(clean, StreamCursor_consume(cursor, it, &value->i64, 8, alloc, e_rr));
			break;

		case EGgufValueType_F32:  gotoIfError3(clean, StreamCursor_consume(cursor, it, &value->f32, 4, alloc, e_rr)); break;
		case EGgufValueType_F64:  gotoIfError3(clean, StreamCursor_consume(cursor, it, &value->f64, 8, alloc, e_rr)); break;

		default:
			retError(clean, Error_invalidEnum(1, valueType, EGgufValueType_Count, "GgufFile_read() unreachable"));
	}

clean:

	if(!s_uccess)
		GgufValue_free(value, alloc);

	return s_uccess;
}

static inline EQuantType Gguf_mapGgmlType(U32 ggmlType) {
	switch((EGgmlType) ggmlType) {
		case EGgmlType_F32:  return EQuantType_F32;
		case EGgmlType_F16:  return EQuantType_F16;
		case EGgmlType_Q8_0: return EQuantType_Q80;
		case EGgmlType_Q4_K: return EQuantType_Q4K;
		case EGgmlType_Q6_K: return EQuantType_Q6K;
		default:             return EQuantType_Count;
	}
}

const GgufKv *GgufFile_findKv(const GgufFile *ggufFile, CharString key) {

	if(!ggufFile)
		return NULL;

	for(U64 i = 0; i < ggufFile->metadata.length; ++i)
		if(CharString_equalsStringSensitive(&ggufFile->metadata.ptr[i].key, &key))
			return &ggufFile->metadata.ptr[i];

	return NULL;
}

const GgufTensor *GgufFile_findTensor(const GgufFile *ggufFile, CharString name) {

	if(!ggufFile)
		return NULL;

	for(U64 i = 0; i < ggufFile->tensors.length; ++i)
		if(CharString_equalsStringSensitive(&ggufFile->tensors.ptr[i].name, &name))
			return &ggufFile->tensors.ptr[i];

	return NULL;
}

Bool GgufValue_asU64(const GgufValue *value, U64 *result) {

	if(!value || !result)
		return false;

	switch((EGgufValueType) value->type) {

		case EGgufValueType_U8:  case EGgufValueType_U16:  case EGgufValueType_U32:
		case EGgufValueType_U64: case EGgufValueType_Bool:
			*result = value->u64;
			return true;

		default:
			return false;
	}
}

Bool GgufValue_asI64(const GgufValue *value, I64 *result) {

	if(!value || !result)
		return false;

	switch((EGgufValueType) value->type) {

		case EGgufValueType_I8:  case EGgufValueType_I16: case EGgufValueType_I32:
		case EGgufValueType_I64:
			*result = value->i64;
			return true;

		case EGgufValueType_U8:  case EGgufValueType_U16: case EGgufValueType_U32:
		case EGgufValueType_Bool:
			*result = (I64) value->u64;
			return true;

		case EGgufValueType_U64:

			if(value->u64 > (U64) I64_MAX)
				return false;

			*result = (I64) value->u64;
			return true;

		default:
			return false;
	}
}

Bool GgufValue_asF64(const GgufValue *value, F64 *result) {

	if(!value || !result)
		return false;

	switch((EGgufValueType) value->type) {
		case EGgufValueType_F32: *result = value->f32; return true;
		case EGgufValueType_F64: *result = value->f64; return true;
		default:                 return false;
	}
}

Bool GgufValue_asString(const GgufValue *value, CharString *result) {

	if(!value || !result || value->type != EGgufValueType_String)
		return false;

	*result = value->string;
	return true;
}

void GgufFile_free(GgufFile *ggufFile, const Allocator *alloc) {

	if(!ggufFile)
		return;

	for(U64 i = 0; i < ggufFile->metadata.length; ++i) {
		CharString_free(&ggufFile->metadata.ptrNonConst[i].key, alloc);
		GgufValue_free(&ggufFile->metadata.ptrNonConst[i].value, alloc);
	}

	for(U64 i = 0; i < ggufFile->tensors.length; ++i)
		CharString_free(&ggufFile->tensors.ptrNonConst[i].name, alloc);

	ListGgufKv_free(&ggufFile->metadata, alloc);
	ListGgufTensor_free(&ggufFile->tensors, alloc);

	if(ggufFile->stream)
		RefPtr_dec(&ggufFile->stream);

	*ggufFile = (GgufFile) { 0 };
}

Bool GgufFile_read(
	StreamRef *streamRef,
	U64 *streamOff,
	const Allocator *alloc,
	GgufFile *ggufFile,
	Error *e_rr
) {

	Bool s_uccess = true;
	StreamCursor cursor = (StreamCursor) { 0 };
	GgufKv kv = (GgufKv) { 0 };
	GgufTensor tensor = (GgufTensor) { 0 };

	if(!ggufFile)
		retError(clean, Error_nullPointer(3, "GgufFile_read()::ggufFile is required"));

	if(ggufFile->stream)
		retError(clean, Error_invalidOperation(0, "GgufFile_read()::ggufFile isn't empty, might indicate memleak"));

	if(!streamRef || !streamOff)
		retError(clean, Error_nullPointer(!streamRef ? 0 : 1, "GgufFile_read()::stream and streamOff are required"));

	if(streamRef->refPtrType->typeId != (ETypeId) EContainerTypeId_Stream)
		retError(clean, Error_invalidParameter(0, 0, "GgufFile_read()::stream is of invalid type"));

	const OxStream *stream = RefPtr_data(streamRef, OxStream);

	if(!stream->read)
		retError(clean, Error_invalidParameter(0, 0, "GgufFile_read()::stream is not readable"));

	gotoIfError3(clean, StreamCursor_create(streamRef, 0, false, alloc, &cursor, e_rr));

	const U64 base = *streamOff;
	U64 it = base;

	//Header

	U32 magic, version;
	U64 tensorCount, kvCount;

	gotoIfError3(clean, StreamCursor_consumeU32(&cursor, &it, &magic, alloc, e_rr));

	if(magic != GGUF_MAGIC)
		retError(clean, Error_invalidParameter(0, 0, "GgufFile_read() requires GGUF magicNumber prefix"));

	gotoIfError3(clean, StreamCursor_consumeU32(&cursor, &it, &version, alloc, e_rr));

	//v2 and v3 share the layout we read (v1 used U32 counts/lengths; the v3 bump added a big
	// endian variant, which would show up as a byteswapped magic and is rejected above).

	if(version != EGgufVersion_V2 && version != EGgufVersion_V3)
		retError(clean, Error_invalidParameter(0, 1, "GgufFile_read() unsupported version (2 or 3)"));

	gotoIfError3(clean, StreamCursor_consumeU64(&cursor, &it, &tensorCount, alloc, e_rr));
	gotoIfError3(clean, StreamCursor_consumeU64(&cursor, &it, &kvCount, alloc, e_rr));

	//Counts are attacker controlled; bound them by the smallest possible on disk footprint
	// before allocating lists (KV: key len + value type = 12B, tensor info: 32B).

	const U64 remaining = stream->size - it;

	if(kvCount > remaining / 12)
		retError(clean, Error_outOfBounds(
			0, kvCount, remaining / 12, "GgufFile_read() metadataKvCount exceeds stream size"
		));

	if(tensorCount > remaining / 32)
		retError(clean, Error_outOfBounds(
			0, tensorCount, remaining / 32, "GgufFile_read() tensorCount exceeds stream size"
		));

	ggufFile->version = version;

	gotoIfError3(clean, ListGgufKv_reserve(&ggufFile->metadata, kvCount, alloc, e_rr));
	gotoIfError3(clean, ListGgufTensor_reserve(&ggufFile->tensors, tensorCount, alloc, e_rr));

	//Metadata KVs

	for(U64 i = 0; i < kvCount; ++i) {

		gotoIfError3(clean, Gguf_consumeString(&cursor, &it, alloc, &kv.key, e_rr));

		if(GgufFile_findKv(ggufFile, kv.key))
			retError(clean, Error_alreadyDefined(0, "GgufFile_read() duplicate metadata key"));

		U32 valueType;
		gotoIfError3(clean, StreamCursor_consumeU32(&cursor, &it, &valueType, alloc, e_rr));
		gotoIfError3(clean, Gguf_consumeValue(&cursor, &it, valueType, 0, alloc, &kv.value, e_rr));
		gotoIfError3(clean, ListGgufKv_pushBack(&ggufFile->metadata, kv, alloc, e_rr));

		kv = (GgufKv) { 0 };    //Owned by the list now
	}

	//Alignment of the data section (must be a power of two; must keep the 4 byte tensor base
	// contract from quant.h, so alignments of 1 and 2 are rejected as unsupported)

	U32 alignment = GGUF_DEFAULT_ALIGNMENT;
	const GgufKv *alignKv = GgufFile_findKv(ggufFile, CharString_createRefSizedConst("general.alignment", 17, false));

	if(alignKv) {

		U64 alignment64;

		if(!GgufValue_asU64(&alignKv->value, &alignment64))
			retError(clean, Error_invalidParameter(0, 2, "GgufFile_read() general.alignment must be an unsigned int"));

		if(!alignment64 || (alignment64 & (alignment64 - 1)) || alignment64 > U32_MAX)
			retError(clean, Error_invalidParameter(0, 2, "GgufFile_read() general.alignment must be a power of two"));

		if(alignment64 < 4)
			retError(clean, Error_unsupportedOperation(
				0, "GgufFile_read() general.alignment <4 breaks the 4 byte tensor alignment contract"
			));

		alignment = (U32) alignment64;
	}

	ggufFile->alignment = alignment;

	//Tensor infos

	for(U64 i = 0; i < tensorCount; ++i) {

		tensor = (GgufTensor) { 0 };

		gotoIfError3(clean, Gguf_consumeString(&cursor, &it, alloc, &tensor.name, e_rr));

		if(GgufFile_findTensor(ggufFile, tensor.name))
			retError(clean, Error_alreadyDefined(1, "GgufFile_read() duplicate tensor name"));

		U32 nDims;
		gotoIfError3(clean, StreamCursor_consumeU32(&cursor, &it, &nDims, alloc, e_rr));

		if(!nDims || nDims > GGUF_MAX_DIMS)
			retError(clean, Error_invalidParameter(0, 3, "GgufFile_read() tensor nDims must be 1..4"));

		tensor.nDims = (U8) nDims;
		tensor.elements = 1;

		for(U8 j = 0; j < GGUF_MAX_DIMS; ++j)
			tensor.dims[j] = 1;

		for(U8 j = 0; j < nDims; ++j) {

			gotoIfError3(clean, StreamCursor_consumeU64(&cursor, &it, &tensor.dims[j], alloc, e_rr));

			if(!tensor.dims[j])
				retError(clean, Error_invalidParameter(0, 4, "GgufFile_read() tensor dim of 0"));

			gotoIfError3(clean, Gguf_mulU64(tensor.elements, tensor.dims[j], &tensor.elements, e_rr));
		}

		gotoIfError3(clean, StreamCursor_consumeU32(&cursor, &it, &tensor.ggmlType, alloc, e_rr));
		gotoIfError3(clean, StreamCursor_consumeU64(&cursor, &it, &tensor.offset, alloc, e_rr));

		tensor.quantType = (U8) Gguf_mapGgmlType(tensor.ggmlType);

		//Unsupported block types are per tensor, not per file: keep the descriptor
		// (name/dims/ggmlType) with streamLen 0 so the loader can report/skip it.

		gotoIfError3(clean, ListGgufTensor_pushBack(&ggufFile->tensors, tensor, alloc, e_rr));

		tensor = (GgufTensor) { 0 };   //Owned by the list now
	}

	//Data section: starts after the infos, aligned to 'alignment' relative to the gguf start.
	//A file without tensor data may end right after the infos, before the alignment boundary.

	const U64 infoEnd = it - base;
	U64 dataStartRel = (infoEnd + alignment - 1) / alignment * alignment;

	if(base + dataStartRel > stream->size)
		dataStartRel = stream->size - base;

	ggufFile->dataStart = base + dataStartRel;
	const U64 dataSize = stream->size - ggufFile->dataStart;

	//Resolve and validate tensor data locations.
	//Offsets must be aligned, in bounds, and monotonically increasing; supported tensors also
	// get a size (this is where quant.h plugs in) and an overlap check against the previous supported tensor
	// dims[0] (the row) must be whole blocks.

	U64 prevOffset = 0, prevEndSupported = 0;

	for(U64 i = 0; i < tensorCount; ++i) {

		GgufTensor *tensori = &ggufFile->tensors.ptrNonConst[i];

		if(tensori->offset % alignment)
			retError(clean, Error_invalidParameter(0, 5, "GgufFile_read() tensor offset is misaligned"));

		if(i && tensori->offset <= prevOffset)
			retError(clean, Error_invalidParameter(0, 5, "GgufFile_read() tensor offsets must be increasing"));

		prevOffset = tensori->offset;

		if(tensori->offset > dataSize)
			retError(clean, Error_outOfBounds(1, tensori->offset, dataSize, "GgufFile_read() tensor offset out of bounds"));

		tensori->streamOff = ggufFile->dataStart + tensori->offset;

		if(tensori->quantType == EQuantType_Count)    //Unknown block size; can't validate further
			continue;

		const EQuantType quantType = (EQuantType) tensori->quantType;
		const U64 blockElements = Quant_blockNumElements(quantType);

		if(tensori->dims[0] % blockElements)
			retError(clean, Error_invalidParameter(
				0, 6, "GgufFile_read() tensor row length isn't a multiple of the block element count"
			));

		const U64 size = Quant_size(quantType, tensori->elements);

		if(!size || size > dataSize - tensori->offset)
			retError(clean, Error_outOfBounds(
				1, size, dataSize - tensori->offset, "GgufFile_read() tensor data out of bounds"
			));

		if(tensori->offset < prevEndSupported)
			retError(clean, Error_invalidParameter(0, 7, "GgufFile_read() tensor data overlaps previous tensor"));

		prevEndSupported = tensori->offset + size;
		tensori->streamLen = size;
	}

	//gguf doesn't encode its total size; the data section runs to the end of the stream

	RefPtr_inc(streamRef);
	ggufFile->stream = streamRef;
	*streamOff = stream->size;

clean:

	if(!s_uccess) {
		CharString_free(&kv.key, alloc);
		GgufValue_free(&kv.value, alloc);
		CharString_free(&tensor.name, alloc);
		GgufFile_free(ggufFile, alloc);
	}

	StreamCursor_close(&cursor, alloc);
	return s_uccess;
}

Bool GgufTensor_readData(
	const GgufFile *ggufFile,
	const GgufTensor *tensor,
	const Allocator *alloc,
	Buffer *result,
	Error *e_rr
) {

	Bool s_uccess = true;

	if(!ggufFile || !ggufFile->stream || !tensor || !result)
		retError(clean, Error_nullPointer(
			!ggufFile || !ggufFile->stream ? 0 : (!tensor ? 1 : 3),
			"GgufTensor_readData()::ggufFile, tensor and result are required"
		));

	if(result->ptr)
		retError(clean, Error_invalidOperation(
			0, "GgufTensor_readData()::result isn't empty, might indicate memleak"
		));

	if(tensor->quantType >= EQuantType_Count || !tensor->streamLen)
		retError(clean, Error_unsupportedOperation(
			0, "GgufTensor_readData() tensor has an unsupported block type"
		));

	//Quant_allocSize = tight size padded to a multiple of 16B, per the quant.h contract
	// (tail loads may read up to 3B past the tight size). Pad is read, never interpreted,
	// but zero it anyway so the buffer contents are deterministic.

	const U64 allocSize = Quant_allocSize((EQuantType) tensor->quantType, tensor->elements);

	if(!allocSize)
		retError(clean, Error_invalidState(0, "GgufTensor_readData() couldn't size the tensor"));

	gotoIfError3(clean, Buffer_createUninitializedBytes(allocSize, alloc, result, e_rr));

	for(U64 i = tensor->streamLen; i < allocSize; ++i)
		result->ptrNonConst[i] = 0;

	OxStream *stream = RefPtr_data(ggufFile->stream, OxStream);

	gotoIfError3(clean, stream->read(
		stream,
		tensor->streamOff,
		tensor->streamLen,
		Buffer_createRef(result->ptrNonConst, tensor->streamLen),
		alloc,
		e_rr
	));

clean:

	if(!s_uccess)
		Buffer_free(result, alloc);

	return s_uccess;
}
