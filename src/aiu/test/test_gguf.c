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

//aiu/test/test_gguf.c

//GGUF parser unit tests.
//
//What is covered:
//  1. Valid file   - hand-crafted in-memory gguf (v3): scalar/string/array/string-array KVs,
//                    an F32 and a Q4_K tensor; metadata, lookups and value coercion; the Q4_K
//                    tensor dequantized through the parsed descriptor matches Quant_dequantize
//                    on the raw source bytes.
//  2. Tolerance    - unsupported ggml tensor types parse per tensor (quantType = Count, data
//                    unset) instead of failing the file.
//  3. Malformed    - bad magic, bad version, truncated header, oversized string length,
//                    non-power-of-two alignment, misaligned/overlapping/out-of-bounds tensor
//                    offsets, duplicate tensor names, row not a multiple of the block size.
//  4. Optional real gguf

#include "test_aiu_shared.h"
#include "aiu/gguf_file.h"
#include "types/test/test.h"
#include "types/container/test/basic_alloc.h"
#include "types/container/memory_stream.h"
#include "types/container/buffer.h"
#include "types/container/ref_ptr.h"
#include "types/math/rand.h"
#include "types/math/flp.h"
#include "types/base/mathf.h"
#include "types/base/constants.h"
#include "platforms/file.h"

//Wrap the writer's bytes in a read only MemoryStream (region ref; w must outlive the stream)

static Bool GgufWriter_stream(
	const void *data, U64 len, const RefPtrType *type, MemoryStreamRef **stream, Error *e_rr
) {
	return MemoryStream_createFromBufferRegion(
		Buffer_createRefConst(data, len), 0, len, EMemoryStreamFlags_None, type, stream, e_rr
	);
}

//Tiny append-only writer; alignment 16 so the base pointer satisfies the 4 byte contract

typedef struct GgufWriter {
	_Alignas(16) U8 data[4096];
	U64 len;
} GgufWriter;

static void GgufWriter_bytes(GgufWriter *w, const void *v, U64 len) {

	for(U64 i = 0; i < len; ++i)
		w->data[w->len + i] = ((const U8*)v)[i];

	w->len += len;
}

static void GgufWriter_u8(GgufWriter *w, U8 v)   { GgufWriter_bytes(w, &v, 1); }
static void GgufWriter_u32(GgufWriter *w, U32 v) { GgufWriter_bytes(w, &v, 4); }
static void GgufWriter_u64(GgufWriter *w, U64 v) { GgufWriter_bytes(w, &v, 8); }
static void GgufWriter_f32(GgufWriter *w, F32 v) { GgufWriter_bytes(w, &v, 4); }

static void GgufWriter_str(GgufWriter *w, const C8 *s) {

	U64 len = 0;
	while(s[len]) ++len;

	GgufWriter_u64(w, len);
	GgufWriter_bytes(w, s, len);
}

static void GgufWriter_align(GgufWriter *w, U64 alignment) {
	while(w->len % alignment)
		GgufWriter_u8(w, 0);
}

static void GgufWriter_header(GgufWriter *w, U32 version, U64 tensorCount, U64 kvCount) {
	w->len = 0;
	GgufWriter_u32(w, GGUF_MAGIC);
	GgufWriter_u32(w, version);
	GgufWriter_u64(w, tensorCount);
	GgufWriter_u64(w, kvCount);
}

static void GgufWriter_tensorInfo(GgufWriter *w, const C8 *name, U64 rows, U64 cols, U32 ggmlType, U64 offset) {

	GgufWriter_str(w, name);

	if(rows > 1) {
		GgufWriter_u32(w, 2);
		GgufWriter_u64(w, cols);    //dims[0] = fastest moving = row length
		GgufWriter_u64(w, rows);
	}

	else {
		GgufWriter_u32(w, 1);
		GgufWriter_u64(w, cols);
	}

	GgufWriter_u32(w, ggmlType);
	GgufWriter_u64(w, offset);
}

static CharString Gguf_key(const C8 *s) {

	U64 len = 0;
	while(s[len]) ++len;

	return CharString_createRefSizedConst(s, len, true);
}

//1 + 2: a full valid file, plus one unsupported tensor type

static void Test_ggufValid(Test *t) {

	Test_setModule(t, "GGUF valid file");

	GgufWriter w;
	GgufWriter_header(&w, EGgufVersion_V3, 3, 4);

	//KVs: alignment (U32), a string, an I32 array, a string array

	GgufWriter_str(&w, "general.alignment");
	GgufWriter_u32(&w, EGgufValueType_U32);
	GgufWriter_u32(&w, 32);

	GgufWriter_str(&w, "general.name");
	GgufWriter_u32(&w, EGgufValueType_String);
	GgufWriter_str(&w, "aiu test model");

	GgufWriter_str(&w, "test.ints");
	GgufWriter_u32(&w, EGgufValueType_Array);
	GgufWriter_u32(&w, EGgufValueType_I32);
	GgufWriter_u64(&w, 3);
	GgufWriter_u32(&w, (U32) -1);
	GgufWriter_u32(&w, 2);
	GgufWriter_u32(&w, 3);

	GgufWriter_str(&w, "test.strings");
	GgufWriter_u32(&w, EGgufValueType_Array);
	GgufWriter_u32(&w, EGgufValueType_String);
	GgufWriter_u64(&w, 3);
	GgufWriter_str(&w, "alpha");
	GgufWriter_str(&w, "beta");
	GgufWriter_str(&w, "");     //Empty element: zero-length in the bulk-read + ref-table path

	//Tensors: F32 [4], Q4_K [2 x 256], and one of an unsupported ggml type (Q5_K = 13).
	//Data section: F32 at 0 (16B), Q4_K at 32 (2 blocks = 288B), unknown at 320.

	GgufWriter_tensorInfo(&w, "bias", 1, 4, EGgmlType_F32, 0);
	GgufWriter_tensorInfo(&w, "weight", 2, 256, EGgmlType_Q4_K, 32);
	GgufWriter_tensorInfo(&w, "exotic", 1, 256, 13, 320);

	GgufWriter_align(&w, 32);
	const U64 dataStart = w.len;

	const F32 biasData[4] = { 1, -2, 0.5f, 4 };

	for(U64 i = 0; i < 4; ++i)
		GgufWriter_f32(&w, biasData[i]);
	GgufWriter_align(&w, 32);

	U8 q4kData[288];
	U32 seed = Random_seed(0x66F0E, 42);

	for(U64 i = 0; i < sizeof(q4kData); ++i)
		q4kData[i] = (U8) (Random_sample(&seed) * 256);

	GgufWriter_bytes(&w, q4kData, sizeof(q4kData));
	GgufWriter_align(&w, 32);
	GgufWriter_bytes(&w, q4kData, 16);    //The "exotic" tensor's (opaque) bytes

	const RefPtrType memStreamType = MemoryStream_makeType(t->alloc);
	MemoryStreamRef *ms = NULL;
	U64 streamOff = 0;

	GgufFile file = (GgufFile) { 0 };
	Test_assert(t, "stream", GgufWriter_stream(w.data, w.len, &memStreamType, &ms, &t->err));
	Test_assert(t, "read", GgufFile_read(ms, &streamOff, t->alloc, &file, &t->err));
	Test_assert(t, "streamOff advanced to end", streamOff == w.len);

	Test_assert(t, "version", file.version == EGgufVersion_V3);
	Test_assert(t, "alignment", file.alignment == 32);
	Test_assert(t, "kv count", file.metadata.length == 4);
	Test_assert(t, "tensor count", file.tensors.length == 3);
	Test_assert(t, "data section", file.dataStart == dataStart);

	//Metadata

	const GgufKv *kv = GgufFile_findKv(&file, Gguf_key("general.name"));
	CharString str;

	Test_assert(t, "string kv found", kv);
	Test_assert(t, "string kv", kv && GgufValue_asString(&kv->value, &str) && CharString_length(str) == 14);
	Test_assert(t, "string kv content", kv && str.ptr[0] == 'a' && str.ptr[13] == 'l');

	kv = GgufFile_findKv(&file, Gguf_key("test.ints"));

	Test_assert(
		t, "scalar array kv",
		kv && kv->value.type == EGgufValueType_Array && kv->value.arrayType == EGgufValueType_I32 &&
		kv->value.arrayCount == 3 && Buffer_length(kv->value.arrayScalars) == 12
	);

	Test_assert(
		t, "scalar array payload",
		kv && ((const I32*)kv->value.arrayScalars.ptr)[0] == -1 && ((const I32*)kv->value.arrayScalars.ptr)[2] == 3
	);

	kv = GgufFile_findKv(&file, Gguf_key("test.strings"));
	const CharString *strs = kv ? GgufValue_arrayStrings(&kv->value) : NULL;

	Test_assert(t, "string array kv", strs && kv->value.arrayCount == 3);

	Test_assert(
		t, "string array content",
		strs && CharString_length(strs[0]) == 5 && strs[0].ptr[0] == 'a' &&
		CharString_length(strs[1]) == 4 && strs[1].ptr[0] == 'b' &&
		CharString_length(strs[2]) == 0    //Empty element parses to a zero-length ref
	);

	Test_assert(t, "missing kv", !GgufFile_findKv(&file, Gguf_key("nope")));

	//Alignment coercion helper

	kv = GgufFile_findKv(&file, Gguf_key("general.alignment"));
	U64 align64 = 0;
	Test_assert(t, "asU64 coercion", kv && GgufValue_asU64(&kv->value, &align64) && align64 == 32);

	//Tensors

	const GgufTensor *bias = GgufFile_findTensor(&file, Gguf_key("bias"));

	Test_assert(
		t, "f32 tensor",
		bias && bias->quantType == EQuantType_F32 && bias->nDims == 1 && bias->dims[0] == 4 &&
		bias->elements == 4 && bias->streamOff == dataStart && bias->streamLen == 16
	);

	if(bias) {

		Buffer biasBuf = Buffer_createNull();
		Test_assert(t, "f32 tensor read", GgufTensor_readData(&file, bias, t->alloc, &biasBuf, &t->err));

		Test_assert(
			t, "f32 tensor data",
			Buffer_length(biasBuf) == 16 &&    //Already a multiple of 16: no padding added
			((const F32*)biasBuf.ptr)[1] == -2 && ((const F32*)biasBuf.ptr)[3] == 4
		);

		Buffer_free(&biasBuf, t->alloc);
	}

	const GgufTensor *weight = GgufFile_findTensor(&file, Gguf_key("weight"));

	Test_assert(
		t, "q4k tensor",
		weight && weight->quantType == EQuantType_Q4K && weight->nDims == 2 &&
		weight->dims[0] == 256 && weight->dims[1] == 2 && weight->elements == 512 &&
		weight->streamOff == dataStart + 32 && weight->streamLen == 288
	);

	//Dequantizing through the parsed descriptor must match dequantizing the raw source bytes

	if(weight) {

		F32 viaFile[512], viaRaw[512];

		Buffer weightBuf = Buffer_createNull();
		Test_assert(t, "q4k tensor read", GgufTensor_readData(&file, weight, t->alloc, &weightBuf, &t->err));
		Test_assert(t, "q4k alloc padded to 16", Buffer_length(weightBuf) == 288);

		Test_assert(t, "dequantize via descriptor", Quant_dequantize(
			(EQuantType) weight->quantType, weightBuf.ptr, weight->elements, viaFile, &t->err
		));

		Test_assert(t, "dequantize raw", Quant_dequantize(EQuantType_Q4K, q4kData, 512, viaRaw, &t->err));

		//Compare bitwise: random F16 scales can produce NaNs and NaN != NaN would false-fail

		const U32 *a = (const U32*) viaFile, *b = (const U32*) viaRaw;
		Bool same = true;

		for(U64 i = 0; i < 512; ++i)
			same &= a[i] == b[i];

		Test_assert(t, "dequantize matches raw bytes", same);
		Buffer_free(&weightBuf, t->alloc);
	}

	//Unsupported type is tolerated per tensor

	const GgufTensor *exotic = GgufFile_findTensor(&file, Gguf_key("exotic"));

	Test_assert(
		t, "unsupported tensor tolerated",
		exotic && exotic->quantType == EQuantType_Count && exotic->ggmlType == 13 && !exotic->streamLen
	);

	//Loading an unsupported tensor must fail cleanly

	if(exotic) {
		Buffer exoticBuf = Buffer_createNull();
		Error exoticErr = (Error) { 0 };
		Test_assert(t, "unsupported tensor load fails", !GgufTensor_readData(&file, exotic, t->alloc, &exoticBuf, &exoticErr));
	}

	GgufFile_free(&file, t->alloc);
	Test_assert(t, "free clears", !file.metadata.ptr && !file.tensors.ptr && !file.stream);
	RefPtr_dec(&ms);
}

//3: malformed inputs. Each case builds a small file and asserts the parse fails cleanly.

static Bool Test_ggufExpectFail(Test *t, const GgufWriter *w) {

	GgufFile file = (GgufFile) { 0 };
	Error err = (Error) { 0 };       //Local: failures here are expected, don't report them

	const RefPtrType memStreamType = MemoryStream_makeType(t->alloc);
	MemoryStreamRef *ms = NULL;
	U64 streamOff = 0;

	if(!GgufWriter_stream(w->data, w->len, &memStreamType, &ms, &t->err))
		return false;

	const Bool ok = GgufFile_read(ms, &streamOff, t->alloc, &file, &err);

	if(ok)
		GgufFile_free(&file, t->alloc);

	RefPtr_dec(&ms);
	return !ok;
}

static void Test_ggufMalformed(Test *t) {

	Test_setModule(t, "GGUF malformed inputs");

	GgufWriter w;

	//Bad magic

	GgufWriter_header(&w, EGgufVersion_V3, 0, 0);
	w.data[0] = 'X';
	Test_assert(t, "bad magic", Test_ggufExpectFail(t, &w));

	//Bad version

	GgufWriter_header(&w, 1, 0, 0);
	Test_assert(t, "bad version", Test_ggufExpectFail(t, &w));

	//Truncated header

	GgufWriter_header(&w, EGgufVersion_V3, 0, 0);
	w.len = 12;
	Test_assert(t, "truncated header", Test_ggufExpectFail(t, &w));

	//Oversized string length (larger than the remaining file)

	GgufWriter_header(&w, EGgufVersion_V3, 0, 1);
	GgufWriter_u64(&w, 0xFFFFFFFFFFFF);
	Test_assert(t, "oversized string len", Test_ggufExpectFail(t, &w));

	//Oversized counts

	GgufWriter_header(&w, EGgufVersion_V3, 0, 0xFFFFFFFFFFFF);
	Test_assert(t, "oversized kv count", Test_ggufExpectFail(t, &w));

	GgufWriter_header(&w, EGgufVersion_V3, 0xFFFFFFFFFFFF, 0);
	Test_assert(t, "oversized tensor count", Test_ggufExpectFail(t, &w));

	//Oversized scalar array count

	GgufWriter_header(&w, EGgufVersion_V3, 0, 1);
	GgufWriter_str(&w, "a");
	GgufWriter_u32(&w, EGgufValueType_Array);
	GgufWriter_u32(&w, EGgufValueType_I32);
	GgufWriter_u64(&w, 0xFFFFFFFFFFFF);
	Test_assert(t, "oversized array count", Test_ggufExpectFail(t, &w));

	//Duplicate key

	GgufWriter_header(&w, EGgufVersion_V3, 0, 2);

	for(U64 i = 0; i < 2; ++i) {
		GgufWriter_str(&w, "dup");
		GgufWriter_u32(&w, EGgufValueType_U8);
		GgufWriter_u8(&w, 1);
	}

	Test_assert(t, "duplicate key", Test_ggufExpectFail(t, &w));

	//Non power of two alignment

	GgufWriter_header(&w, EGgufVersion_V3, 0, 1);
	GgufWriter_str(&w, "general.alignment");
	GgufWriter_u32(&w, EGgufValueType_U32);
	GgufWriter_u32(&w, 24);
	Test_assert(t, "bad alignment", Test_ggufExpectFail(t, &w));

	//Tensor cases share a preamble: header + no KVs + one tensor info

	//Misaligned offset

	GgufWriter_header(&w, EGgufVersion_V3, 1, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 32, EGgmlType_Q8_0, 12);
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 64; ++i) GgufWriter_u8(&w, 0);
	Test_assert(t, "misaligned tensor offset", Test_ggufExpectFail(t, &w));

	//Out of bounds data

	GgufWriter_header(&w, EGgufVersion_V3, 1, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 256, EGgmlType_Q4_K, 0);
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 32; ++i) GgufWriter_u8(&w, 0);    //144B needed, 32 present
	Test_assert(t, "tensor data out of bounds", Test_ggufExpectFail(t, &w));

	//Row length not a multiple of the block element count

	GgufWriter_header(&w, EGgufVersion_V3, 1, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 100, EGgmlType_Q4_K, 0);
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 144; ++i) GgufWriter_u8(&w, 0);
	Test_assert(t, "row not whole blocks", Test_ggufExpectFail(t, &w));

	//Duplicate tensor name

	GgufWriter_header(&w, EGgufVersion_V3, 2, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 4, EGgmlType_F32, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 4, EGgmlType_F32, 32);
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 64; ++i) GgufWriter_u8(&w, 0);
	Test_assert(t, "duplicate tensor name", Test_ggufExpectFail(t, &w));

	//Overlapping tensors (offsets must be increasing and non overlapping)

	GgufWriter_header(&w, EGgufVersion_V3, 2, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 32, EGgmlType_F32, 0);    //128B: 0..128
	GgufWriter_tensorInfo(&w, "b", 1, 32, EGgmlType_F32, 32);   //Starts inside a
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 256; ++i) GgufWriter_u8(&w, 0);
	Test_assert(t, "overlapping tensors", Test_ggufExpectFail(t, &w));

	//Non increasing offsets

	GgufWriter_header(&w, EGgufVersion_V3, 2, 0);
	GgufWriter_tensorInfo(&w, "a", 1, 4, EGgmlType_F32, 32);
	GgufWriter_tensorInfo(&w, "b", 1, 4, EGgmlType_F32, 0);
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 64; ++i) GgufWriter_u8(&w, 0);
	Test_assert(t, "non increasing offsets", Test_ggufExpectFail(t, &w));

	//Truncation sweep: every prefix of a known-valid file must parse or fail cleanly
	// (no crash, no leak). Run under ASan/UBSan this doubles as structured fuzzing.

	GgufWriter_header(&w, EGgufVersion_V3, 1, 2);

	GgufWriter_str(&w, "general.name");
	GgufWriter_u32(&w, EGgufValueType_String);
	GgufWriter_str(&w, "trunc");

	GgufWriter_str(&w, "test.strings");
	GgufWriter_u32(&w, EGgufValueType_Array);
	GgufWriter_u32(&w, EGgufValueType_String);
	GgufWriter_u64(&w, 2);
	GgufWriter_str(&w, "alpha");
	GgufWriter_str(&w, "beta");

	GgufWriter_tensorInfo(&w, "a", 1, 4, EGgmlType_F32, 0);
	GgufWriter_align(&w, 32);
	for(U64 i = 0; i < 16; ++i) GgufWriter_u8(&w, 0);

	const RefPtrType sweepStreamType = MemoryStream_makeType(t->alloc);
	Bool sweepOk = true;

	for(U64 i = 1; i < w.len; ++i) {

		GgufFile truncated = (GgufFile) { 0 };
		Error truncErr = (Error) { 0 };
		MemoryStreamRef *truncMs = NULL;
		U64 truncOff = 0;

		if(!GgufWriter_stream(w.data, i, &sweepStreamType, &truncMs, &t->err)) {
			sweepOk = false;
			break;
		}

		if(GgufFile_read(truncMs, &truncOff, t->alloc, &truncated, &truncErr)) {
			sweepOk &= i >= 24;    //Nothing shorter than the header may parse
			GgufFile_free(&truncated, t->alloc);
		}

		RefPtr_dec(&truncMs);
	}

	Test_assert(t, "truncation sweep", sweepOk);

	//API misuse

	GgufFile file = (GgufFile) { 0 };
	Error err = (Error) { 0 };
	GgufWriter_header(&w, EGgufVersion_V3, 0, 0);

	const RefPtrType memStreamType = MemoryStream_makeType(t->alloc);
	MemoryStreamRef *ms = NULL;
	U64 streamOff = 0;

	Test_assert(t, "stream", GgufWriter_stream(w.data, w.len, &memStreamType, &ms, &t->err));
	Test_assert(t, "null ggufFile", !GgufFile_read(ms, &streamOff, t->alloc, NULL, &err));
	Test_assert(t, "null streamOff", !GgufFile_read(ms, NULL, t->alloc, &file, &err));
	Test_assert(t, "null stream", !GgufFile_read(NULL, &streamOff, t->alloc, &file, &err));

	//Minimal empty file parses

	streamOff = 0;
	Test_assert(t, "empty file parses", GgufFile_read(ms, &streamOff, t->alloc, &file, &t->err));
	Test_assert(t, "empty file contents", !file.metadata.length && !file.tensors.length);
	GgufFile_free(&file, t->alloc);
	RefPtr_dec(&ms);
}

// -- 3. Real model file: SmolLM2-135M-Instruct-Q4_K_S ---------------------------
//
// Place the file at src/aiu/test/models/SmolLM2-135M-Instruct-Q4_K_S.gguf  in the repo root
// (download from bartowski/SmolLM2-135M-Instruct-GGUF on HuggingFace, ~102 MB).
// The test skips gracefully when the file is absent.
//
// What is validated:
//   - Metadata KVs match known model config (architecture, dims, head counts, etc.)
//   - 272 tensors total (tied embeddings: llama.cpp omits output.weight)
//   - Norm tensors are F32 with the expected element count
//   - Weight tensors exist with the expected element count and are not F32
//     (exact quant format is recipe-dependent and not asserted here;
//      unsupported-type tolerance is covered by Test_ggufValid)
//   - output_norm.weight (576 F32 values) loads cleanly; all values finite and in
//     a sane range [-100, 100] (trained RMS norm scales are typically near 1)

//Return the KV value as U64 or 0 on miss/wrong type for compact assert expressions

static U64 Test_ggufModelKvU64(const GgufFile *file, const C8 *key) {

	U64 keyLen = 0;
	while(key[keyLen]) ++keyLen;

	const GgufKv *kv = GgufFile_findKv(file, CharString_createRefSizedConst(key, keyLen, false));
	U64 v = 0;
	return (kv && GgufValue_asU64(&kv->value, &v)) ? v : 0;
}

static Bool Test_ggufModelKvStr(const GgufFile *file, const C8 *key, const C8 *expected) {

	U64 keyLen = 0;
	while(key[keyLen]) ++keyLen;

	const GgufKv *kv = GgufFile_findKv(file, CharString_createRefSizedConst(key, keyLen, false));

	if(!kv)
		return false;

	CharString str;

	if(!GgufValue_asString(&kv->value, &str))
		return false;

	U64 expLen = 0;
	while(expected[expLen]) ++expLen;

	return
		CharString_length(str) == expLen &&
		Buffer_eq(Buffer_createRefConst(str.ptr, expLen), Buffer_createRefConst(expected, expLen));
}

static const GgufTensor *Test_ggufModelTensor(const GgufFile *file, const C8 *name) {

	U64 len = 0;
	while(name[len]) ++len;

	return GgufFile_findTensor(file, CharString_createRefSizedConst(name, len, false));
}

static void Test_ggufModel(Test *t) {

	Test_setModule(t, "GGUF SmolLM2-135M model file");

	const C8 *pathCStr = "src/aiu/test/models/SmolLM2-135M-Instruct-Q4_K_S.gguf";
	const CharString path = CharString_createRefCStrConst(pathCStr);

	if(!File_hasFile(&path, t->alloc)) {
		Test_print(t, "SKIP: src/aiu/test/models/SmolLM2-135M-Instruct-Q4_K_S.gguf not found");
		Test_print(t, "      Download from bartowski/SmolLM2-135M-Instruct-GGUF on HuggingFace");
		return;
	}

	const RefPtrType fileHandleType = FileHandle_makeType(t->alloc);
	const RefPtrType streamType     = FileStream_makeType(t->alloc);
	StreamRef *stream = NULL;

	if(!Test_assert(t, "open", File_openStream(
		&path, 50 * MS, EFileOpenType_Read, false, &fileHandleType, &streamType, &stream, &t->err
	)))
		return;

	GgufFile file = (GgufFile) { 0 };
	U64 streamOff = 0;

	if(!Test_assert(t, "parse", GgufFile_read(stream, &streamOff, t->alloc, &file, &t->err))) {
		RefPtr_dec(&stream);
		return;
	}

	//Architecture metadata, sourced from the published SmolLM2-135M config.json

	Test_assert(t, "architecture",        Test_ggufModelKvStr(&file, "general.architecture",              "llama"));
	Test_assert(t, "context_length",      Test_ggufModelKvU64(&file, "llama.context_length")              == 8192);
	Test_assert(t, "embedding_length",    Test_ggufModelKvU64(&file, "llama.embedding_length")            == 576);
	Test_assert(t, "block_count",         Test_ggufModelKvU64(&file, "llama.block_count")                 == 30);
	Test_assert(t, "feed_forward_length", Test_ggufModelKvU64(&file, "llama.feed_forward_length")         == 1536);
	Test_assert(t, "head_count",          Test_ggufModelKvU64(&file, "llama.attention.head_count")        == 9);
	Test_assert(t, "head_count_kv",       Test_ggufModelKvU64(&file, "llama.attention.head_count_kv")     == 3);
	Test_assert(t, "rope_dim",            Test_ggufModelKvU64(&file, "llama.rope.dimension_count")        == 64);
	Test_assert(t, "tokenizer_model",     Test_ggufModelKvStr(&file, "tokenizer.ggml.model",              "gpt2"));

	//SmolLM2-135M uses tied embeddings (tie_word_embeddings: true) llama.cpp omits output.weight.
	//  token_embd.weight + output_norm.weight = 2 top-level, plus 30 * 9 = 270 per-layer = 272 total.

	Test_assert(t, "tensor_count", file.tensors.length == 272);

	//Specific tensors, types and element counts from the model architecture.
	//Norm weights must be F32. Weight tensors must not be F32 (some quantized type);
	//the exact format is recipe-dependent (Q4_K, Q6_K, Q5_K, IQ* all appear in practice).

	const GgufTensor *outputNorm = Test_ggufModelTensor(&file, "output_norm.weight");
	const GgufTensor *attnNorm0  = Test_ggufModelTensor(&file, "blk.0.attn_norm.weight");
	const GgufTensor *ffnNorm0   = Test_ggufModelTensor(&file, "blk.0.ffn_norm.weight");
	const GgufTensor *attnQ0     = Test_ggufModelTensor(&file, "blk.0.attn_q.weight");
	const GgufTensor *attnK0     = Test_ggufModelTensor(&file, "blk.0.attn_k.weight");
	const GgufTensor *ffnGate0   = Test_ggufModelTensor(&file, "blk.0.ffn_gate.weight");
	const GgufTensor *tokenEmbd  = Test_ggufModelTensor(&file, "token_embd.weight");

	//Norm weights: F32 vectors of length embedding_length = 576

	Test_assert(t, "output_norm exists",   outputNorm && outputNorm->quantType == EQuantType_F32 && outputNorm->elements == 576);
	Test_assert(t, "attn_norm.0 exists",   attnNorm0  && attnNorm0->quantType  == EQuantType_F32 && attnNorm0->elements  == 576);
	Test_assert(t, "ffn_norm.0 exists",    ffnNorm0   && ffnNorm0->quantType   == EQuantType_F32 && ffnNorm0->elements   == 576);

	//Weight tensors: not F32, correct element counts.
	// dims[0] = fastest-moving = input_features (row length); dims[1] = output_features.
	// attn_q:   [576, 576]   = [in=embedding, out=n_heads * head_dim = 9 * 64]
	// attn_k:   [576, 192]   = [in=embedding, out=n_kv_heads * head_dim = 3 * 64]
	// ffn_gate: [576, 1536]  = [in=embedding, out=intermediate]
	// token_embd: [576, 49152] = [in=embedding, out=vocab_size]

	Test_assert(t, "attn_q.0 exists",   attnQ0    && attnQ0->quantType    != EQuantType_F32 && attnQ0->elements    == 576 * 576);
	Test_assert(t, "attn_k.0 exists",   attnK0    && attnK0->quantType    != EQuantType_F32 && attnK0->elements    == 576 * 192);
	Test_assert(t, "ffn_gate.0 exists", ffnGate0  && ffnGate0->quantType  != EQuantType_F32 && ffnGate0->elements  == 576 * 1536);
	Test_assert(t, "token_embd exists", tokenEmbd && tokenEmbd->quantType != EQuantType_F32 && tokenEmbd->elements == 576 * 49152);

	//Load output_norm.weight (576 F32 values = 2304 bytes) and validate content.
	//Trained RMS norm scales are typically near 1; hard bound of 100 catches NaN/Inf/zero.

	if(outputNorm) {

		Buffer normBuf = Buffer_createNull();
		Test_assert(t, "output_norm load", GgufTensor_readData(&file, outputNorm, t->alloc, &normBuf, &t->err));

		if(Buffer_length(normBuf)) {

			const F32 *vals = (const F32*) normBuf.ptr;
			Bool allFinite = true, anyNonZero = false;

			for(U64 i = 0; i < 576; ++i) {
				const F32 v = vals[i];
				allFinite  &= F32_abs(v) <= 100.0f;   //NaN/Inf fail this; trained weights bounded
				anyNonZero |= v != 0.0f;
			}

			Test_assert(t, "output_norm finite",   allFinite);
			Test_assert(t, "output_norm non-zero",  anyNonZero);
		}

		Buffer_free(&normBuf, t->alloc);
	}

	GgufFile_free(&file, t->alloc);
	RefPtr_dec(&stream);
}

void Test_gguf(Test *t) {
	Test_ggufValid(t);
	Test_ggufMalformed(t);
	Test_ggufModel(t);
}
