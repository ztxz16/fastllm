#pragma once
namespace naivert::prepared_mlp {
constexpr int kRows = 8;
constexpr int kHidden = 4096;
constexpr int kBlocks = kHidden/128;
constexpr int kRowStride = kHidden+16;
constexpr int kScaleOffset = kRows*kRowStride;
// FP32 scales are [K128 block][token], matching direct selector reads.
constexpr int kScaleBlockStride = kRows;
constexpr int kScaleBytes = kRows*kBlocks*4;
constexpr int kBytes = kScaleOffset+kScaleBytes;
// Resident prepared scale image in selector dynamic shared memory.
constexpr int kScaleScratch = 0x3800;
static_assert(kBytes == 33920 && kScaleOffset%16 == 0 && kRowStride%16 == 0);
}
