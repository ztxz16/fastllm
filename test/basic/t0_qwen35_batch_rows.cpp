// B phase 1 T0: CPU-only checks of the batch-row contract in models/qwen35_batch_rows.h.
// Build: g++ -std=c++17 -I include test/basic/t0_qwen35_batch_rows.cpp. No Data/GPU/model.
#include "models/qwen35_batch_rows.h"
#include <iostream>

using namespace fastllm;
using namespace fastllm::qwen35_batch_rows;
static int failures = 0, cases = 0;
static void Check(const char *name, bool actual, bool expected = true) {
    ++cases;
    if (actual != expected) ++failures;
    std::cout << "{\"kind\":\"case\",\"name\":\"" << name << "\",\"expected\":"
              << (expected ? "true" : "false") << ",\"actual\":" << (actual ? "true" : "false") << "}\n";
}
static Qwen35MultimodalBatchRow Row(uint64_t handle, uint64_t generation, int seqLen, std::size_t offset) {
    Qwen35MultimodalBatchRow row;
    row.nativeHandle = handle;
    row.nativeGeneration = generation;
    row.seqLen = seqLen;
    row.tokenOffset = offset;
    row.positionTokens = seqLen;
    return row;
}
static std::vector<float> CausalMask(int query, int cache) {
    std::vector<float> mask(static_cast<std::size_t>(query) * (cache + query), 0.0f);
    for (int q = 0; q < query; ++q)
        for (int k = cache + q + 1; k < cache + query; ++k) mask[q * (cache + query) + k] = 1.0f;
    return mask;
}
int main() {
    std::string error;
    std::vector<std::size_t> offsets;
    Check("offsets_ragged", BuildRowOffsets({3, 2}, offsets, &error) &&
                                offsets == std::vector<std::size_t>({0, 3, 5}));
    Check("offsets_empty_rejected", BuildRowOffsets({}, offsets, &error), false);
    Check("offsets_zero_rejected", BuildRowOffsets({3, 0}, offsets, &error), false);

    CpuPositionPlan plan;
    plan.stride = 5;
    for (int axis = 0; axis < 3; ++axis)
        for (int token = 0; token < 5; ++token) plan.values.push_back(axis * 100.0f + token);
    CpuPositionPlan chunk;
    Check("split_3x5_middle", SplitMultimodalPositionIdsCPU(plan, 1, 4, chunk, &error) && chunk.stride == 3 &&
                                  chunk.values == std::vector<float>({1, 2, 3, 101, 102, 103, 201, 202, 203}));
    Check("split_out_of_range_rejected", SplitMultimodalPositionIdsCPU(plan, 3, 6, chunk, &error), false);
    Check("split_empty_rejected", SplitMultimodalPositionIdsCPU(plan, 2, 2, chunk, &error), false);

    std::vector<Qwen35MultimodalBatchRow> rows = {Row(1, 1, 3, 0), Row(2, 1, 2, 3)};
    Check("rows_ragged_ok", ValidateRows(rows, 5, &error));
    Check("rows_total_mismatch_rejected", ValidateRows(rows, 6, &error), false);
    std::vector<Qwen35MultimodalBatchRow> duplicate = {Row(1, 1, 3, 0), Row(1, 1, 2, 3)};
    Check("rows_duplicate_identity_rejected", ValidateRows(duplicate, 5, &error), false);
    std::vector<Qwen35MultimodalBatchRow> zero = {Row(1, 1, 0, 0)};
    Check("rows_zero_length_rejected", ValidateRows(zero, 0, &error), false);
    std::vector<Qwen35MultimodalBatchRow> swapped = {Row(1, 1, 3, 2), Row(2, 1, 2, 0)};
    Check("rows_non_prefix_offset_rejected", ValidateRows(swapped, 5, &error), false);
    std::vector<Qwen35MultimodalBatchRow> axes = rows;
    axes[1].positionAxes = 1;
    Check("rows_axis_rejected", ValidateRows(axes, 5, &error), false);
    std::vector<Qwen35MultimodalBatchRow> cursor = rows;
    cursor[0].embeddingTokens = 4;
    cursor[0].embeddingCursor = 2;
    Check("rows_cursor_overrun_rejected", ValidateRows(cursor, 5, &error), false);
    std::vector<Qwen35MultimodalBatchRow> textRow = rows;
    textRow[1].multimodal = false;
    Check("rows_text_row_ok", ValidateRows(textRow, 5, &error));
    textRow[1].embeddingTokens = 2;
    Check("rows_text_row_with_media_rejected", ValidateRows(textRow, 5, &error), false);

    std::vector<float> good = CausalMask(3, 2), bad = CausalMask(3, 2);
    bad[0 * 5 + 4] = 0.0f;
    std::vector<Qwen35MultimodalBatchRow> masked = {Row(1, 1, 3, 0)};
    masked[0].cacheLen = 2;
    Check("mask_causal_ok", ValidateRows(masked, {CpuMaskView{good.data(), good.size(), 3, 5}}, 3, &error));
    Check("mask_noncausal_rejected", ValidateRows(masked, {CpuMaskView{bad.data(), bad.size(), 3, 5}}, 3, &error), false);
    Check("mask_shape_rejected", ValidateRows(masked, {CpuMaskView{good.data(), good.size(), 3, 4}}, 3, &error), false);

    // Negative before mutation: the caller may touch cache state only after a true validation.
    int cacheMutationCount = 0;
    if (ValidateRows(duplicate, 5, &error)) ++cacheMutationCount;
    Check("invalid_rows_leave_cache_untouched", cacheMutationCount == 0);
    if (ValidateRows(rows, 5, &error)) ++cacheMutationCount;
    Check("valid_rows_allow_mutation", cacheMutationCount == 1);

    std::cout << "{\"kind\":\"summary\",\"cases\":" << cases << ",\"failures\":" << failures << "}\n";
    return failures == 0 ? 0 : 1;
}
