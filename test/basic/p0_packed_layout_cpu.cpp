// Build against fastllm/include, after A applies layout.patch. No Data/GPU shim.
#include "utils/qwen35_packed_layout.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace fastllm::qwen35_packed;
static int failures = 0, cases = 0;
static void Check(const char *name, bool actual, bool expected = true) {
    ++cases;
    if (actual != expected) ++failures;
    std::cout << "{\"kind\":\"case\",\"implementation\":\"actual_native_header\",\"name\":\""
        << name << "\",\"expected\":" << (expected ? "true" : "false")
        << ",\"actual\":" << (actual ? "true" : "false") << "}\n";
}
template<class T> static void Array(const std::vector<T> &values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
static std::vector<float> Reference(const std::vector<std::vector<float>> &rows,
                                  const std::vector<int> &lengths,
                                  const std::vector<int> &strides, int axes) {
    // Independent single-row indexing reference, not the production memcpy loop.
    std::vector<float> output;
    for (int axis = 0; axis < axes; ++axis)
        for (size_t row = 0; row < rows.size(); ++row)
            for (int token = 0; token < lengths[row]; ++token)
                output.push_back(rows[row].at(axis * strides[row] + token));
    return output;
}
int main() {
    std::string error;
    std::vector<int> offsets;
    Check("offsets_ragged", BuildRowOffsets({2,3}, offsets, &error) && offsets == std::vector<int>({0,2,5}));
    Check("offsets_empty_rejected", BuildRowOffsets({}, offsets, &error), false);
    Check("offsets_zero_rejected", BuildRowOffsets({2,0}, offsets, &error), false);
    Check("offsets_negative_rejected", BuildRowOffsets({2,-1}, offsets, &error), false);
    Check("offsets_overflow_rejected", BuildRowOffsets({INT_MAX,1}, offsets, &error), false);
    std::vector<std::vector<float>> rowValues = {{100,101,110,111,120,121},
                                                {200,201,202,300,301,302,400,401,402}};
    std::vector<PositionRow> rows = {{rowValues[0].data(),6,3,2},{rowValues[1].data(),9,3,3}};
    std::vector<float> packed;
    int axes;
    const auto expected = Reference(rowValues,{2,3},{2,3},3);
    bool ok = PackPositionRows(rows,{2,3},false,packed,axes,&error);
    Check("mrope_ragged_axis_major_cached_bases", ok && axes == 3 && packed == expected);
    std::vector<float> mask0={0,0,0,1,0,0,0,0};
    std::vector<float> mask1={0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0};
    Check("structural_row_masks",ValidateCausalMask(mask0.data(),mask0.size(),2,2,2,4,&error) &&
        ValidateCausalMask(mask1.data(),mask1.size(),3,3,3,6,&error));
    std::cout << "{\"kind\":\"packed_structural\",\"implementation\":\"actual_native_header\","
        "\"seqLens\":[2,3],\"tokenOffsets\":[0,2,5],\"rowPositionShapes\":[[3,2],[3,3]],"
        "\"cacheBases\":[100,200],\"cacheLengths\":[2,3],\"hiddenShape\":[1,5,2],\"maskShapes\":[[2,4],[3,6]],"
        "\"packedPositionShape\":[3,5],\"positionRows\":[";
    Array(rowValues[0]); std::cout << ','; Array(rowValues[1]);
    std::cout << "],\"expectedAxisMajor\":"; Array(expected);
    std::cout << ",\"actualAxisMajor\":"; Array(packed);
    std::cout << ",\"maskRows\":["; Array(mask0); std::cout << ','; Array(mask1); std::cout << "]}\n";
    std::reverse(rows.begin(),rows.end()); std::reverse(rowValues.begin(),rowValues.end());
    Check("mrope_reversed_owner_order", PackPositionRows(rows,{3,2},false,packed,axes,&error) &&
        packed == Reference(rowValues,{3,2},{3,2},3));
    std::vector<float> tail0={7,8,9,70,80,90,700,800,900}, tail1={5,6,50,60,500,600};
    rows={{tail0.data()+2,7,3,3},{tail1.data()+1,5,3,2}};
    Check("mrope_tail_subviews_keep_axis_stride", PackPositionRows(rows,{1,1},true,packed,axes,&error) &&
        packed==std::vector<float>({9,6,90,60,900,600}));
    rows={{tail0.data(),9,3,3},{tail1.data(),6,3,2}};
    Check("mrope_ragged_tail_blocks", PackPositionRows(rows,{2,1},false,packed,axes,&error) &&
        packed==std::vector<float>({7,8,5,70,80,50,700,800,500}));
    rows[1].axes=1;
    Check("mixed_axis_rejected",PackPositionRows(rows,{2,1},false,packed,axes,&error),false);
    rows[1].axes=3; rows[1].count=4;
    Check("position_short_buffer_rejected",PackPositionRows(rows,{2,1},false,packed,axes,&error),false);
    rows[1].count=6; rows[1].stride=0;
    Check("position_bad_stride_rejected",PackPositionRows(rows,{2,1},false,packed,axes,&error),false);
    rows[1].stride=2; rows[1].values=nullptr;
    Check("position_null_rejected",PackPositionRows(rows,{2,1},false,packed,axes,&error),false);
    rows[1].values=tail1.data();
    Check("all1_inconsistent_rejected",PackPositionRows(rows,{2,1},true,packed,axes,&error),false);
    Check("position_row_count_rejected",PackPositionRows(rows,{2},false,packed,axes,&error),false);
    rows={{tail0.data(),3,1,3},{tail1.data(),2,1,2}};
    Check("text_one_axis_preserved",PackPositionRows(rows,{3,2},false,packed,axes,&error) &&
        axes==1 && packed==std::vector<float>({7,8,9,5,6}));
    rows={{tail0.data(),9,3,3}};
    Check("single_mrope_preserved",PackPositionRows(rows,{3},false,packed,axes,&error) && packed==tail0);
    Check("media_rows_all_mm",ValidateMediaRows({true,true},&error));
    Check("mixed_text_mm_rejected",ValidateMediaRows({true,false},&error),false);
    Check("mixed_mm_text_rejected",ValidateMediaRows({false,true},&error),false);
    Check("empty_media_rejected",ValidateMediaRows({},&error),false);
    std::vector<float> h0={10,11,12,13}, h1={20,21,22,23,24,25};
    std::vector<float> guarded(12,-777);
    std::vector<HiddenRow> hidden={{h0.data(),16,2,2,1},{h1.data(),24,3,2,1}};
    Check("hidden_row_boundaries",CopyPackedHiddenRows(hidden,{2,3},8,guarded.data()+1,40,&error) &&
        guarded==std::vector<float>({-777,10,11,12,13,20,21,22,23,24,25,-777}));
    std::cout << "{\"kind\":\"packed_hidden_structural\",\"implementation\":\"actual_native_header\","
        "\"seqLens\":[2,3],\"tokenOffsets\":[0,2,5],\"hiddenByteOffsets\":[0,16,40],"
        "\"hiddenShape\":[1,5,2],\"hiddenRowShapes\":[[1,2,2],[1,3,2]],\"tokenBytes\":8,\"rowValues\":[";
    Array(h0); std::cout << ','; Array(h1); std::cout << "],\"expected\":[10,11,12,13,20,21,22,23,24,25],\"actual\":";
    Array(std::vector<float>(guarded.begin()+1,guarded.end()-1)); std::cout << ",\"guards\":[" << guarded.front() << ',' << guarded.back() << "]}\n";
    auto rejectHidden=[&](const char *name) {
        std::fill(guarded.begin(),guarded.end(),-777);
        bool rejected=!CopyPackedHiddenRows(hidden,{2,3},8,guarded.data()+1,40,&error);
        Check(name,rejected && std::all_of(guarded.begin(),guarded.end(),[](float v){return v==-777;}));
    };
    hidden[1].bytes=16; rejectHidden("hidden_short_rejected_without_write"); hidden[1].bytes=24;
    hidden[1].width=3; rejectHidden("hidden_width_rejected_without_write"); hidden[1].width=2;
    hidden[1].type=2; rejectHidden("hidden_dtype_rejected_without_write"); hidden[1].type=1;
    hidden[1].tokens=2; rejectHidden("hidden_length_rejected_without_write"); hidden[1].tokens=3;
    hidden[1].values=nullptr; rejectHidden("hidden_null_rejected_without_write"); hidden[1].values=h1.data();
    Check("hidden_destination_size_rejected",CopyPackedHiddenRows(hidden,{2,3},8,guarded.data()+1,36,&error),false);
    Check("implicit_causal_mask",ValidateCausalMask(nullptr,0,2,3,0,0,&error));
    std::vector<float> mask={0,0,0,0,1,0,0,0,0,0};
    Check("cached_causal_mask",ValidateCausalMask(mask.data(),10,2,3,2,5,&error));
    mask[1]=1;
    Check("noncausal_mask_rejected",ValidateCausalMask(mask.data(),10,2,3,2,5,&error),false); mask[1]=0;
    Check("mask_shape_rejected",ValidateCausalMask(mask.data(),10,2,3,1,10,&error),false);
    Check("mask_short_rejected",ValidateCausalMask(mask.data(),9,2,3,2,5,&error),false);
    mask[0]=std::nanf("");
    Check("mask_nan_rejected",ValidateCausalMask(mask.data(),10,2,3,2,5,&error),false);
    Check("negative_cache_rejected",ValidateCausalMask(nullptr,0,2,-1,0,0,&error),false);
    Check("cache_sum_overflow_rejected",ValidateCausalMask(nullptr,0,2,INT_MAX,0,0,&error),false);
    std::cout << "{\"kind\":\"summary\",\"cases\":" << cases << ",\"failures\":" << failures
        << ",\"scope\":\"actual_native_header_cpu_only\",\"gpu\":\"NOT_RUN\"}\n";
    return failures == 0 ? 0 : 1;
}
