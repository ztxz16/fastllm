#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fastllm {
namespace qwen35_observability {

// Formatting-only data. These structs deliberately contain no model/cache handles,
// locks, allocators, or callbacks; callers snapshot values before formatting.
struct RequestClockRow {
    uint64_t requestId = 0;
    uint64_t arrivalNs = 0;
    uint64_t enqueueNs = 0;
    uint64_t readyNs = 0;
    uint64_t safeBoundaryNs = 0;
    uint64_t admitNs = 0;
};

struct NativeForwardRow {
    uint64_t requestId = 0;
    int nativeHandle = 0;
    uint64_t generation = 0;
    int rowId = 0;
    int seqLen = 0;
    std::string mode;
    std::string path;
};

struct NativeForwardPayload {
    uint64_t forwardId = 0;
    uint64_t roundId = 0;
    std::string dispatch;
    std::string phase;
    int batch = 0;
    int rowCount = 0;
    std::vector<NativeForwardRow> rows;
    std::vector<int> tokenOffsets;
    std::vector<std::string> positionSpans;
    std::vector<std::string> mropeSpans;
    std::vector<std::string> maskSpans;
    std::vector<std::string> kvPageOwnerIds;
    std::vector<std::string> linearOwnerIds;
    std::vector<std::string> dflashMtpOwnerIds;
    std::vector<int> outputRowMap;
    std::vector<int> accepted;
    std::vector<int> drafts;
    std::string executionPath;
    std::string deviceSet = "unknown";
    std::string dflashDevice = "unknown";
    uint64_t beginNs = 0;
    uint64_t endNs = 0;
};

inline std::string FormatU64List(const std::vector<uint64_t> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << values[i];
    }
    out << "]";
    return out.str();
}

inline std::string FormatIntList(const std::vector<int> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << values[i];
    }
    out << "]";
    return out.str();
}

inline std::string FormatStringList(const std::vector<std::string> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << values[i];
    }
    out << "]";
    return out.str();
}

inline std::string FormatRequestClocks(const std::vector<RequestClockRow> &rows) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i != 0) out << ",";
        const RequestClockRow &row = rows[i];
        out << "{request_id=" << row.requestId
            << ",arrival_ns=" << row.arrivalNs
            << ",enqueue_ns=" << row.enqueueNs
            << ",ready_ns=" << row.readyNs
            << ",safe_boundary_ns=" << row.safeBoundaryNs
            << ",admit_ns=" << row.admitNs << "}";
    }
    out << "]";
    return out.str();
}

inline std::string FormatNativeRows(const std::vector<NativeForwardRow> &rows) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i != 0) out << ",";
        const NativeForwardRow &row = rows[i];
        out << "{request_id=" << row.requestId
            << ",native_handle=" << row.nativeHandle
            << ",generation=" << row.generation
            << ",row_id=" << row.rowId
            << ",seq_len=" << row.seqLen
            << ",mode=" << row.mode
            << ",path=" << row.path << "}";
    }
    out << "]";
    return out.str();
}

inline std::string FormatNativeForwardPayload(const NativeForwardPayload &payload) {
    std::vector<uint64_t> requestIds;
    std::vector<uint64_t> generations;
    std::vector<int> handles;
    std::vector<int> rowIds;
    std::vector<int> seqLens;
    std::vector<std::string> modes;
    std::vector<std::string> paths;
    requestIds.reserve(payload.rows.size());
    generations.reserve(payload.rows.size());
    handles.reserve(payload.rows.size());
    rowIds.reserve(payload.rows.size());
    seqLens.reserve(payload.rows.size());
    modes.reserve(payload.rows.size());
    paths.reserve(payload.rows.size());
    int totalInputTokens = 0;
    for (const NativeForwardRow &row : payload.rows) {
        requestIds.push_back(row.requestId);
        generations.push_back(row.generation);
        handles.push_back(row.nativeHandle);
        rowIds.push_back(row.rowId);
        seqLens.push_back(row.seqLen);
        modes.push_back(row.mode);
        paths.push_back(row.path);
        totalInputTokens += row.seqLen;
    }

    std::ostringstream out;
    out << "forward_id=" << payload.forwardId
        << " scheduler_round=" << payload.roundId
        << " dispatch=" << payload.dispatch
        << " phase=" << payload.phase
        << " batch=" << payload.batch
        << " row_count=" << payload.rowCount
        << " request_ids=" << FormatU64List(requestIds)
        << " handles=" << FormatIntList(handles)
        << " native_handles=" << FormatIntList(handles)
        << " generations=" << FormatU64List(generations)
        << " row_ids=" << FormatIntList(rowIds)
        << " seq_lens=" << FormatIntList(seqLens)
        << " token_offsets=" << FormatIntList(payload.tokenOffsets)
        << " total_input_tokens=" << totalInputTokens
        << " modes=" << FormatStringList(modes)
        << " paths=" << FormatStringList(paths)
        << " rows=" << FormatNativeRows(payload.rows)
        << " position_spans=" << FormatStringList(payload.positionSpans)
        << " mrope_spans=" << FormatStringList(payload.mropeSpans)
        << " mask_spans=" << FormatStringList(payload.maskSpans)
        << " kv_page_owner_ids=" << FormatStringList(payload.kvPageOwnerIds)
        << " linear_owner_ids=" << FormatStringList(payload.linearOwnerIds)
        << " dflash_mtp_owner_ids=" << FormatStringList(payload.dflashMtpOwnerIds)
        << " output_row_map=" << FormatIntList(payload.outputRowMap)
        << " execution_path=" << payload.executionPath
        << " device_set=" << payload.deviceSet
        << " dflash_device=" << payload.dflashDevice
        << " begin_ns=" << payload.beginNs
        << " end_ns=" << payload.endNs
        << " native_begin_ns=" << payload.beginNs
        << " native_end_ns=" << payload.endNs
        << " native_duration_ns="
        << (payload.endNs >= payload.beginNs ? payload.endNs - payload.beginNs : 0)
        << " clock_state="
        << ((payload.beginNs != 0 && (payload.dispatch == "begin" ||
                                      payload.endNs >= payload.beginNs)) ?
            "valid" : "invalid");
    if (payload.dispatch == "end" && payload.executionPath == "dflash" &&
        !payload.accepted.empty()) {
        out << " accepted=" << FormatIntList(payload.accepted)
            << " drafts=" << FormatIntList(payload.drafts);
    }
    return out.str();
}

} // namespace qwen35_observability
} // namespace fastllm
