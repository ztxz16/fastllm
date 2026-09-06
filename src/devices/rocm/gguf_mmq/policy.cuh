#pragma once

// Included inside fastllm_rocm_mmq, after the upstream MMQ declarations.
// Keep this list in step with vendors/hip.h: host and device must select the
// same config family and physical wave size, including on multi-device builds.
static int mmq_device_cc(const char *arch, int warp_size) {
    if (!arch || strncmp(arch, "gfx", 3) != 0) return -1;
    char *end = nullptr;
    const long isa = strtol(arch + 3, &end, 16);
    if (end == arch + 3 || (*end != '\0' && *end != ':')) return -1;
    int expected_warp;
    switch (isa) {
        case 0x803: case 0x900: case 0x906:
        case 0x908: case 0x90a: case 0x942: case 0x950:
            expected_warp = 64;
            break;
        case 0x1010: case 0x1012:
        case 0x1030: case 0x1031: case 0x1032: case 0x1033:
        case 0x1034: case 0x1035: case 0x1036:
        case 0x1100: case 0x1101: case 0x1102: case 0x1103:
        case 0x1150: case 0x1151: case 0x1152: case 0x1153:
        case 0x1200: case 0x1201:
            expected_warp = 32;
            break;
        default:
            return -1;
    }
    return warp_size == expected_warp ? GGML_CUDA_CC_OFFSET_AMD + int(isa) : -1;
}

static ggml_cuda_mmq_config mmq_select_config(ggml_type type, int n, int k, int cc,
                                             int warp_size, size_t shared_bytes,
                                             int max_threads) {
    if (cc < GGML_CUDA_CC_OFFSET_AMD || (warp_size != 32 && warp_size != 64))
        return {GGML_TYPE_COUNT, 0, 0, 0, 0, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 0, false, false};
    // These widths have compiled specializations in the adapter. CDNA and
    // pre-WMMA GPUs use 64; RDNA3/4 can use 128 when it fits in shared memory.
    for (int J : {128, 64}) {
        if (J == 128 && n <= 64) continue;
        const auto config = ggml_cuda_mmq_get_config(type, J, k % 128 != 0, cc);
        if (config.type == GGML_TYPE_COUNT || config.I <= 0 ||
            config.nthreads <= 0 || config.nthreads > max_threads ||
            config.nthreads % warp_size || config.I % warp_size ||
            (config.stream_k && config.nthreads < 2 * warp_size) ||
            mmq_get_nbytes_shared(config, cc) > shared_bytes) continue;
        return config;
    }
    return {GGML_TYPE_COUNT, 0, 0, 0, 0, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 0, false, false};
}
