#include "avc.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <x264.h>

namespace fastllm::duo::live {
namespace {
void word(std::ostream &out, uint32_t value) {
    const uint8_t bytes[]{uint8_t(value), uint8_t(value >> 8), uint8_t(value >> 16), uint8_t(value >> 24)};
    out.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}
} // namespace
AvcEncoder::AvcEncoder(int w, int h) : width(w), height(h) {
    x264_param_t p;
    if (x264_param_default_preset(&p, "ultrafast", "zerolatency") < 0)
        throw std::runtime_error("unsupported x264 preset");
    p.i_csp = X264_CSP_I420;
    p.i_width = w;
    p.i_height = h;
    p.i_fps_num = p.i_timebase_den = 24;
    p.i_fps_den = p.i_timebase_num = 1;
    p.i_threads = 2;
    p.b_vfr_input = 0;
    p.i_keyint_max = 48;
    p.i_scenecut_threshold = 0;
    p.i_bframe = 0;
    p.b_repeat_headers = p.b_annexb = 1;
    p.rc.i_rc_method = X264_RC_CRF;
    p.rc.f_rf_constant = 20;
    p.i_level_idc = 31;
    p.i_log_level = X264_LOG_ERROR;
    if (x264_param_apply_profile(&p, "baseline") < 0 || !(encoder = x264_encoder_open(&p)))
        throw std::runtime_error("cannot open low-delay x264 encoder");
}
AvcEncoder::~AvcEncoder() {
    if (encoder)
        x264_encoder_close(encoder);
}
void AvcEncoder::write(const std::filesystem::path &file, int block, const uint8_t *i420,
                       const std::vector<float> &audio) {
    constexpr int frames = 17;
    const int pixels = width * height;
    std::vector<uint8_t> video;
    uint32_t sizes[frames];
    for (int i = 0; i < frames; ++i) {
        x264_picture_t in, out;
        x264_picture_init(&in);
        in.i_pts = frame++;
        in.img.i_csp = X264_CSP_I420;
        in.img.i_plane = 3;
        auto y = const_cast<uint8_t *>(i420 + size_t(i) * pixels * 3 / 2);
        in.img.plane[0] = y;
        in.img.plane[1] = y + pixels;
        in.img.plane[2] = y + pixels + pixels / 4;
        in.img.i_stride[0] = width;
        in.img.i_stride[1] = in.img.i_stride[2] = width / 2;
        x264_nal_t *nals;
        int count;
        const int length = x264_encoder_encode(encoder, &nals, &count, &in, &out);
        if (length <= 0 || !count || in.i_pts != out.i_pts)
            throw std::runtime_error("x264 did not return the current frame");
        sizes[i] = uint32_t(length) | (out.b_keyframe ? 0x80000000u : 0u);
        for (int j = 0; j < count; ++j)
            video.insert(video.end(), nals[j].p_payload, nals[j].p_payload + nals[j].i_payload);
    }
    std::vector<int16_t> pcm;
    pcm.reserve(audio.size());
    for (float sample : audio)
        pcm.push_back(int16_t(std::lrint(std::clamp(sample * 32768.f, -32768.f, 32767.f))));
    std::ofstream stream(file, std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    word(stream, 12 + 4 * frames + pcm.size() * 2 + video.size());
    word(stream, block);
    word(stream, frames);
    word(stream, pcm.size() / 2);
    for (auto size : sizes)
        word(stream, size);
    stream.write(reinterpret_cast<const char *>(pcm.data()), pcm.size() * 2);
    stream.write(reinterpret_cast<const char *>(video.data()), video.size());
    stream.close();
}
} // namespace fastllm::duo::live
