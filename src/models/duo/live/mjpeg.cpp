#include "input.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
#include <memory>

namespace fastllm::duo::live {
namespace {
struct DecodeState {
    AVCodecParserContext *parser = av_parser_init(AV_CODEC_ID_MJPEG);
    AVCodecContext *codec = avcodec_alloc_context3(avcodec_find_decoder(AV_CODEC_ID_MJPEG));
    AVFrame *frame = av_frame_alloc();
    AVPacket *packet = av_packet_alloc();
    SwsContext *sws = nullptr;
    ~DecodeState() {
        if (parser)
            av_parser_close(parser);
        avcodec_free_context(&codec);
        av_frame_free(&frame);
        av_packet_free(&packet);
        sws_freeContext(sws);
    }
};
} // namespace
void AVBlock::decodeJpeg() {
    if (mjpeg.empty())
        return;
    if (!rgb.empty() || width <= 0 || height <= 0 || width > 4096 || height > 4096 ||
        mjpeg.size() > (64u << 20))
        throw InputError("invalid MJPEG block");
    DecodeState s;
    if (!s.parser || !s.codec || !s.frame || !s.packet)
        throw std::bad_alloc();
    s.codec->thread_count = 1;
    s.codec->max_pixels = int64_t(width) * height;
    if (avcodec_open2(s.codec, s.codec->codec, nullptr) < 0)
        throw InputError("cannot open JPEG decoder");
    const size_t frameBytes = size_t(width) * height * 3;
    rgb.resize(frameBytes * 17);
    int count = 0;
    auto decode = [&] {
        if (count >= 17 || avcodec_send_packet(s.codec, s.packet) < 0 ||
            avcodec_receive_frame(s.codec, s.frame) < 0 || s.frame->width != width ||
            s.frame->height != height)
            throw InputError("invalid JPEG frame or geometry");
        auto fmt = static_cast<AVPixelFormat>(s.frame->format);
        bool fullRange = s.frame->color_range == AVCOL_RANGE_JPEG;
        // YUVJ is the full-range spelling of the same planar storage. Passing the
        // canonical format lets swscale reuse its context across the 17 frames.
        switch (fmt) {
        case AV_PIX_FMT_YUVJ420P:
            fmt = AV_PIX_FMT_YUV420P;
            fullRange = true;
            break;
        case AV_PIX_FMT_YUVJ422P:
            fmt = AV_PIX_FMT_YUV422P;
            fullRange = true;
            break;
        case AV_PIX_FMT_YUVJ444P:
            fmt = AV_PIX_FMT_YUV444P;
            fullRange = true;
            break;
        case AV_PIX_FMT_YUVJ440P:
            fmt = AV_PIX_FMT_YUV440P;
            fullRange = true;
            break;
        default:
            break;
        }
        s.sws = sws_getCachedContext(s.sws, width, height, fmt, width, height, AV_PIX_FMT_RGB24, SWS_BICUBIC,
                                     nullptr, nullptr, nullptr);
        if (!s.sws)
            throw InputError("unsupported JPEG colorspace");
        const int matrix =
            s.frame->colorspace == AVCOL_SPC_UNSPECIFIED ? SWS_CS_DEFAULT : int(s.frame->colorspace);
        const int *coeff = sws_getCoefficients(matrix);
        if (sws_setColorspaceDetails(s.sws, coeff, fullRange, coeff, 1, 0, 1 << 16, 1 << 16) < 0)
            throw InputError("unsupported JPEG color range");
        uint8_t *planes[]{rgb.data() + count * frameBytes, nullptr, nullptr, nullptr};
        int strides[]{width * 3, 0, 0, 0};
        if (sws_scale(s.sws, s.frame->data, s.frame->linesize, 0, height, planes, strides) != height)
            throw InputError("incomplete JPEG conversion");
        av_frame_unref(s.frame);
        ++count;
    };
    const int bytes = int(mjpeg.size());
    mjpeg.resize(bytes + AV_INPUT_BUFFER_PADDING_SIZE, 0);
    for (int pos = 0; pos < bytes;) {
        int n = av_parser_parse2(s.parser, s.codec, &s.packet->data, &s.packet->size, mjpeg.data() + pos,
                                 bytes - pos, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (n < 0 || (!n && !s.packet->size))
            throw InputError("invalid MJPEG framing");
        if (s.packet->size)
            decode();
        pos += n;
    }
    av_parser_parse2(s.parser, s.codec, &s.packet->data, &s.packet->size, nullptr, 0, AV_NOPTS_VALUE,
                     AV_NOPTS_VALUE, 0);
    if (s.packet->size)
        decode();
    if (count != 17)
        throw InputError("expected 17 JPEG frames");
}
} // namespace fastllm::duo::live
