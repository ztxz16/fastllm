#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace fastllm {
namespace minimax_h3 {
std::vector<Image> prepareKeyframes(const GenerateOptions &options) {
    std::vector<Image> images;
    for (const auto *source : {&options.firstFrame, &options.lastFrame}) {
        if (!*source)
            continue;
        auto image = **source;
        if (image.width < 1 || image.height < 1 || image.width > 4096 || image.height > 4096 ||
            image.rgba.size() != size_t(image.width) * image.height * 4)
            throw std::invalid_argument("invalid H3 keyframe image");
        // The native input is PIL.convert('RGB'): alpha is discarded, not composited.
        for (size_t i = 3; i < image.rgba.size(); i += 4)
            image.rgba[i] = 255;
        if (images.empty())
            image = qwen_image::resizeImage(image, options.width, options.height);
        else {
            double scale =
                std::max(double(options.width) / image.width, double(options.height) / image.height);
            int w = std::max(options.width, int(std::nearbyint(image.width * scale)));
            int h = std::max(options.height, int(std::nearbyint(image.height * scale)));
            image = qwen_image::resizeImage(image, w, h);
            Image cropped{options.width, options.height,
                          std::vector<uint8_t>(size_t(options.width) * options.height * 4)};
            for (int y = 0; y < options.height; ++y)
                std::copy_n(image.rgba.data() +
                                (size_t(y + (h - options.height) / 2) * w + (w - options.width) / 2) * 4,
                            size_t(options.width) * 4, cropped.rgba.data() + size_t(y) * options.width * 4);
            image = std::move(cropped);
        }
        images.push_back(std::move(image));
    }
    return images;
}
Image prepareVisionImage(const Image &source, int videoFrames) {
    int h = source.height, w = source.width;
    int rh = std::max(32, int(std::nearbyint(h / 32.)) * 32);
    int rw = std::max(32, int(std::nearbyint(w / 32.)) * 32);
    int minPixels = videoFrames ? 4096 : 65536, maxPixels = videoFrames ? 25165824 : 16777216;
    int roundedFrames = videoFrames ? std::max(2, int(std::nearbyint(videoFrames / 2.)) * 2) : 1;
    int actualFrames = std::max(1, videoFrames);
    if (int64_t(roundedFrames) * rh * rw < minPixels) {
        double scale = std::sqrt(double(minPixels) / (double(actualFrames) * h * w));
        rh = int(std::ceil(h * scale / 32)) * 32;
        rw = int(std::ceil(w * scale / 32)) * 32;
    } else if (int64_t(roundedFrames) * rh * rw > maxPixels) {
        double scale = std::sqrt(double(actualFrames) * h * w / maxPixels);
        rh = int(std::floor(h / scale / 32)) * 32;
        rw = int(std::floor(w / scale / 32)) * 32;
    }
    return qwen_image::resizeImage(source, rw, rh, qwen_image::ImageResizeFilter::TorchBicubic, 16384);
}
namespace {
void validateImage(const Image &image) {
    if (image.width < 1 || image.height < 1 || image.width > 16384 || image.height > 16384 ||
        int64_t(image.width) * image.height > 16777216 ||
        image.rgba.size() != size_t(image.width) * image.height * 4)
        throw std::invalid_argument("invalid H3 reference image");
    if (image.width > 4 * image.height || image.height > 4 * image.width)
        throw std::invalid_argument("H3 reference aspect ratio must be within 1:4..4:1");
}
std::vector<float> normalizeAudio(const Reference &ref, double duration) {
    if (ref.audio.empty())
        return {};
    if ((ref.channels != 1 && ref.channels != 2) || ref.sampleRate < 8000 || ref.sampleRate > 192000 ||
        ref.audio.size() % ref.channels)
        throw std::invalid_argument("reference audio requires mono/stereo PCM at 8000..192000 Hz");
    size_t samples = std::min(ref.audio.size() / ref.channels, size_t(duration * ref.sampleRate));
    for (float x : ref.audio)
        if (!std::isfinite(x))
            throw std::invalid_argument("non-finite reference audio");
    if (ref.sampleRate == 32000) {
        std::vector<float> result(samples * 2);
        for (size_t i = 0; i < samples; ++i)
            for (int c = 0; c < 2; ++c)
                result[i * 2 + c] = ref.audio[i * ref.channels + c % ref.channels];
        return result;
    }
    // torchaudio's default sinc_interp_hann polyphase filter: width=6, rolloff=.99.
    int divisor = std::gcd(ref.sampleRate, 32000), from = ref.sampleRate / divisor, to = 32000 / divisor;
    double base = std::min(from, to) * .99;
    int width = int(std::ceil(6. * from / base)), taps = 2 * width + from;
    std::vector<float> kernel(size_t(to) * taps);
    for (int phase = 0; phase < to; ++phase)
        for (int tap = 0; tap < taps; ++tap) {
            double t = std::clamp((-double(phase) / to + double(tap - width) / from) * base, -6., 6.);
            double window = std::cos(t * M_PI / 12);
            t *= M_PI;
            kernel[size_t(phase) * taps + tap] =
                float((t == 0 ? 1 : std::sin(t) / t) * window * window * base / from);
        }
    size_t output = (samples * to + from - 1) / from;
    std::vector<float> result(output * 2);
    for (size_t i = 0; i < output; ++i)
        for (int c = 0; c < ref.channels; ++c) {
            float value = 0;
            int64_t begin = int64_t(i / to) * from - width;
            for (int tap = 0; tap < taps; ++tap)
                if (begin + tap >= 0 && begin + tap < int64_t(samples))
                    value +=
                        ref.audio[size_t(begin + tap) * ref.channels + c] * kernel[(i % to) * taps + tap];
            result[i * 2 + c] = value;
            if (ref.channels == 1)
                result[i * 2 + 1] = value;
        }
    return result;
}
} // namespace
std::vector<Reference> prepareReferences(const GenerateOptions &options) {
    std::vector<Reference> result;
    int images = 0, videos = 0, audios = 0;
    double videoDuration = 0, audioDuration = 0;
    for (const auto &source : options.references) {
        Reference ref;
        ref.kind = source.kind;
        if (source.kind == Reference::Kind::Audio) {
            ++audios;
            if (source.channels < 1 || source.sampleRate < 1)
                throw std::invalid_argument("invalid reference audio rate");
            double duration = source.sourceDuration > 0
                                  ? source.sourceDuration
                                  : double(source.audio.size()) / source.channels / source.sampleRate;
            if (duration < 2 || duration > 15)
                throw std::invalid_argument("reference audio must last 2..15 seconds");
            audioDuration += duration;
        } else {
            if (source.frames.empty())
                throw std::invalid_argument("reference has no image/video frames");
            for (const auto &image : source.frames)
                validateImage(image);
            int w = source.frames[0].width, h = source.frames[0].height;
            double scale;
            if (source.kind == Reference::Kind::Image) {
                ++images;
                if (source.frames.size() != 1 || !source.audio.empty())
                    throw std::invalid_argument("an image reference requires one image");
                scale = double(options.referenceImageShortEdge) / std::min(w, h);
            } else {
                ++videos;
                if (!std::isfinite(source.fps) || source.fps <= 0)
                    throw std::invalid_argument("invalid reference video fps");
                double duration =
                    source.sourceDuration > 0 ? source.sourceDuration : source.frames.size() / source.fps;
                if (duration < 2 || duration > 15)
                    throw std::invalid_argument("reference video must last 2..15 seconds");
                videoDuration += duration;
                scale = double(options.referenceVideoShortEdge) / std::min(w, h);
                scale = std::min(scale, std::sqrt(768. * 1344 / (double(w) * h)));
            }
            int rw = std::max(32, int(std::nearbyint(w * scale / 32)) * 32);
            int rh = std::max(32, int(std::nearbyint(h * scale / 32)) * 32);
            for (size_t i = 0; i < source.frames.size(); ++i) {
                if (source.frames[i].width != w || source.frames[i].height != h)
                    throw std::invalid_argument("reference video frame dimensions must match");
                int begin =
                    source.kind == Reference::Kind::Image ? 0 : int(std::floor(i * 24. / source.fps + .5));
                int end = source.kind == Reference::Kind::Image
                              ? 1
                              : int(std::floor((i + 1) * 24. / source.fps + .5));
                end = std::min(end, options.frames);
                if (end <= begin)
                    continue;
                auto image = source.frames[i];
                for (size_t j = 3; j < image.rgba.size(); j += 4)
                    image.rgba[j] = 255;
                image = qwen_image::resizeImage(image, rw, rh, qwen_image::ImageResizeFilter::Lanczos, 8192);
                for (int frame = begin; frame < end; ++frame)
                    ref.frames.push_back(image);
            }
        }
        if (source.kind == Reference::Kind::Audio && source.audio.empty())
            throw std::invalid_argument("reference audio is empty");
        ref.audio = normalizeAudio(source, options.frames / 24.);
        result.push_back(std::move(ref));
    }
    if (images > 9 || videos > 3 || audios > 3 || result.size() > 12 || videoDuration > 15.000001 ||
        audioDuration > 15.000001 || (audios && !images && !videos))
        throw std::invalid_argument("Ref2VA accepts up to 9 images, 3 videos (15s total), 3 audio clips (15s "
                                    "total), 12 references; audio requires an image/video");
    return result;
}
PackedLayout referenceLayout(const std::vector<int> &textTags, int frames, int height, int width,
                             int audioFrames, const std::vector<ReferenceGeometry> &references) {
    auto base = textToVideoLayout(textTags.size(), frames, height, width, audioFrames);
    PackedLayout result;
    result.text = base.text;
    result.tags = textTags;
    result.positions.assign(base.positions.begin(), base.positions.begin() + textTags.size() * 3);
    double time = textTags.size();
    auto append = [&](int ft, int h, int w, int at, double origin) {
        if (h < 2 || w < 2 || h % 2 || w % 2 || ft < 0 || at < 0)
            throw std::invalid_argument("invalid H3 reference latent geometry");
        double area = std::sqrt(double(h) * w), rh = h / area, rw = w / area;
        auto hp = [&](int y) { return ((1 - rh) / 2 + double(y) * (rh / (h / 2))) * 32; };
        auto wp = [&](int x) { return ((1 - rw) / 2 + double(x) * (rw / (w / 2))) * 32; };
        for (int i = 0; i < at * 2; ++i) {
            result.audio.push_back(result.tags.size());
            result.tags.push_back(2);
            result.positions.insert(result.positions.end(),
                                    {float(origin + i % at), 0.f, float(wp(i < at ? 0 : w / 2 - 1))});
        }
        double clock = origin;
        for (int t = 0; t < ft; ++t) {
            for (int y = 0; y < h / 2; ++y)
                for (int x = 0; x < w / 2; ++x) {
                    result.video.push_back(result.tags.size());
                    result.tags.push_back(0);
                    result.positions.insert(result.positions.end(),
                                            {float(clock), float(hp(y)), float(wp(x))});
                }
            clock += (5. / 3.) * (t % 5 == 0 ? 1 : 4);
        }
    };
    for (const auto &ref : references) {
        int h = ref.kind == Reference::Kind::Audio ? height : ref.height;
        int w = ref.kind == Reference::Kind::Audio ? width : ref.width;
        append(ref.frames, h, w, ref.audioFrames, time);
        double span = 0;
        for (int i = 0; i < ref.frames; ++i)
            span += (5. / 3.) * (i % 5 == 0 ? 1 : 4);
        time += ref.kind == Reference::Kind::Image ? 1 : std::max(double(ref.audioFrames), span);
    }
    append(frames, height, width, audioFrames, time);
    result.timeIndices.resize(result.tags.size());
    return result;
}
std::vector<float> normalNoise(std::mt19937 &generator, size_t count) {
    // All H3 latent draws have at least 16 elements. The last incomplete block
    // consumes a fresh 16 uniforms and overwrites the final 16 output positions.
    if (count < 16)
        throw std::invalid_argument("H3 normal draw requires at least 16 values");
    auto uniform = [&] { return float(generator() & 0xffffff) * (1.f / 16777216.f); };
    std::vector<float> values(count);
    for (auto &v : values)
        v = uniform();
    auto transform = [](float *v) {
        for (int j = 0; j < 8; ++j) {
            float radius = std::sqrt(-2.f * std::log(1.f - v[j]));
            float angle = float(2.0 * M_PI) * v[j + 8];
            v[j] = radius * std::cos(angle);
            v[j + 8] = radius * std::sin(angle);
        }
    };
    for (size_t i = 0; i + 16 <= count; i += 16)
        transform(values.data() + i);
    if (count % 16) {
        for (size_t i = count - 16; i < count; ++i)
            values[i] = uniform();
        transform(values.data() + count - 16);
    }
    return values;
}
VideoTensor sampleVideoPosterior(Runtime &fp32, const VideoTensor &moments, uint64_t seed) {
    if (!fp32.isFloat32() || moments.channels != 48 || moments.frames < 1 || moments.height < 1 ||
        moments.width < 1 ||
        moments.values.size() != size_t(moments.frames) * moments.height * moments.width * 48)
        throw std::invalid_argument("invalid H3 video posterior");
    VideoTensor result(moments.frames, moments.height, moments.width, 24);
    std::mt19937 generator(seed);
    auto noise = normalNoise(generator, result.values.size());
    size_t rows = size_t(moments.frames) * moments.height * moments.width;
    for (size_t row = 0; row < rows; ++row)
        for (int c = 0; c < 24; ++c) {
            float deviation = std::exp(.5f * std::clamp(moments.values[row * 48 + c + 24], -30.f, 20.f));
            result.values[row * 24 + c] =
                moments.values[row * 48 + c] + deviation * noise[size_t(c) * rows + row];
        }
    // This explicit FP16 rounding is part of the released tokenizer, even when
    // the encoder and the subsequent DiT input projections run in FP32.
    result.values = fp32.download(fp32.cast(fp32.upload(result.values.data(), rows, 24), DType::FP16));
    return result;
}
PackedLayout keyframeLayout(const std::vector<int> &textTags, int frames, int h, int w, int audio,
                            const std::vector<int> &anchors) {
    if (anchors != std::vector<int>{} && anchors != std::vector<int>{0} && anchors != std::vector<int>{-1} &&
        anchors != std::vector<int>{0, -1})
        throw std::invalid_argument("H3 keyframes must be first, last, or first and last");
    auto base = textToVideoLayout(textTags.size(), frames, h, w, audio);
    for (int tag : textTags)
        if (tag != 0 && tag != 1)
            throw std::invalid_argument("invalid H3 presentation tag");
    int nt = textTags.size(), perFrame = h / 2 * (w / 2), extra = anchors.size() * perFrame;
    PackedLayout out;
    out.positions.resize(base.positions.size() + size_t(extra) * 3);
    out.tags.resize(base.tags.size() + extra);
    out.timeIndices.resize(out.tags.size());
    out.text = base.text;
    std::copy_n(base.positions.begin(), nt * 3, out.positions.begin());
    std::copy(textTags.begin(), textTags.end(), out.tags.begin());
    std::copy(base.positions.begin() + nt * 3, base.positions.end(),
              out.positions.begin() + (nt + extra) * 3);
    std::copy(base.tags.begin() + nt, base.tags.end(), out.tags.begin() + nt + extra);
    double end = nt - 5.0 / 3.0;
    for (int t = 0; t < frames; ++t)
        end += (5.0 / 3.0) * (t % 5 == 0 ? 1 : 4);
    for (size_t a = 0; a < anchors.size(); ++a)
        for (int i = 0; i < perFrame; ++i) {
            int row = nt + a * perFrame + i;
            out.video.push_back(row);
            out.tags[row] = 0;
            out.positions[row * 3] = anchors[a] == 0 ? float(nt) : float(end);
            out.positions[row * 3 + 1] = base.positions[(base.video[0] + i) * 3 + 1];
            out.positions[row * 3 + 2] = base.positions[(base.video[0] + i) * 3 + 2];
        }
    for (int row : base.video)
        out.video.push_back(row + extra);
    for (int row : base.audio)
        out.audio.push_back(row + extra);
    return out;
}
} // namespace minimax_h3
} // namespace fastllm
