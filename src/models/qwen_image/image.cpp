#include "models/qwen_image/runtime.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace fastllm {
namespace qwen_image {
namespace {
void check(bool ok, const char *message = "invalid PNG") {
    if (!ok)
        throw std::invalid_argument(message);
}
uint32_t be(const uint8_t *p) {
    return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
}
uint32_t crc(const uint8_t *p, size_t n) {
    uint32_t c = ~0u;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int b = 0; b < 8; ++b)
            c = (c >> 1) ^ (0xedb88320u & uint32_t(-int(c & 1)));
    }
    return ~c;
}
void append32(std::vector<uint8_t> &v, uint32_t x) {
    for (int i = 3; i >= 0; --i)
        v.push_back(uint8_t(x >> (i * 8)));
}
void chunk(std::vector<uint8_t> &out, const char *name, const std::vector<uint8_t> &bytes) {
    append32(out, bytes.size());
    size_t start = out.size();
    out.insert(out.end(), name, name + 4);
    out.insert(out.end(), bytes.begin(), bytes.end());
    append32(out, crc(out.data() + start, out.size() - start));
}
struct Bits {
    const std::vector<uint8_t> &data;
    size_t bit = 16;
    unsigned read(int n) {
        check(bit + n <= (data.size() - 4) * 8, "truncated DEFLATE stream");
        unsigned v = 0;
        for (int i = 0; i < n; ++i, ++bit)
            v |= ((data[bit / 8] >> (bit % 8)) & 1) << i;
        return v;
    }
};
struct Huffman {
    std::array<int, 16> counts{};
    std::vector<int> symbols;
    explicit Huffman(const std::vector<int> &lengths) {
        for (int n : lengths) {
            check(n >= 0 && n <= 15);
            ++counts[n];
        }
        int available = 1;
        for (int n = 1; n <= 15; ++n) {
            available = available * 2 - counts[n];
            check(available >= 0, "oversubscribed DEFLATE tree");
        }
        std::array<int, 16> offset{};
        for (int n = 1; n < 15; ++n)
            offset[n + 1] = offset[n] + counts[n];
        symbols.resize(lengths.size() - counts[0]);
        for (size_t i = 0; i < lengths.size(); ++i)
            if (lengths[i])
                symbols[offset[lengths[i]]++] = int(i);
    }
    int decode(Bits &bits) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len <= 15; ++len) {
            code |= bits.read(1);
            if (code - first < counts[len])
                return symbols[index + code - first];
            index += counts[len];
            first = (first + counts[len]) << 1;
            code <<= 1;
        }
        throw std::invalid_argument("invalid DEFLATE symbol");
    }
};
std::vector<uint8_t> inflate(const std::vector<uint8_t> &z, size_t size) {
    check(z.size() >= 6 && (z[0] & 15) == 8 && (z[0] >> 4) <= 7 && (unsigned(z[0]) * 256 + z[1]) % 31 == 0 &&
              !(z[1] & 32),
          "invalid zlib header");
    Bits bits{z};
    std::vector<uint8_t> out;
    out.reserve(size);
    bool last = false;
    const int lengthBase[] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                              31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    const int lengthExtra[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                               2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    const int distanceBase[] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    const int distanceExtra[] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    while (!last) {
        last = bits.read(1);
        unsigned type = bits.read(2);
        check(type != 3);
        if (type == 0) {
            bits.bit = (bits.bit + 7) & ~size_t(7);
            unsigned n = bits.read(16), inv = bits.read(16);
            check((n ^ inv) == 65535 && out.size() + n <= size);
            while (n--)
                out.push_back(bits.read(8));
            continue;
        }
        std::vector<int> lengths, distances;
        if (type == 1) {
            lengths.resize(288);
            for (int i = 0; i < 288; ++i)
                lengths[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
            distances.assign(32, 5);
        } else {
            int nl = bits.read(5) + 257, nd = bits.read(5) + 1, nc = bits.read(4) + 4;
            check(nl <= 286);
            const int canonicalOrder[] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::vector<int> cl(19);
            for (int i = 0; i < nc; ++i)
                cl[canonicalOrder[i]] = bits.read(3);
            Huffman tree(cl);
            std::vector<int> all;
            while (int(all.size()) < nl + nd) {
                int symbol = tree.decode(bits), repeat = 1, value = symbol;
                if (symbol == 16) {
                    check(!all.empty());
                    value = all.back();
                    repeat = bits.read(2) + 3;
                }
                if (symbol == 17) {
                    value = 0;
                    repeat = bits.read(3) + 3;
                }
                if (symbol == 18) {
                    value = 0;
                    repeat = bits.read(7) + 11;
                }
                check(int(all.size()) + repeat <= nl + nd);
                all.insert(all.end(), repeat, value);
            }
            lengths.assign(all.begin(), all.begin() + nl);
            distances.assign(all.begin() + nl, all.end());
        }
        check(lengths[256] != 0);
        Huffman lit(lengths), dist(distances);
        for (;;) {
            int symbol = lit.decode(bits);
            if (symbol == 256)
                break;
            if (symbol < 256) {
                check(out.size() < size);
                out.push_back(symbol);
                continue;
            }
            check(symbol <= 285);
            int li = symbol - 257, length = lengthBase[li] + bits.read(lengthExtra[li]);
            int ds = dist.decode(bits);
            check(ds < 30);
            size_t distance = distanceBase[ds] + bits.read(distanceExtra[ds]);
            check(distance <= out.size() && out.size() + length <= size);
            for (int i = 0; i < length; ++i)
                out.push_back(out[out.size() - distance]);
        }
    }
    check(out.size() == size, "PNG decompressed size mismatch");
    uint32_t a = 1, b = 0;
    for (auto c : out) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    check(be(z.data() + z.size() - 4) == ((b << 16) | a), "PNG Adler checksum mismatch");
    return out;
}
int paeth(int a, int b, int c) {
    int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}
double lanczos(double x) {
    x = std::abs(x);
    if (x == 0)
        return 1;
    if (x >= 3)
        return 0;
    x *= 3.14159265358979323846;
    return std::sin(x) * std::sin(x / 3) / (x * x / 3);
}
} // namespace
std::vector<uint8_t> encodePNG(const Image &image) {
    if (image.width < 1 || image.height < 1 || image.rgba.size() != size_t(image.width) * image.height * 4)
        throw std::invalid_argument("invalid RGBA image");
    std::vector<uint8_t> raw;
    raw.reserve((size_t(image.width) * 4 + 1) * image.height);
    for (int y = 0; y < image.height; ++y) {
        raw.push_back(0);
        auto p = image.rgba.begin() + size_t(y) * image.width * 4;
        raw.insert(raw.end(), p, p + image.width * 4);
    }
    // PNG's stored DEFLATE blocks need no image codec or zlib dependency.
    std::vector<uint8_t> z{0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (size_t p = 0; p < raw.size();) {
        uint16_t n = std::min<size_t>(65535, raw.size() - p);
        z.push_back(p + n == raw.size() ? 1 : 0);
        z.push_back(n & 255);
        z.push_back(n >> 8);
        uint16_t inv = ~n;
        z.push_back(inv & 255);
        z.push_back(inv >> 8);
        z.insert(z.end(), raw.begin() + p, raw.begin() + p + n);
        for (size_t j = p; j < p + n; ++j) {
            a = (a + raw[j]) % 65521;
            b = (b + a) % 65521;
        }
        p += n;
    }
    append32(z, (b << 16) | a);
    std::vector<uint8_t> out{137, 80, 78, 71, 13, 10, 26, 10}, header;
    append32(header, image.width);
    append32(header, image.height);
    header.insert(header.end(), {8, 6, 0, 0, 0});
    chunk(out, "IHDR", header);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    return out;
}
Image decodePNG(const std::string &bytes) {
    auto p = reinterpret_cast<const uint8_t *>(bytes.data());
    check(bytes.size() >= 33 && std::memcmp(p, "\x89PNG\r\n\x1a\n", 8) == 0,
          "upload a PNG image (the demo converts other formats)");
    Image image;
    int channels = 0, type = -1;
    bool ended = false, header = false;
    std::vector<uint8_t> compressed, palette;
    std::vector<uint8_t> transparent;
    for (size_t pos = 8; pos + 12 <= bytes.size();) {
        size_t n = be(p + pos);
        check(n <= bytes.size() - pos - 12);
        const uint8_t *name = p + pos + 4, *data = name + 4;
        check(crc(name, n + 4) == be(data + n), "PNG CRC mismatch");
        if (std::memcmp(name, "IHDR", 4) == 0) {
            check(!header && pos == 8 && n == 13);
            header = true;
            uint32_t w = be(data), h = be(data + 4);
            check(w >= 1 && h >= 1 && w <= 8192 && h <= 8192 && uint64_t(w) * h <= 16777216,
                  "PNG exceeds 16 megapixels");
            image.width = w;
            image.height = h;
            type = data[9];
            channels = type == 0 ? 1 : type == 2 ? 3 : type == 3 ? 1 : type == 4 ? 2 : type == 6 ? 4 : 0;
            check(data[8] == 8 && channels && data[10] == 0 && data[11] == 0 && data[12] == 0,
                  "PNG requires 8-bit non-interlaced pixels");
        } else if (std::memcmp(name, "IDAT", 4) == 0) {
            check(header);
            compressed.insert(compressed.end(), data, data + n);
        } else if (std::memcmp(name, "PLTE", 4) == 0) {
            check(n && n <= 768 && n % 3 == 0);
            palette.assign(data, data + n);
        } else if (std::memcmp(name, "tRNS", 4) == 0) {
            transparent.assign(data, data + n);
        } else if (std::memcmp(name, "IEND", 4) == 0) {
            check(n == 0);
            ended = true;
            break;
        } else
            check(name[0] & 32, "unsupported critical PNG chunk");
        pos += n + 12;
    }
    check(header && ended);
    size_t stride = size_t(image.width) * channels;
    auto raw = inflate(compressed, (stride + 1) * image.height);
    std::vector<uint8_t> pixels(stride * image.height);
    for (int y = 0; y < image.height; ++y) {
        int filter = raw[size_t(y) * (stride + 1)];
        check(filter <= 4);
        for (size_t x = 0; x < stride; ++x) {
            int a = x >= size_t(channels) ? pixels[size_t(y) * stride + x - channels] : 0,
                b = y ? pixels[size_t(y - 1) * stride + x] : 0,
                c = y && x >= size_t(channels) ? pixels[size_t(y - 1) * stride + x - channels] : 0;
            pixels[size_t(y) * stride + x] =
                raw[size_t(y) * (stride + 1) + x + 1] + (filter == 1   ? a
                                                         : filter == 2 ? b
                                                         : filter == 3 ? (a + b) / 2
                                                         : filter == 4 ? paeth(a, b, c)
                                                                       : 0);
        }
    }
    if (type == 3)
        check(!palette.empty() && transparent.size() <= palette.size() / 3);
    if (type == 0 && !transparent.empty())
        check(transparent.size() == 2);
    if (type == 2 && !transparent.empty())
        check(transparent.size() == 6);
    image.rgba.resize(size_t(image.width) * image.height * 4);
    for (size_t i = 0; i < image.rgba.size() / 4; ++i) {
        auto s = pixels.data() + i * channels, d = image.rgba.data() + i * 4;
        d[3] = 255;
        if (type == 3) {
            check(size_t(s[0]) * 3 + 2 < palette.size());
            for (int c = 0; c < 3; ++c)
                d[c] = palette[s[0] * 3 + c];
            if (s[0] < transparent.size())
                d[3] = transparent[s[0]];
        } else if (type == 0 || type == 4) {
            d[0] = d[1] = d[2] = s[0];
            if (type == 4)
                d[3] = s[1];
            else if (transparent.size() == 2 && transparent[0] == 0 && s[0] == transparent[1])
                d[3] = 0;
        } else {
            std::copy(s, s + 3, d);
            if (type == 6)
                d[3] = s[3];
            else if (transparent.size() == 6 && !transparent[0] && !transparent[2] && !transparent[4] &&
                     s[0] == transparent[1] && s[1] == transparent[3] && s[2] == transparent[5])
                d[3] = 0;
        }
    }
    return image;
}
Image resizeImage(const Image &source, int width, int height) {
    check(source.width > 0 && source.height > 0 &&
              source.rgba.size() == size_t(source.width) * source.height * 4,
          "invalid source image");
    check(width > 0 && height > 0 && width <= 4096 && height <= 4096,
          "reference aspect ratio is too extreme");
    if (width == source.width && height == source.height)
        return source;
    // Pillow's RGBA Lanczos uses premultiplied 8-bit channels and separable filters.
    Image current = source;
    for (size_t i = 0; i < current.rgba.size(); i += 4)
        for (int c = 0; c < 3; ++c)
            current.rgba[i + c] = (int(current.rgba[i + c]) * current.rgba[i + 3] + 127) / 255;
    for (int axis = 0; axis < 2; ++axis) {
        int from = axis ? current.height : current.width, to = axis ? height : width;
        if (from == to)
            continue;
        Image result{axis ? current.width : width, axis ? height : current.height, {}};
        result.rgba.resize(size_t(result.width) * result.height * 4);
        double scale = double(from) / to, filterScale = std::max(1., scale), support = 3 * filterScale;
        for (int j = 0; j < to; ++j) {
            double center = (j + .5) * scale;
            int begin = std::max(0, int(center - support + .5)),
                end = std::min(from, int(center + support + .5));
            std::vector<double> weights(end - begin);
            double sum = 0;
            for (int i = begin; i < end; ++i) {
                weights[i - begin] = lanczos((i - center + .5) / filterScale);
                sum += weights[i - begin];
            }
            std::vector<int> fixed(weights.size());
            for (size_t i = 0; i < fixed.size(); ++i) {
                double v = weights[i] / sum * (1 << 22);
                fixed[i] = int(v + (v >= 0 ? .5 : -.5));
            }
            int outer = axis ? result.width : result.height;
            for (int k = 0; k < outer; ++k)
                for (int c = 0; c < 4; ++c) {
                    int64_t value = 1 << 21;
                    for (int i = begin; i < end; ++i)
                        value += int64_t(current.rgba[(axis ? size_t(i) * current.width + k
                                                            : size_t(k) * current.width + i) *
                                                          4 +
                                                      c]) *
                                 fixed[i - begin];
                    result
                        .rgba[(axis ? size_t(j) * result.width + k : size_t(k) * result.width + j) * 4 + c] =
                        std::clamp<int64_t>(value >> 22, 0, 255);
                }
        }
        current = std::move(result);
    }
    for (size_t i = 0; i < current.rgba.size(); i += 4)
        if (current.rgba[i + 3] != 0 && current.rgba[i + 3] != 255)
            for (int c = 0; c < 3; ++c)
                current.rgba[i + c] = std::min(255, int(current.rgba[i + c]) * 255 / current.rgba[i + 3]);
    return current;
}
} // namespace qwen_image
} // namespace fastllm
