#include "image_loader.h"

#include <algorithm>
#include <cctype>
#include <csetjmp>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef FASTLLM_APISERVER_OPENSSL
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "../webui/httplib.h"

#ifdef FASTLLM_APISERVER_IMAGE_DECODERS
#include <jpeglib.h>
#include <png.h>
#endif

namespace {
    constexpr size_t kMaxImageBytes = 32u * 1024u * 1024u;
    constexpr uint64_t kMaxDecodedPixels = 100000000u;

    bool DecodeBase64(const std::string &encoded,
                      std::vector<uint8_t> &decoded,
                      std::string &error) {
        static const int8_t table[256] = {
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
            52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
            -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
            15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
            -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
            -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
        };
        decoded.clear();
        if (encoded.empty() || encoded.size() % 4 != 0 ||
            encoded.size() > (kMaxImageBytes * 4 / 3 + 8)) {
            error = "image data URL has invalid or oversized base64 payload";
            return false;
        }
        decoded.reserve(encoded.size() / 4 * 3);
        for (size_t offset = 0; offset < encoded.size(); offset += 4) {
            int values[4];
            for (int i = 0; i < 4; i++) {
                values[i] = table[static_cast<uint8_t>(encoded[offset + i])];
            }
            const bool finalBlock = offset + 4 == encoded.size();
            if (values[0] < 0 || values[1] < 0 || values[2] == -1 ||
                values[3] == -1 || values[0] == -2 || values[1] == -2 ||
                (!finalBlock && (values[2] == -2 || values[3] == -2)) ||
                (values[2] == -2 && values[3] != -2)) {
                error = "image data URL contains invalid base64";
                decoded.clear();
                return false;
            }
            const uint32_t packed =
                (static_cast<uint32_t>(values[0]) << 18) |
                (static_cast<uint32_t>(values[1]) << 12) |
                (static_cast<uint32_t>(std::max(0, values[2])) << 6) |
                static_cast<uint32_t>(std::max(0, values[3]));
            decoded.push_back(static_cast<uint8_t>(packed >> 16));
            if (values[2] != -2) {
                decoded.push_back(static_cast<uint8_t>(packed >> 8));
            }
            if (values[3] != -2) {
                decoded.push_back(static_cast<uint8_t>(packed));
            }
        }
        if (decoded.size() > kMaxImageBytes) {
            error = "image payload exceeds 32 MiB";
            decoded.clear();
            return false;
        }
        return true;
    }

    bool ReadImageBytes(const std::string &url,
                        std::vector<uint8_t> &bytes,
                        std::string &error) {
        if (url.rfind("data:", 0) == 0) {
            const size_t comma = url.find(',');
            if (comma == std::string::npos ||
                url.substr(0, comma).find(";base64") == std::string::npos) {
                error = "image data URL must use base64 encoding";
                return false;
            }
            return DecodeBase64(url.substr(comma + 1), bytes, error);
        }

        const bool isHttp = url.rfind("http://", 0) == 0;
        const bool isHttps = url.rfind("https://", 0) == 0;
        if (!isHttp && !isHttps) {
            error = "image_url.url must use data:, http://, or https://";
            return false;
        }
#ifndef FASTLLM_APISERVER_OPENSSL
        if (isHttps) {
            error = "HTTPS image loading is unavailable because OpenSSL was not found at build time";
            return false;
        }
#endif
        const size_t authorityStart = url.find("://") + 3;
        size_t pathStart = url.find_first_of("/?#", authorityStart);
        std::string base = pathStart == std::string::npos
            ? url : url.substr(0, pathStart);
        std::string path = "/";
        if (pathStart != std::string::npos) {
            path = url[pathStart] == '/' ? url.substr(pathStart)
                                         : "/" + url.substr(pathStart);
        }
        const size_t fragment = path.find('#');
        if (fragment != std::string::npos) {
            path.resize(fragment);
        }

        httplib::Client client(base);
        if (!client.is_valid()) {
            error = "invalid image URL";
            return false;
        }
        client.set_connection_timeout(10, 0);
        client.set_read_timeout(20, 0);
        client.set_write_timeout(10, 0);
        client.set_follow_location(true);
#ifdef FASTLLM_APISERVER_OPENSSL
        if (isHttps) {
            client.enable_server_certificate_verification(true);
        }
#endif
        bytes.clear();
        auto result = client.Get(path, [&](const char *data, size_t length) {
            if (length > kMaxImageBytes - bytes.size()) {
                return false;
            }
            bytes.insert(bytes.end(), data, data + length);
            return true;
        });
        if (!result) {
            error = bytes.size() >= kMaxImageBytes
                ? "image payload exceeds 32 MiB"
                : "failed to download image: " + httplib::to_string(result.error());
            bytes.clear();
            return false;
        }
        if (result->status < 200 || result->status >= 300) {
            error = "image download returned HTTP " +
                    std::to_string(result->status);
            bytes.clear();
            return false;
        }
        if (bytes.empty()) {
            error = "image download returned an empty body";
            return false;
        }
        return true;
    }

#ifdef FASTLLM_APISERVER_IMAGE_DECODERS
    bool ValidateDimensions(uint64_t width, uint64_t height,
                            std::string &error) {
        if (width == 0 || height == 0 ||
            width > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            height > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            width > kMaxDecodedPixels / height) {
            error = "decoded image dimensions are invalid or too large";
            return false;
        }
        return true;
    }

    bool DecodePng(const std::vector<uint8_t> &bytes,
                   OpenAIDecodedImage &image,
                   std::string &error) {
        png_image png = {};
        png.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_memory(&png, bytes.data(), bytes.size())) {
            error = "failed to read PNG header: " + std::string(png.message);
            return false;
        }
        if (!ValidateDimensions(png.width, png.height, error)) {
            png_image_free(&png);
            return false;
        }
        png.format = PNG_FORMAT_RGB;
        std::vector<uint8_t> rgb(PNG_IMAGE_SIZE(png));
        if (!png_image_finish_read(&png, nullptr, rgb.data(), 0, nullptr)) {
            error = "failed to decode PNG: " + std::string(png.message);
            png_image_free(&png);
            return false;
        }
        png_image_free(&png);
        OpenAIDecodedImage decoded;
        decoded.width = static_cast<int>(png.width);
        decoded.height = static_cast<int>(png.height);
        decoded.rgb.assign(rgb.begin(), rgb.end());
        image = std::move(decoded);
        return true;
    }

    struct JpegErrorManager {
        jpeg_error_mgr base;
        std::jmp_buf jump;
        char message[JMSG_LENGTH_MAX] = {};
    };

    void JpegErrorExit(j_common_ptr info) {
        auto *manager = reinterpret_cast<JpegErrorManager*>(info->err);
        (*info->err->format_message)(info, manager->message);
        std::longjmp(manager->jump, 1);
    }

    bool DecodeJpeg(const std::vector<uint8_t> &bytes,
                    OpenAIDecodedImage &image,
                    std::string &error) {
        jpeg_decompress_struct decoder = {};
        JpegErrorManager manager;
        decoder.err = jpeg_std_error(&manager.base);
        manager.base.error_exit = JpegErrorExit;
        if (setjmp(manager.jump)) {
            jpeg_destroy_decompress(&decoder);
            error = "failed to decode JPEG: " + std::string(manager.message);
            return false;
        }
        jpeg_create_decompress(&decoder);
        jpeg_mem_src(&decoder, bytes.data(), bytes.size());
        jpeg_read_header(&decoder, TRUE);
        decoder.out_color_space = JCS_RGB;
        jpeg_start_decompress(&decoder);
        if (!ValidateDimensions(decoder.output_width, decoder.output_height,
                                error) || decoder.output_components != 3) {
            jpeg_destroy_decompress(&decoder);
            if (error.empty()) {
                error = "JPEG decoder did not produce RGB pixels";
            }
            return false;
        }
        OpenAIDecodedImage decoded;
        decoded.width = static_cast<int>(decoder.output_width);
        decoded.height = static_cast<int>(decoder.output_height);
        decoded.rgb.resize(static_cast<size_t>(decoded.width) *
                           decoded.height * 3);
        while (decoder.output_scanline < decoder.output_height) {
            JSAMPROW row = reinterpret_cast<JSAMPROW>(
                decoded.rgb.data() + static_cast<size_t>(decoder.output_scanline) *
                decoded.width * 3);
            jpeg_read_scanlines(&decoder, &row, 1);
        }
        jpeg_finish_decompress(&decoder);
        jpeg_destroy_decompress(&decoder);
        image = std::move(decoded);
        return true;
    }
#endif
}

bool LoadOpenAIImageUrl(const std::string &url,
                        OpenAIDecodedImage &image,
                        std::string &error) {
    image = OpenAIDecodedImage();
    error.clear();
    std::vector<uint8_t> bytes;
    if (!ReadImageBytes(url, bytes, error)) {
        return false;
    }
#ifdef FASTLLM_APISERVER_IMAGE_DECODERS
    static const uint8_t pngMagic[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (bytes.size() >= sizeof(pngMagic) &&
        std::memcmp(bytes.data(), pngMagic, sizeof(pngMagic)) == 0) {
        return DecodePng(bytes, image, error);
    }
    if (bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8) {
        return DecodeJpeg(bytes, image, error);
    }
    error = "unsupported image format; only PNG and JPEG are accepted";
    return false;
#else
    (void)bytes;
    error = "PNG/JPEG image decoding is unavailable in this build";
    return false;
#endif
}
