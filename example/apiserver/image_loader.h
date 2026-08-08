#pragma once

#include <string>
#include <vector>

struct OpenAIDecodedImage {
    int width = 0;
    int height = 0;
    std::vector<float> rgb;
};

bool LoadOpenAIImageUrl(const std::string &url,
                        OpenAIDecodedImage &image,
                        std::string &error);
