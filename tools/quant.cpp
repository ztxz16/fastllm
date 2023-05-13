//
// Created by huangyuyang on 5/13/23.
//

#include <cstdio>
#include <cstring>
#include <iostream>
#include <getopt.h>

#include "moss.h"
#include "chatglm.h"

struct QuantConfig {
    std::string model = "chatglm"; // 模型类型, chatglm或moss
    std::string path; // 模型文件路径
    std::string output; // 输出文件路径
    int bits; // 量化位数
};

static struct option long_options[] = {
        {"help",               no_argument,       nullptr, 'h'},
        {"model",              required_argument, nullptr, 'm'},
        {"path",               required_argument, nullptr, 'p'},
        {"bits",               required_argument, nullptr, 'b'},
        {"output",             required_argument, nullptr, 'o'},
        {nullptr, 0,                              nullptr, 0},
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                      显示帮助" << std::endl;
    std::cout << "<-m|--model> <args>:              模型类型，默认为chatglm, 可以设置为chatglm, moss" << std::endl;
    std::cout << "<-p|--path> <args>:               模型文件的路径" << std::endl;
    std::cout << "<-b|--bits> <args>:               量化位数" << std::endl;
    std::cout << "<-o|--output> <args>:             输出文件路径" << std::endl;
}

void ParseArgs(int argc, char **argv, QuantConfig &config) {
    int opt;
    int option_index = 0;
    const char *opt_string = "h:m:p:b:o:";

    while ((opt = getopt_long_only(argc, argv, opt_string, long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                Usage();
                exit (0);
            case 'm':
                config.model = argv[optind - 1];
                break;
            case 'p':
                config.path = argv[optind - 1];
                break;
            case 'b':
                config.bits = atoi(argv[optind - 1]);
                break;
            case 'o':
                config.output = argv[optind - 1];
                break;
            default:
                Usage();
                exit (-1);
        }
    }
}

int main(int argc, char **argv) {
    QuantConfig config;
    ParseArgs(argc, argv, config);

    if (config.model == "moss") {
        fastllm::MOSSModel moss;
        moss.LoadFromFile(config.path);
        moss.SaveLowBitModel(config.output, config.bits);
    } else if (config.model == "chatglm") {
        fastllm::ChatGLMModel chatGlm;
        chatGlm.LoadFromFile(config.path);
        chatGlm.SaveLowBitModel(config.output, config.bits);
    } else {
        Usage();
        exit(-1);
    }

    return 0;
}