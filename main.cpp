#include <cstdio>
#include <cstring>
#include <iostream>
#include <getopt.h>

#include "moss.h"
#include "chatglm.h"

struct RunConfig {
    std::string model = "chatglm"; // 模型类型, chatglm或moss
    std::string path; // 模型文件路径
    int threads = 4; // 使用的线程数
};

static struct option long_options[] = {
        {"help",               no_argument,       nullptr, 'h'},
        {"model",              required_argument, nullptr, 'm'},
        {"path",               required_argument, nullptr, 'p'},
        {"threads",            required_argument, nullptr, 't'},
        {nullptr, 0,                              nullptr, 0},
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                      显示帮助" << std::endl;
    std::cout << "<-m|--model> <args>:              模型类型，默认为chatglm, 可以设置为chatglm, moss" << std::endl;
    std::cout << "<-p|--path> <args>:               模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:            使用的线程数量" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config) {
    int opt;
    int option_index = 0;
    const char *opt_string = "h:m:p:t:";

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
            case 't':
                config.threads = atoi(argv[optind - 1]);
                break;
            default:
                Usage();
                exit (-1);
        }
    }
}

int main(int argc, char **argv) {
    RunConfig config;
    ParseArgs(argc, argv, config);
    fastllm::SetThreads(config.threads);

    if (config.model == "moss") {
        fastllm::MOSSModel moss;
        moss.LoadFromFile(config.path);

        while (true) {
            printf("用户: ");
            std::string input;
            std::getline(std::cin, input);
            if (input == "stop") {
                break;
            }
            std::string ret = moss.Response("You are an AI assistant whose name is MOSS. <|Human|>: " + input + "<eoh>");
        }
    } else if (config.model == "chatglm") {
        fastllm::ChatGLMModel chatGlm;
        chatGlm.LoadFromFile(config.path);

        int round = 0;
        std::string history;
        while (true) {
            printf("用户: ");
            std::string input;
            std::getline(std::cin, input);
            if (input == "stop") {
                break;
            }
            history += ("[Round " + std::to_string(round++) + "]\n问：" + input);
            auto prompt = round > 1 ? history : input;
            std::cout << prompt << std::endl;
            printf("ChatGLM: ");
            std::string ret = chatGlm.Response(prompt);
            history += ("\n答：" + ret + "\n");
        }

        //chatGlm.weight.SaveLowBitModel("/root/chatglm-6b-int4.bin", 4);
    } else {
        Usage();
        exit(-1);
    }

    return 0;
}
