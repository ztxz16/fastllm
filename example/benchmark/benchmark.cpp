//
// Created by huangyuyang on 6/9/23.
//

#include "model.h"
#include "utils.h"
#include "fstream"

struct BenchmarkConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    int threads = 4; // 使用的线程数
    int limit = -1; // 输出token数限制，如果 < 0 则代表无限制
    int batch = -1; // batch数, -1时使用文件中的行数作为batch
    std::string file; // 输入文件
    std::string output; // 输出文件，如果不设定则输出到屏幕
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--limit> <args>:          输出token数限制" << std::endl;
    std::cout << "<-b|--batch> <args>:          batch数"      << std::endl;
    std::cout << "<-f|--file> <args>:           输入文件，文件中每行一个prompt，如果行数不足batch则用之前的prompt补充"      << std::endl;
}

void ParseArgs(int argc, char **argv, BenchmarkConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        }
        else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--limit") {
            config.limit = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-b" || sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-f" || sargv[i] == "--file") {
            config.file = sargv[++i];
        } else if (sargv[i] == "-o" || sargv[i] == "--output") {
            config.output = sargv[++i];
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    BenchmarkConfig config;
    ParseArgs(argc, argv, config);
    fastllm::SetThreads(config.threads);
    auto model = fastllm::CreateLLMModelFromFile(config.path);
    fastllm::GenerationConfig generationConfig;
    generationConfig.output_token_limit = config.limit;

    fastllm::PrintInstructionInfo();
    std::vector <std::string> inputs;
    if (config.file != "") {
        std::ifstream finputs(config.file, std::ios::in);
        while (true) {
            std::string input = "";
            std::getline(finputs, input);
            if (input == "") {
                break;
            } else {
                inputs.push_back(input);
            }
        }
    } else {
        inputs.push_back("Hello！");
    }
    if (config.batch < 0) {
        config.batch = inputs.size();
    }
    while (inputs.size() < config.batch) {
        inputs.push_back(inputs[rand() % inputs.size()]);
    }
    if (inputs.size() > config.batch && config.batch != -1) {
        inputs.resize(config.batch);
    }

    int promptTokenNum = 0;
    for (int i = 0; i < inputs.size(); i++) {
        inputs[i] = model->MakeInput("", 0, inputs[i]);
        promptTokenNum += model->weight.tokenizer.Encode(inputs[i]).Count(0);
    }

    std::vector <std::string> outputs;
    static int tokens = 0;
    auto st = std::chrono::system_clock::now();
    static auto promptTime = st;
    model->ResponseBatch(inputs, outputs, [](int index, std::vector<std::string> &contents) {
        if (index != -1) {
            if (index == 0) {
                promptTime = std::chrono::system_clock::now();
            } else {
                for (int i = 0; i < contents.size(); i++) {
                    tokens += (contents[i].size() > 0);
                }
            }
        }
    }, generationConfig);
    float promptSpend = fastllm::GetSpan(st, promptTime);
    float spend = fastllm::GetSpan(promptTime, std::chrono::system_clock::now());

    if (config.output != "") {
        FILE *fo = fopen(config.output.c_str(), "w");
        for (int i = 0; i < outputs.size(); i++) {
            fprintf(fo, "[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
        }
        fclose(fo);
    } else {
        for (int i = 0; i < outputs.size(); i++) {
            printf("[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
        }
    }

    printf("batch: %d\n", (int)inputs.size());
    printf("prompt token number = %d\n", promptTokenNum);
    printf("prompt use %f s\n", promptSpend);
    printf("prompt speed = %f tokens / s\n", (float)promptTokenNum / promptSpend);
    printf("output %d tokens\nuse %f s\nspeed = %f tokens / s\n", tokens, spend, tokens / spend);
    return 0;
}